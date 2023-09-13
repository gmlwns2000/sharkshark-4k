import io
import subprocess
import time
import numpy as np
import streamlink
from streamlink.session import Streamlink
import threading
import queue
import av
import requests
import shlex
from PIL import Image

class YoutubeImageRecoder:
    def __init__(self, url: str, quality: str, rate: int):
        self.url = url
        self.quality = quality
        self.rate = rate
        
        self.buffering_chunk_counts = 8
        self.safe_buffer_size = 500 * 1000
        self.chunk_size = 200000 # 500KB chunk
        self.current_position = 0
        self.buffer_position = 0
        self.content_size = 0
        self.buffers = {} # start_position -> bytes
        self.width, self.height = {
            '160p': (320, 160),
            '360p': (640, 360),
            '480p': (854, 480),
            '720p': (1280, 720),
            '720p48': (1280, 720),
            '720p60': (1280, 720),
            '1080p': (1920, 1080),
            '1080p60': (1920, 1080),
            'source': (1920,1080)
        }[quality]
        
        self.stream_url = self.get_stream_url()
        print(self.stream_url)
        self.parse_url()
        
        self.terminated = False
        
        self.thread = threading.Thread(target=self.proc_main, daemon=True)
        self.workers = [
            threading.Thread(target=self.worker_main, daemon=True) 
            for _ in range(self.buffering_chunk_counts)
        ]
        
        self.worker_queue = queue.Queue(maxsize=self.buffering_chunk_counts)
        self.buffer_queue = queue.Queue(maxsize=self.buffering_chunk_counts)
        self.frame_queue = queue.Queue()
        
        self.thread.start()
        [t.start() for t in self.workers]
        
        self.last_grab = 0
    
    def parse_url(self):
        from urllib.parse import urlparse, parse_qs
        parsed_url = urlparse(self.stream_url)
        self.content_size = int(parse_qs(parsed_url.query)['clen'][0])
        print('content size', self.content_size)
    
    def get_stream_url(self):
        assert self.url is not None

        try:
            sess = Streamlink()
            stream_hls = sess.streams(self.url)
            print("YoutubeImageRecoder: Found resolutions:", stream_hls.keys())
            if (self.quality not in stream_hls) and self.quality == 'audio_only':
                if "audio_opus" in stream_hls:
                    print("YoutubeImageRecoder: opus selected for audio stream")
                    self.quality = 'audio_opus'
                elif "audio" in stream_hls:
                    print("YoutubeImageRecoder: audio selected for audio stream")
                    self.quality = 'audio'
                else:
                    self.quality = '360p'
        except streamlink.exceptions.NoPluginError:
            raise ValueError(f"No stream availabe for {self.url}")
        
        if self.quality not in stream_hls:
            raise ValueError(f"The stream has not the given quality({self.quality}) but ({stream_hls.keys()})")
        #print(stream_hls)
        
        if hasattr(stream_hls[self.quality], 'substreams'):
            # print('substream', stream_hls[self.quality].substreams)
            return stream_hls[self.quality].substreams[0].url
        else:
            return stream_hls[self.quality].url
    
    def worker_main(self):
        print('worker start')
        while not self.terminated:
            start_position = self.worker_queue.get()
            end_position = start_position + self.chunk_size - 1
            content_url = self.stream_url+f'&range={int(start_position)}-{int(end_position)}'
            
            # print(f'worker: query {start_position}-{end_position}')
            
            res = requests.get(content_url)
            
            if res.status_code == 200:
                # print(f'worker: retcode {res.status_code} [{len(res.content)}]')
                self.buffer_queue.put((start_position, res.content))
            else:
                print(f'worker: retcode {res.status_code}')
                self.worker_queue.put(start_position)
                self.buffer_queue.put(None)
            
            if self.current_position >= self.content_size:
                break
    
    def proc_main(self):
        try:
            print('main thread start')
            pending_chunks = {}
            stream = io.BytesIO(b'0'*self.content_size)
            stream_position = 0
            container = None
            container_position = 0
            frame_index = 0
            adjusted_frame_index = 0
            while not self.terminated:
                while self.worker_queue.qsize() < self.worker_queue.maxsize and self.buffer_position < self.content_size:
                    self.worker_queue.put(self.buffer_position)
                    self.buffer_position += self.chunk_size

                while not self.buffer_queue.empty():
                    worker_result = self.buffer_queue.get()
                    if worker_result is None:
                        return
                    (chunk_position, chunk) = worker_result
                    pending_chunks[chunk_position] = chunk
                
                if self.current_position in pending_chunks:
                    chunk = pending_chunks[self.current_position]
                    del pending_chunks[self.current_position]
                    
                    stream.seek(stream_position)
                    stream.write(chunk)
                    stream_position += len(chunk)
                    assert stream_position == stream.tell()
                    stream.seek(container_position)
                    if container is None:
                        container = av.open(stream, mode='r')
                    
                    for packet in container.demux():
                        if packet.size < 1:
                            continue
                        for frame in packet.decode():
                            # print(frame, stream.tell(), container_position, stream_position)
                            img = frame.to_image() # type: Image
                            img = np.array(img.convert('RGB'))
                            # print(img.shape)
                            fps = container.streams.video[0].rate
                            # print(fps)
                            assert img is not None
                            assert self.rate <= fps
                            new_adjusted_frame_index = round(frame_index / fps * self.rate)
                            if new_adjusted_frame_index != adjusted_frame_index:
                                self.frame_queue.put(img)
                            adjusted_frame_index = new_adjusted_frame_index
                            frame_index += 1
                        container_position = stream.tell()
                        if container_position > (stream_position - self.safe_buffer_size):
                            break
                    container_position = stream.tell()
                    # print('demux done')
                    
                    self.current_position += self.chunk_size
                
                if self.current_position >= self.content_size:
                    print('main thread done')
                    self.frame_queue.put(None)
                    break
        except Exception as ex:
            self.terminated = True
            raise ex
        
    def join(self):
        self.thread.join()
    
    def grab(self) -> np.ndarray:
        # return single frame
        # while (time.time() - self.last_grab) < (1 / self.rate) * 0.9:
        #     time.sleep(0.001)
        # self.last_grab = time.time()
        return self.frame_queue.get()
    
    def terminate(self):
        self.terminated = True
        
if __name__ == '__main__':
    reader = YoutubeImageRecoder(
        url='https://www.youtube.com/watch?v=h5hMNF3kDm0',
        quality='1080p',
        rate=24,
    )
    n = 0
    t = time.time()
    while True:
        frame = reader.grab()
        if frame is None:
            print('main: decode finished')
            break
        print(f'main: frame {frame.shape} {n} {n/(time.time()-t)}')
        n += 1
    reader.join()