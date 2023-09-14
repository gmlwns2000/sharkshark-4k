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

def log(*args, **kwargs):
    print('YoutubeImageRecoder:', *args, **kwargs)

class YoutubeImageRecoder:
    def __init__(self, url: str, quality: str, rate: int):
        self.url = url
        self.quality = quality
        self.rate = rate
        
        self.num_workers = 16
        self.buffering_chunk_counts = self.num_workers
        self.safe_buffer_size = 500 * 1000 # 500KB is prefetch before start streaming
        self.chunk_size = 200000 # 200KB chunk
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
        log(self.stream_url)
        self.parse_url()
        
        self.terminated = False
        
        self.thread = threading.Thread(target=self.proc_main, daemon=True)
        self.workers = [
            threading.Thread(target=self.worker_main, daemon=True) 
            for _ in range(self.num_workers)
        ]
        
        self.worker_retry_queue = queue.Queue() # this should not be full...
        self.worker_queue = queue.Queue(maxsize=self.buffering_chunk_counts*2) # prefetch operations
        self.buffer_queue = queue.Queue(maxsize=self.buffering_chunk_counts)
        self.frame_queue = queue.Queue()
        
        self.thread.start()
        [t.start() for t in self.workers]
        
        self.last_grab = 0
    
    def parse_url(self):
        from urllib.parse import urlparse, parse_qs
        parsed_url = urlparse(self.stream_url)
        parsed_query = parse_qs(parsed_url.query)
        if 'clen' in parsed_query:
            self.content_size = int(parsed_query['clen'][0])
        else:
            res = requests.head(self.stream_url)
            self.content_size = int(res.headers['Content-Length'])
        log('content size', self.content_size)
    
    def get_stream_url(self):
        assert self.url is not None

        try:
            sess = Streamlink()
            stream_hls = sess.streams(self.url)
            log("Found resolutions:", stream_hls.keys())
            if (self.quality not in stream_hls) and self.quality == 'audio_only':
                if "audio_opus" in stream_hls:
                    log("opus selected for audio stream")
                    self.quality = 'audio_opus'
                elif "audio" in stream_hls:
                    log("audio selected for audio stream")
                    self.quality = 'audio'
                else:
                    self.quality = '360p'
        except streamlink.exceptions.NoPluginError:
            raise ValueError(f"No stream availabe for {self.url}")
        
        if self.quality not in stream_hls:
            raise ValueError(f"The stream has not the given quality({self.quality}) but ({stream_hls.keys()})")
        #log(stream_hls)
        
        if hasattr(stream_hls[self.quality], 'substreams'):
            # log('substream', stream_hls[self.quality].substreams)
            return stream_hls[self.quality].substreams[0].url
        else:
            return stream_hls[self.quality].url
    
    def worker_main(self):
        # log('worker: start')
        while not self.terminated:
            try:
                start_position = self.worker_retry_queue.get_nowait()
            except queue.Empty:
                start_position = None
            if start_position is None:
                # if nothing to retry
                try:
                    start_position = self.worker_queue.get(timeout=1.0)
                except (queue.Full, queue.Empty):
                    continue
            
            if start_position == None:
                break
            
            end_position = start_position + self.chunk_size - 1
            content_url = self.stream_url+f'&range={int(start_position)}-{int(end_position)}'
            
            # print(f'worker: query {start_position}-{end_position}')
            
            res = requests.get(content_url)
            
            if res.status_code == 200:
                # print(f'worker: retcode {res.status_code} [{len(res.content)}]')
                self.buffer_queue.put((start_position, res.content))
            else:
                log(f'worker: retcode {res.status_code}')
                self.worker_retry_queue.put(start_position)
            
            if self.current_position >= self.content_size:
                break
        # log('worker: done')
    
    def proc_main(self):
        try:
            log('main: thread start')
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
                    self.current_position += self.chunk_size
                    stream_position += len(chunk)
                    assert stream_position == stream.tell()
                    stream.seek(container_position)
                    if container is None:
                        container = av.open(stream, mode='r')
                    
                    for packet in container.demux():
                        if packet.size < 1:
                            continue
                        for frame in packet.decode():
                            if isinstance(frame, av.AudioFrame):
                                continue
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
                        if (container_position > (stream_position - self.safe_buffer_size)) and self.current_position < self.content_size:
                            break
                    container_position = stream.tell()
                
                if self.current_position >= self.content_size:
                    log('main: thread done')
                    for i in range(len(self.workers)):
                        self.worker_queue.put(None)
                    self.frame_queue.put(None)
                    break
        except Exception as ex:
            self.terminated = True
            raise ex
        
    def join(self):
        self.thread.join()
        for w in self.workers:
            w.join()
    
    def grab(self) -> np.ndarray:
        return self.frame_queue.get()
    
    def terminate(self):
        self.terminated = True
        
if __name__ == '__main__':
    reader = YoutubeImageRecoder(
        url='https://www.youtube.com/watch?v=h5hMNF3kDm0',
        quality='1080p',
        # url='https://www.youtube.com/watch?v=mFJzsgKoCNw',
        # quality='720p',
        rate=24,
    )
    n = 0
    t = time.time()
    while True:
        frame = reader.grab()
        if frame is None:
            print('main: decode finished')
            break
        print(f'main: frame {frame.shape} {n} {n/(time.time()-t)}\r', end='')
        n += 1
    reader.join()