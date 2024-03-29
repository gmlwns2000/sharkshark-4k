from dataclasses import dataclass
from queue import Empty
import queue
import cv2, time, os
import numpy as np
import torch.multiprocessing as mp

from ..util.profiler import Profiler
from .twitch_realtime_handler import (
    TwitchAudioGrabber,
    TwitchImageGrabber
)
from .youtube_recoder.image_recoder import YoutubeImageRecoder

TW_SHARK = 'https://twitch.tv/tizmtizm'
TW_MARU = 'https://www.twitch.tv/maoruya'
TW_PIANOCAT = 'https://www.twitch.tv/pianocatvr'
TW_RUMYONG = 'https://www.twitch.tv/lumyon3'
TW_MAOU = 'https://www.twitch.tv/mawang0216'
TW_DALTA = 'https://www.twitch.tv/dalta_23'
TW_VIICHAN = 'https://www.twitch.tv/viichan6'
TW_ZURURU = 'https://www.twitch.tv/cotton__123'
TW_SHYLILY = 'https://www.twitch.tv/shylily'
TW_DANCINGSANA = 'https://www.twitch.tv/dancingshana'

@dataclass
class RecoderEntry:
    index: int
    audio_segment: np.ndarray
    frames: np.ndarray
    fps: float
    profiler: Profiler

class TwitchRecoder:
    def __init__(self, target_url=TW_MARU, batch_sec=1, fps=24, on_queue=None, quality='1080p', buffer_size=1, audio_skip=0):
        assert isinstance(batch_sec, int)
        self.url = target_url
        self.batch_sec = batch_sec
        self.fps = fps
        self.queue = mp.Queue(maxsize=buffer_size)
        self.cmd_queue = mp.Queue()
        self.on_queue = on_queue
        self.output_shape = None
        self.frame_count = 0
        self.quality = quality
        self.audio_skip = audio_skip
        if(audio_skip > 0):
            self.audio_queue = mp.Queue(maxsize=audio_skip)
    
    def __getstate__(self):
        state = self.__dict__.copy()
        if 'proc' in state:
            del state["proc"]
        return state

    def proc_main(self):
        print('TwitchRecoder: TwitchImageGrabber init')
        if 'youtube' in self.url:
            image_grabber = YoutubeImageRecoder(
                url=self.url,
                quality=self.quality,
                rate=self.fps,
            )
        else:
            image_grabber = TwitchImageGrabber(
                twitch_url=self.url,
                quality=self.quality,  # quality of the stream could be ["160p", "360p", "480p", "720p", "720p60", "1080p", "1080p60"]
                blocking=True,
                rate=self.fps  # frame per rate (fps)
            )
        
        # change to a stream that is actually online
        print('TwitchRecoder: TwitchAudioGrabber init')
        audio_grabber = TwitchAudioGrabber(
            twitch_url=self.url,
            blocking=True,  # wait until a segment is available
            segment_length=int(self.batch_sec),  # segment length in seconds
            rate=44100,  # sampling rate of the audio
            channels=2,  # number of channels
            dtype=np.float32  # quality of the audio could be [np.int16, np.int32, np.float32, np.float64]
        )

        t = time.time()
        t_sum = []
        index = 0
        while True:
            try:
                cmd = self.cmd_queue.get_nowait()
                if cmd == 'exit':
                    print('TwitchRecoder: Get exit')
                    self.cmd_queue.close()
                    break
                else: raise Exception()
            except Empty:
                pass
            
            #print('ff')
            frames = []
            reader_eof = False
            for i in range(self.batch_sec * self.fps):
                frame = image_grabber.grab()
                if frame is None: 
                    print('frame recoded none EOF')
                    reader_eof = True
                    break
                    #raise Exception('frame recodered None!')
                # print(f'grabbed {self.frame_count}, {frame[0,0,0]}')
                if self.output_shape is not None:
                    frame = cv2.resize(frame, dsize=[self.output_shape[1], self.output_shape[0]], interpolation=cv2.INTER_AREA)
                    frame = cv2.putText(frame, f"Received: {self.frame_count} frames", (10, 32), cv2.FONT_HERSHEY_PLAIN, 0.5, (255,0,0), 1)
                    self.frame_count += 1
                frames.append(frame)
            
            if reader_eof:
                entry = RecoderEntry(
                    index=index,
                    audio_segment=None, #(22000,2)
                    frames=None, #(24, 1080, 1920,3) -> (24, 2160, 3840, 3)
                    fps=self.fps,
                    profiler=Profiler()
                )
                entry.profiler.start('recoder.output')
                if self.on_queue is not None:
                    self.on_queue(entry)
                else:
                    try:
                        self.queue.put_nowait(entry)
                    except queue.Full:
                        print(f'TwitchRecoder: output queue is full. Is consumer too slow?')
                break
            
            if len(frames) == 0:
                print(f'TwitchRecoder: frame does not recorded...')
                continue
            
            #print('f')
            audio_segment = audio_grabber.grab()
            if self.audio_skip > 0:
                while self.audio_queue.qsize() < self.audio_skip:
                    self.audio_queue.put(audio_segment.copy())
                audio_segment = self.audio_queue.get()
            
            frames = np.stack(frames, axis=0)
            t_sum.append(time.time()-t)
            if len(t_sum) > 100:
                t_sum.pop(0)
            t_avg = sum(t_sum)/len(t_sum)
            print(f'TwitchRecoder: batch[{index}] captured took average {t_avg:.2f} sec. Audio[{audio_segment.shape}] Video[{frames.shape}]')
            t = time.time()
            entry = RecoderEntry(
                index=index,
                audio_segment=audio_segment, #(22000,2)
                frames=frames, #(24, 1080, 1920,3) -> (24, 2160, 3840, 3)
                fps=self.fps,
                profiler=Profiler()
            )
            entry.profiler.start('recoder.output')
            if self.on_queue is not None:
                self.on_queue(entry)
            else:
                try:
                    self.queue.put_nowait(entry)
                except queue.Full:
                    print(f'TwitchRecoder: output queue is full. Is consumer too slow?')
            index += 1
        
        print('TwitchRecoder: try term img')
        image_grabber.terminate()
        print('TwitchRecoder: try term audio')
        audio_grabber.terminate()
        print('TwitchRecoder: exit subproc')

        os.kill(os.getpid(), 9)
    
    def start(self):
        self.proc = mp.Process(target=self.proc_main, daemon=True)
        self.proc.start()
    
    def get(self) -> RecoderEntry:
        return self.queue.get()
    
    def stop(self):
        self.cmd_queue.put("exit")
        self.queue.close()
        print('TwitchRecoder: joining all subprocs')
        self.join()
        print('TwitchRecoder: joined subprocs')

    def join(self):
        self.proc.join()

if __name__ == '__main__':
    print('asdf')
    recoder = TwitchRecoder(target_url=TW_MAOU, quality='1080p60')
    recoder.start()

    time.sleep(3)

    if not os.path.exists('./saves/frames/'): os.mkdir('./saves/frames/')
    j = 0
    for i in range(10):
        batch = recoder.queue.get(timeout=30) #type: RecoderEntry
        for k in range(batch.frames.shape[0]):
            cv2.imwrite(f"saves/frames/{j:04}.png", cv2.cvtColor(batch.frames[k], cv2.COLOR_RGB2BGR))
            j += 1
        print(f"{i} batch get. {batch.frames.shape}")
        
    recoder.stop()