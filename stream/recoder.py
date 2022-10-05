from dataclasses import dataclass
from queue import Empty
import queue
from twitch_realtime_handler import (TwitchAudioGrabber,
                                   TwitchImageGrabber)
import cv2, time, os
import numpy as np
import torch.multiprocessing as mp

from util.profiler import Profiler

TW_SHARK = 'https://twitch.tv/tizmtizm'
TW_MARU = 'https://www.twitch.tv/maoruya'
TW_PIANOCAT = 'https://www.twitch.tv/pianocatvr'
TW_RUMYONG = 'https://www.twitch.tv/lumyon3'
TW_MAOU = 'https://www.twitch.tv/mawang0216'

@dataclass
class RecoderEntry:
    index: int
    audio_segment: np.ndarray
    frames: np.ndarray
    fps: float
    profiler: Profiler

class TwitchRecoder:
    def __init__(self, target_url=TW_MARU, batch_sec=1, fps=24, on_queue=None):
        assert isinstance(batch_sec, int)
        self.url = target_url
        self.batch_sec = batch_sec
        self.fps = fps
        self.queue = mp.Queue(maxsize=1)
        self.cmd_queue = mp.Queue()
        self.on_queue = on_queue
        self.output_shape = None
    
    def __getstate__(self):
        state = self.__dict__.copy()
        if 'proc' in state:
            del state["proc"]
        return state

    def proc(self):
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

        print('TwitchRecoder: TwitchImageGrabber init')
        image_grabber = TwitchImageGrabber(
            twitch_url=self.url,
            quality="720p60",  # quality of the stream could be ["160p", "360p", "480p", "720p", "720p60", "1080p", "1080p60"]
            blocking=True,
            rate=self.fps  # frame per rate (fps)
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
            
            #print('f')
            audio_segment = audio_grabber.grab()
            #print('ff')
            frames = []
            for i in range(self.batch_sec * self.fps):
                frame = image_grabber.grab()
                if self.output_shape is not None:
                    frame = cv2.resize(frame, dsize=[self.output_shape[1], self.output_shape[0]], interpolation=cv2.INTER_AREA)
                frames.append(frame)
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
        self.proc = mp.Process(target=self.proc, daemon=True)
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
    recoder = TwitchRecoder(target_url=TW_MARU)
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