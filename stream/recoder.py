from dataclasses import dataclass
from queue import Empty
from twitch_realtime_handler import (TwitchAudioGrabber,
                                   TwitchImageGrabber)
import cv2, time, os
import numpy as np
import multiprocessing as mp

SHARK = 'https://twitch.tv/tizmtizm'
MARU = 'https://www.twitch.tv/maoruya'

@dataclass
class RecoderEntry:
    index: int
    audio_segment: np.ndarray
    frames: np.ndarray
    fps: float

class TwitchRecoder:
    def __init__(self, target_url=MARU, batch_sec=1, fps=24):
        assert isinstance(batch_sec, int)
        self.url = target_url
        self.batch_sec = batch_sec
        self.fps = fps
        self.queue = mp.Queue(maxsize=100)
        self.cmd_queue = mp.Queue()
    
    def proc(self):
        # change to a stream that is actually online
        audio_grabber = TwitchAudioGrabber(
            twitch_url=self.url,
            blocking=True,  # wait until a segment is available
            segment_length=int(self.batch_sec),  # segment length in seconds
            rate=44000,  # sampling rate of the audio
            channels=2,  # number of channels
            dtype=np.int32  # quality of the audio could be [np.int16, np.int32, np.float32, np.float64]
        )

        image_grabber = TwitchImageGrabber(
            twitch_url=self.url,
            quality="1080p60",  # quality of the stream could be ["160p", "360p", "480p", "720p", "720p60", "1080p", "1080p60"]
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
            
            audio_segment = audio_grabber.grab()
            frames = []
            for i in range(self.batch_sec * self.fps):
                frame = image_grabber.grab()
                frames.append(frame)
            frames = np.stack(frames, axis=0)
            t_sum.append(time.time()-t)
            if len(t_sum) > 100:
                t_sum.pop(0)
            t_avg = sum(t_sum)/len(t_sum)
            print(f'TwitchRecoder: batch[{index}] captured took average {t_avg:.2f} sec. Audio[{audio_segment.shape}] Video[{frames.shape}]')
            t = time.time()
            self.queue.put(RecoderEntry(
                index=index,
                audio_segment=audio_segment, #(22000,2)
                frames=frames, #(24, 1080, 1920,3) -> (24, 2160, 3840, 3)
                fps=self.fps
            ))
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
    recoder = TwitchRecoder(target_url='https://www.twitch.tv/gosegugosegu')
    recoder.start()

    time.sleep(3)

    if not os.path.exists('./saves/frames/'): os.mkdir('./saves/frames/')
    j = 0
    for i in range(10):
        batch = recoder.queue.get(timeout=3) #type: RecoderEntry
        for k in range(batch.frames.shape[0]):
            cv2.imwrite(f"saves/frames/{j:04}.png", cv2.cvtColor(batch.frames[k], cv2.COLOR_RGB2BGR))
            j += 1
        print(f"{i} batch get. {batch.frames.shape}")
        
    recoder.stop()