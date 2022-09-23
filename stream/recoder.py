from dataclasses import dataclass
from twitchrealtimehandler import (TwitchAudioGrabber,
                                   TwitchImageGrabber)
import cv2, time
import numpy as np
import multiprocessing as mp

SHARK = 'https://twitch.tv/tizmtizm'
MARU = 'https://www.twitch.tv/maoruya'

@dataclass
class RecoderEntry:
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
        t_sum = 0
        t_count = 0
        while True:
            audio_segment = audio_grabber.grab()
            frames = []
            for i in range(self.batch_sec * self.fps):
                frame = image_grabber.grab()
                frames.append(frame)
            self.queue.put(RecoderEntry(
                audio_segment=audio_segment,
                frames=frames,
                fps=self.fps
            ))
            t_sum += time.time()-t
            t_count += 1
            print('queue took', t_sum/t_count, audio_segment.shape, frames.shape)
            t = time.time()
        
        image_grabber.terminate()
        audio_grabber.terminate()

    def start(self):
        self.proc = mp.Process(target=self.proc, daemon=True)
        self.proc.start()
    
    def join(self):
        self.proc.join()

if __name__ == '__main__':
    print('asdf')
    recoder = TwitchRecoder()
    recoder.start()
    recoder.join()