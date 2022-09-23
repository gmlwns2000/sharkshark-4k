from twitchrealtimehandler import (TwitchAudioGrabber,
                                   TwitchImageGrabber)
import cv2
import numpy as np

SHARK = 'https://twitch.tv/tizmtizm'
MARU = 'https://www.twitch.tv/maoruya'

class TwitchRecoder:
    def __init__(self, target_url=MARU):
        self.url = target_url
    
    def start(self):
        # change to a stream that is actually online
        audio_grabber = TwitchAudioGrabber(
            twitch_url=self.url,
            blocking=True,  # wait until a segment is available
            segment_length=2,  # segment length in seconds
            rate=16000,  # sampling rate of the audio
            channels=2,  # number of channels
            dtype=np.int16  # quality of the audio could be [np.int16, np.int32, np.float32, np.float64]
            )

        audio_segment = audio_grabber.grab()
        audio_grabber.terminate()  # stop the transcoding
        print(audio_segment)

        image_grabber = TwitchImageGrabber(
            twitch_url=self.url,
            quality="1080p60",  # quality of the stream could be ["160p", "360p", "480p", "720p", "720p60", "1080p", "1080p60"]
            blocking=True,
            rate=10  # frame per rate (fps)
        )
        
        for i in range()
        frame = image_grabber.grab()
        print(frame)
        image_grabber.terminate()  # stop the transcoding

if __name__ == '__main__':
    recoder = TwitchRecoder()
    recoder.start()