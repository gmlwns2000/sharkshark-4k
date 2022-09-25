from dataclasses import dataclass
import queue, torch
import time
import threading
from twitchstream.outputvideo import TwitchBufferedOutputStream
from twitchstream.chat import TwitchChatStream
from upscale.base_service import BaseService
from env_var import *
import numpy as np
import multiprocessing as mp
import cv2

@dataclass
class TwitchStreamerEntry:
    frames: np.ndarray
    audio_segments: np.ndarray
    step: int

class TwitchStreamer(BaseService):
    def __init__(self, 
        resolution=(1080,1920), fps=24, 
        streamkey=TWITCH_STREAMKEY, oauth=TWITCH_OAUTH, username=TWITCH_USERNAME,
        on_queue=None
    ) -> None:
        self.streamkey = streamkey
        self.username = username
        self.resolution = resolution
        self.fps = fps
        self.oauth = oauth
        self.last_step = -1
        self.on_queue = on_queue
        super().__init__()

    def proc_pre_main(self):
        with TwitchBufferedOutputStream(
            twitch_stream_key=self.streamkey,
            width=self.resolution[1],
            height=self.resolution[0],
            fps=self.fps,
            enable_audio=True,
            verbose=False
        ) as videostream:
        # TwitchChatStream(
        #     username=self.username.lower(),  # Must provide a lowercase username.
        #     oauth=self.oauth,
        #     verbose=False
        # ) as chatstream:
            self.videostream = videostream
            self.chatstream = None#chatstream

            #chatstream.send_chat_message("Startup")

            self.proc_main()

    def proc_init(self):
        self.frequency = 100
        self.last_phase = 0
        self.last_frame_warn = 0
    
    def proc_job_recieved(self, job:TwitchStreamerEntry):
        #manage stream
        videostream = self.videostream
        chatstream = self.chatstream
        if videostream is None:
            print("TwitchStreamer.thread: Thread exit")
            return job
        
        #push received job into queue
        if job.step < self.last_step:
            print('TwitchStreamer: [W] Job is queued with incorrect order.')
        
        for i in range(len(job.frames)):
            frame = job.frames[i]
            if isinstance(frame, torch.Tensor):
                frame = frame.numpy().astype(np.uint8)
            if frame.shape != (*self.resolution, 3):
                frame = cv2.resize(frame, dsize=(self.resolution[1], self.resolution[0]), interpolation=cv2.INTER_AREA)
            frame = frame.astype(np.float32) / 255.0
            #print('frame stat', np.min(frame), np.max(frame), frame.dtype, frame.shape)
            videostream.send_video_frame(frame)
        
        videostream.send_audio(job.audio_segments[:,0], job.audio_segments[:,1])

        self.last_step = job.step
        
        return job
    
    def proc_cleanup(self):
        self.chatstream = None
        self.videostream = None

if __name__ == '__main__':
    def on_queue(entry:TwitchStreamerEntry):
        print('Streamer: entry processed', entry.step)
    streamer = TwitchStreamer(on_queue=on_queue, fps=24)
    streamer.start()

    from .recoder import TwitchRecoder, RecoderEntry, TW_MARU, TW_PIANOCAT
    def on_queue_recoder(entry:RecoderEntry):
        streamer.push_job(TwitchStreamerEntry(
            frames=entry.frames,
            audio_segments=entry.audio_segment,
            step=entry.index
        ))
    
    recoder = TwitchRecoder(target_url=TW_MARU, fps=24, on_queue=on_queue_recoder)
    recoder.start()

    recoder.join()
    streamer.join()