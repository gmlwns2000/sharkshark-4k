from dataclasses import dataclass
import queue, torch
import time
import threading
from twitchstream.outputvideo import TwitchBufferedOutputStream
from twitchstream.chat import TwitchChatStream
from upscale.base_service import BaseService
from env_var import *
import numpy as np
import torch.multiprocessing as mp
import cv2

from util.profiler import Profiler

@dataclass
class TwitchStreamerEntry:
    frames: np.ndarray
    audio_segments: np.ndarray
    step: int
    profiler: Profiler

class TwitchStreamer(BaseService):
    def __init__(self, 
        resolution=(1080,1920), fps=24, 
        streamkey=TWITCH_STREAMKEY, oauth=TWITCH_OAUTH, username=TWITCH_USERNAME,
        on_queue=None, output_file = None
    ) -> None:
        self.streamkey = streamkey
        self.username = username
        self.resolution = resolution
        self.fps = fps
        self.oauth = oauth
        self.last_step = -1
        self.on_queue = on_queue
        self.output_file = output_file
        super().__init__()

    def proc_pre_main(self):
        with TwitchBufferedOutputStream(
            twitch_stream_key=self.streamkey,
            width=self.resolution[1],
            height=self.resolution[0],
            fps=self.fps,
            enable_audio=True,
            verbose=True,
            output_file=self.output_file
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
        self.frame_count = 0
    
    def proc_job_recieved(self, job:TwitchStreamerEntry):
        job.profiler.end('upscaler.output')

        #manage stream
        videostream = self.videostream
        chatstream = self.chatstream
        if videostream is None:
            print("TwitchStreamer.thread: Thread exit")
            return job
        
        #push received job into queue
        if job.step < self.last_step:
            print('TwitchStreamer: [W] Job is queued with incorrect order.')
        
        job.profiler.start('streamer.frames.queue')
        for i in range(len(job.frames)):
            frame = job.frames[i]
            if isinstance(frame, torch.Tensor):
                #print(f"TwitchStreamer.proc: {frame.shape} {frame.device}")
                if frame.shape != (*self.resolution, 3):
                    print('missmatch', frame.shape, self.resolution)
                    if frame.shape[0] >= self.resolution[0]:
                        frame = torch.nn.functional.interpolate(frame, self.resolution, mode='area')
                    else:
                        frame = torch.nn.functional.interpolate(frame, self.resolution, mode='bicubic')
                if frame.device != 'cpu':
                    frame = frame.cpu()
                frame = frame.numpy().astype(np.uint8)
            if frame.shape != (*self.resolution, 3):
                frame = cv2.resize(frame, dsize=(self.resolution[1], self.resolution[0]), interpolation=cv2.INTER_AREA)
                print('err')
            frame = cv2.putText(frame, f"{self.frame_count}", (10, 40), cv2.FONT_HERSHEY_PLAIN, 2.0, (0,255,0), 2)
            self.frame_count += 1
            frame = frame.astype(np.float32) / 255.0
            #print('frame stat', np.min(frame), np.max(frame), frame.dtype, frame.shape)
            videostream.send_video_frame(frame)
        job.profiler.end('streamer.frames.queue')
        
        job.profiler.start('streamer.audio.queue')
        audio_seg = job.audio_segments
        if isinstance(audio_seg, torch.Tensor):
            audio_seg = audio_seg.numpy()
        batch_size = len(job.frames)
        for i in range(batch_size):
            seg = audio_seg[i*(audio_seg.shape[0]//batch_size):(i+1)*(audio_seg.shape[0]//batch_size)]
            videostream.send_audio(seg[:,0], seg[:,1])
        job.profiler.end('streamer.audio.queue')

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