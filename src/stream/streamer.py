from dataclasses import dataclass
import queue, torch
import time
import threading
from ..twitchstream.outputvideo import TwitchBufferedOutputStream, TwitchOutputStream
from ..twitchstream.chat import TwitchChatStream
from ..upscale.base_service import BaseService
from env_var import *
import numpy as np
import torch.multiprocessing as mp
import cv2

from ..util.profiler import Profiler

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
        frames_to_send = []
        if isinstance(job.frames, torch.Tensor):
            assert job.frames.shape[-1] in [3,4]
            
            if job.frames.shape[1:] != (*self.resolution, 3):
                print('missmatch', job.frames.shape, self.resolution)
                if job.frames.shape[0] >= self.resolution[0]:
                    job.frames = torch.nn.functional.interpolate(job.frames.permute(0, 3, 1, 2), self.resolution, mode='area').permute(0, 2, 3, 1)
                else:
                    job.frames = torch.nn.functional.interpolate(job.frames.permute(0, 3, 1, 2), self.resolution, mode='bicubic').permute(0, 2, 3, 1)
            
            if job.frames.device != 'cpu':
                if job.frames.dtype != torch.uint8:
                    job.frames = job.frames.to(torch.uint8, non_blocking=True)
                job.frames = job.frames.to('cpu', non_blocking=True)
                job.frames = job.frames.numpy()
                if(job.frames.dtype != np.uint8):
                    job.frames = job.frames.astype(np.uint8)
        
        for i in range(len(job.frames)):
            frame = job.frames[i]
            # if isinstance(frame, torch.Tensor):
            #     # print(f"TwitchStreamer.proc: {frame.shape} {frame.device}")
            #     frame = frame.numpy()
            #     if frame.dtype != np.uint8:
            #         frame = frame.astype(np.uint8)
            
            if frame.shape != (*self.resolution, 3):
                raise Exception('size mismatch', frame.shape, self.resolution)
            
            frames_to_send.append(frame)
        
        job.profiler.end('streamer.frames.queue')
        
        job.profiler.start('streamer.audio.queue')
        audio_seg = job.audio_segments
        if isinstance(audio_seg, torch.Tensor):
            audio_seg = audio_seg.numpy()
        batch_size = len(frames_to_send)
        audio_segs_to_send = []
        for i in range(batch_size):
            seg = audio_seg[i*(audio_seg.shape[0]//batch_size):(i+1)*(audio_seg.shape[0]//batch_size)]
            audio_segs_to_send.append(seg)
            #videostream.send_audio(seg[:,0], seg[:,1])
        job.profiler.end('streamer.audio.queue')

        job.profiler.start('streamer.send.queue')
        for i in range(len(frames_to_send)):
            frame = frames_to_send[i] #type: np.ndarray
            job.profiler.start('streamer.send.queue.txt')
            if isinstance(frame, np.ndarray):
                if not frame.data.c_contiguous:
                    frame = np.ascontiguousarray(frame)
                frame = cv2.putText(frame, 
                    f"[SHKSHK-AinL] Processed: {self.frame_count} frames {job.step * len(frames_to_send) - self.frame_count + i} "+\
                    f"skipped ({(job.step * len(frames_to_send) - self.frame_count + i)/(self.frame_count+1e-8)*100:.1f}%)", 
                    (10, 32), cv2.FONT_HERSHEY_PLAIN, 1.0, (0,255,0), 2
                )
            job.profiler.end('streamer.send.queue.txt')
            job.profiler.start('streamer.send.queue.video')
            self.frame_count += 1
            videostream.send_video_frame(frame)
            job.profiler.end('streamer.send.queue.video')
            
            seg = audio_segs_to_send[i]
            job.profiler.start('streamer.send.queue.audio')
            videostream.send_audio(seg[:,0], seg[:,1])
            job.profiler.end('streamer.send.queue.audio')
        job.profiler.end('streamer.send.queue')

        self.last_step = job.step
        
        return job
    
    def proc_cleanup(self):
        self.chatstream = None
        self.videostream = None

if __name__ == '__main__':
    def on_queue(entry:TwitchStreamerEntry):
        print('Streamer: entry processed', entry.step)
    streamer = TwitchStreamer(on_queue=on_queue, fps=24, output_file='rtmp://127.0.0.1/live')
    streamer.start()

    from .recoder import TwitchRecoder, RecoderEntry, TW_MARU, TW_PIANOCAT
    def on_queue_recoder(entry:RecoderEntry):
        streamer.push_job(TwitchStreamerEntry(
            frames=entry.frames,
            audio_segments=entry.audio_segment,
            step=entry.index,
            profiler=entry.profiler
        ))
    
    recoder = TwitchRecoder(target_url='https://www.twitch.tv/cotton__123', fps=24, quality='1080p60', on_queue=on_queue_recoder)
    recoder.start()

    recoder.join()
    streamer.join()