from dataclasses import dataclass
import queue, torch
import time
import threading
from twitchstream.outputvideo import TwitchBufferedOutputStream
from twitchstream.chat import TwitchChatStream
from upscale.base_service import BaseService
from env_var import *
import numpy as np

@dataclass
class TwitchStreamerEntry:
    frames: np.ndarray
    audio_segments: np.ndarray
    step: int

class TwitchStreamer(BaseService):
    def __init__(self, 
        resolution=(2160,3840), fps=24, 
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
            verbose=True
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
        self.frame_queue = queue.Queue(maxsize=int(self.fps*30))
        self.audio_queue = queue.Queue(maxsize=30)
        self.thread = threading.Thread(target=self.streaming_thread, daemon=True)
        self.thread.start()
    
    def streaming_thread(self):
        frequency = 100
        last_phase = 0
        last_frame_warn = 0

        while True:
            #manage stream
            videostream = self.videostream
            chatstream = self.chatstream
            if videostream is None:
                print("TwitchStreamer.thread: Thread exit")
                return

            #received = chatstream.twitch_receive_messages()
            received = False

            # process all the messages
            if received:
                for chat_message in received:
                    print(f"TwitchStramer.thread: Got a message '{chat_message['message']}' from {chat_message['username']}")

            if videostream.get_video_frame_buffer_state() < 30:
                try:
                    frame = self.frame_queue.get_nowait()
                    videostream.send_video_frame(frame)
                    print('frame sent')
                except queue.Empty:
                    if (time.time() - last_frame_warn) > 3:
                        print("TwitchStreamer: [W] frame needed, but not supplied")
                        last_frame_warn = time.time()
            elif videostream.get_audio_buffer_state() < 30:
                x = np.linspace(last_phase,
                                last_phase +
                                frequency*2*np.pi/videostream.fps,
                                int(44100 / videostream.fps) + 1)
                last_phase = x[-1]
                audio = np.sin(x[:-1])
                videostream.send_audio(audio, audio)
            else:
                time.sleep(.001)
    
    def proc_job_recieved(self, job:TwitchStreamerEntry):
        #push received job into queue
        if job.step > self.last_step:
            print('TwitchStreamer: [W] Job is queued with incorrect order.')
        for i in range(len(job.frames)):
            frame = job.frames[i]
            if isinstance(frame, torch.Tensor):
                frame = frame.numpy().astype(np.uint8)
            self.frame_queue.put(frame)
        self.audio_queue.put(job.audio_segments)
        self.last_step = job.step
        
        return job
    
    def proc_cleanup(self):
        self.chatstream = None
        self.videostream = None
        self.thread.join(timeout=15)

if __name__ == '__main__':
    streamer = TwitchStreamer()
    streamer.start()

    for i in range(1200):
        streamer.push_job(TwitchStreamerEntry(
            frames=(np.ones((*streamer.resolution, 3), dtype=np.float32) * (i%255)).astype(np.uint8),
            audio_segments=None,
            step=i,
        ), timeout=60)

    streamer.wait_for_job_clear()
    streamer.stop()