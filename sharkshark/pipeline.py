import json
import time, queue
import torch
import numpy as np
from upscale.egvsr_upscaler import EgvsrUpscalerService, UpscalerQueueEntry
from stream.recoder import TW_RUMYONG, TwitchRecoder, TW_MARU, TW_PIANOCAT, TW_SHARK, RecoderEntry
from stream.streamer import TwitchStreamer, TwitchStreamerEntry

class TwitchUpscalerPostStreamer:
    def __init__(self, url, device=0, fps=12) -> None:
        self.url = url
        self.fps = fps
        self.device = device

        self.recoder = TwitchRecoder(
            target_url=self.url, batch_sec=1, fps=self.fps, on_queue=self.recoder_on_queue
        )
        self.batch_size = self.fps
        self.upscaler = EgvsrUpscalerService(
            device=self.device, lr_level=0, on_queue=self.upscaler_on_queue
        )
        self.streamer = TwitchStreamer(
            on_queue=self.streamer_on_queue, fps=self.fps, resolution=(720,1280)
        )
        self.frame_step = 0
        self.last_streamed = time.time()
    
    def recoder_on_queue(self, entry:RecoderEntry):
        try:
            entry.profiler.start('recoder.output.entry')
            new_entry = UpscalerQueueEntry(
                frames=torch.tensor(entry.frames, dtype=torch.float32, requires_grad=False), 
                audio_segment=torch.tensor(entry.audio_segment, requires_grad=False), 
                step=self.frame_step,
                profiler=entry.profiler
            )
            entry.profiler.end('recoder.output.entry')
            self.upscaler.push_job_nowait(new_entry)
        except queue.Full:
            print("TwitchUpscalerPostStreamer: recoder output skipped")
        self.frame_step += 1

    def upscaler_on_queue(self, entry:UpscalerQueueEntry):
        print(
            f'TwitchUpscalerPostStreamer: upscaled, '+\
            f'tensor{entry.frames[0].shape}[{len(entry.frames)}], {entry.step}, '+\
            f'elapsed: {entry.elapsed*1000:.1f}ms, onqueue'
        )
        try:
            entry.profiler.start('upscaler.output.queue')
            new_entry = TwitchStreamerEntry(
                frames=entry.frames.clone(),
                audio_segments=entry.audio_segment.clone(),
                step=entry.step,
                profiler=entry.profiler
            )
            entry.profiler.end('upscaler.output.queue')
            self.streamer.push_job_nowait(new_entry)
        except queue.Full:
            print("TwitchUpscalerPostStreamer: upscaler output skipped")
    
    def streamer_on_queue(self, entry:TwitchStreamerEntry):
        print(f'TwitchUpscalerPostStreamer: streamed, idx: {entry.step}, took: {(time.time()-self.last_streamed)*1000:.1f}ms, frames[{len(entry.frames)},{entry.frames[0].shape}]')
        print(json.dumps(entry.profiler.data, indent=2))
        self.last_streamed = time.time()
    
    def start(self):
        self.streamer.start()
        self.upscaler.start()
        self.recoder.start()
    
    def stop(self):
        self.recoder.stop()
        self.upscaler.stop()
        self.streamer.stop()

    def join(self):
        self.streamer.join()
        self.upscaler.join()
        self.recoder.join()

if __name__ == '__main__':
    pipeline = TwitchUpscalerPostStreamer(
        url = TW_RUMYONG, fps = 24
    )
    pipeline.start()
    pipeline.join()