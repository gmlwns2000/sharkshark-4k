import json, os
os.environ['CUDA_MODULE_LOADING'] = 'lazy'
import math
import time, queue
import torch
import numpy as np
from upscale.upscaler_base import UpscalerQueueEntry
from upscale.egvsr_upscaler import EgvsrUpscalerService
from upscale.fsrcnn_upscaler import FsrcnnUpscalerService
from stream.recoder import TW_DALTA, TW_RUMYONG, TW_SHYLILY, TW_ZURURU, TwitchRecoder, TW_MARU, TW_PIANOCAT, TW_SHARK, RecoderEntry, TW_MAOU, TW_VIICHAN, TW_DANCINGSANA
from stream.streamer import TwitchStreamer, TwitchStreamerEntry

class TwitchUpscalerPostStreamer:
    def __init__(self, 
        #streaming settings
        url, fps=12, quality='720p60', frame_skips=True, output_file='rtmp://127.0.0.1/live',
        #device settings
        device=0, 
        #upscaler settings
        lr_level=3, hr_level=0, upscale_method='fsrcnn',
        #denoiser settings
        denoising=True, denoise_rate=1.0, 
        #sync
        audio_skip=0
    ) -> None:
        self.url = url
        self.fps = fps
        self.device = device
        self.small_batch_size = min(4, int(fps))

        self.recoder = TwitchRecoder(
            target_url=self.url, batch_sec=1, fps=self.fps, on_queue=self.recoder_on_queue,
            quality=quality, audio_skip=audio_skip,
        )
        #self.batch_size = self.fps
        # self.upscaler = EgvsrUpscalerService(
        #     device=self.device, lr_level=0, on_queue=self.upscaler_on_queue
        # )
        self.upscaler = FsrcnnUpscalerService(
            device=self.device, lr_level=lr_level, on_queue=self.upscaler_on_queue, 
            denoising=denoising, denoise_rate=denoise_rate, batch_size=self.small_batch_size
        )
        self.recoder.output_shape = self.upscaler.lr_shape
        self.upscaler.output_shape = [
            (1440, 2560),
            (1800, 3200),
            (2160, 3840),
        ][hr_level]

        self.streamer = TwitchStreamer(
            on_queue=self.streamer_on_queue, fps=self.fps, resolution=self.upscaler.output_shape, 
            output_file=output_file #'output.flv'
        )
        
        self.frame_step = 0
        self.last_reported = self.last_streamed = time.time()
        self.frame_skips = frame_skips
        
    def recoder_on_queue(self, entry:RecoderEntry):
        small_batch_size = self.small_batch_size
        for i in range(math.ceil(len(entry.frames)/small_batch_size)):
            try:
                entry.profiler.start('recoder.output.entry')
                frames = torch.tensor(entry.frames[i*small_batch_size:(i+1)*small_batch_size], dtype=torch.uint8, requires_grad=False)
                assert small_batch_size != 0
                assert (math.ceil(len(entry.frames)/small_batch_size)) != 0
                audio_segment = torch.tensor(entry.audio_segment[
                    i*(len(entry.audio_segment)//(math.ceil(len(entry.frames)/small_batch_size))):
                    (i+1)*(len(entry.audio_segment)//(math.ceil(len(entry.frames)/small_batch_size)))
                ], requires_grad=False)
                frames = frames.to(self.device)
                frames.share_memory_()
                audio_segment.share_memory_()
                entry.profiler.set('recoder.output.frames.shape', str(tuple(frames.shape)))
                new_entry = UpscalerQueueEntry(
                    frames=frames, 
                    audio_segment=audio_segment, 
                    step=self.frame_step,
                    profiler=entry.profiler
                )
                self.frame_step += 1
                entry.profiler.end('recoder.output.entry')
                if self.frame_skips:
                    self.upscaler.push_job_nowait(new_entry)
                else:
                    self.upscaler.push_job(new_entry)
            except queue.Full:
                print("TwitchUpscalerPostStreamer: recoder output skipped")

    def upscaler_on_queue(self, entry:UpscalerQueueEntry):
        # print(
        #     f'TwitchUpscalerPostStreamer: upscaled, '+\
        #     f'tensor{entry.frames[0].shape}[{len(entry.frames)}], {entry.step}, '+\
        #     f'elapsed: {entry.elapsed*1000:.1f}ms, onqueue'
        # )
        try:
            entry.profiler.start('upscaler.output.queue')
            frames = entry.frames.detach().clone()#.to('cpu')
            #frames = (frames*255).to(torch.uint8)
            audio_segments = entry.audio_segment.detach().clone()#.to('cpu')
            if not frames.is_shared():
                frames.share_memory_()
            if not audio_segments.is_shared():
                audio_segments.share_memory_()
            entry.profiler.set('upscaler.output.frames.shape', str(tuple(frames.shape)))
            new_entry = TwitchStreamerEntry(
                frames=frames,
                audio_segments=audio_segments,
                step=entry.step,
                profiler=entry.profiler
            )
            entry.profiler.end('upscaler.output.queue')
            if self.frame_skips:
                self.streamer.push_job_nowait(new_entry)
            else:
                self.streamer.push_job(new_entry)
        except queue.Full:
            print("TwitchUpscalerPostStreamer: upscaler output skipped")
    
    def streamer_on_queue(self, entry:TwitchStreamerEntry):
        print(f'TwitchUpscalerPostStreamer: streamed, idx: {entry.step}, took: {(time.time()-self.last_streamed)*1000:.1f}ms, '+\
            f'frames[{len(entry.frames)},{entry.frames[0].shape}], {entry.frames[0,0,0,0]}/{entry.audio_segments[0,0]:.4f}[{entry.audio_segments.shape}]')
        entry.profiler.set('upscaler.upscale.per_frame_ms', (entry.profiler.data['upscaler.upscale'] / len(entry.frames))*1000)
        if (time.time()-self.last_reported) > 3.0:
            entry.profiler.set('upscaler.inputq', self.upscaler.job_queue.qsize())
            entry.profiler.set('streamer.inputq', self.streamer.job_queue.qsize())
            print(json.dumps(entry.profiler.data, indent=2))
            self.last_reported = time.time()
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
        code = self.streamer.join()
        if code: raise Exception('streamer failed')
        self.upscaler.join()
        self.recoder.join()

if __name__ == '__main__':
    import streamlink.plugins.twitch as twstream
    #twstream.set_hls_proxy_method('none')
    # pipeline = TwitchUpscalerPostStreamer(
    #     url = 'https://www.twitch.tv/videos/1609788369', fps = 8, denoising=True, lr_level=3, quality='source', frame_skips=False
    # )

    pipeline = TwitchUpscalerPostStreamer(
        url = TW_VIICHAN, fps = 24, denoising=False, lr_level=3, quality='1080p60', frame_skips=True, denoise_rate=0.75, hr_level=0,
        output_file='rtmp://127.0.0.1:1935/live', audio_skip=0,
    )

    # pipeline = TwitchUpscalerPostStreamer(
    #     url = 'https://www.twitch.tv/videos/1610992145', fps = 24, denoising=True, lr_level=4, hr_level=1, denoise_rate=2.0,
    #     quality='1080p60', frame_skips=False, output_file='output.flv'
    # )
    
    pipeline.start()
    pipeline.join()