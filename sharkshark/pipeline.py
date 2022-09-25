import torch
from upscale.egvsr_upscaler import EgvsrUpscalerService, UpscalerQueueEntry
from stream.recoder import TwitchRecoder, TW_MARU, TW_PIANOCAT, TW_SHARK, RecoderEntry

class TwitchUpscalerPostStreamer:
    def __init__(self, url, device=0, fps=24) -> None:
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
        self.frame_step = 0
    
    def recoder_on_queue(self, entry:RecoderEntry):
        self.upscaler.push_job(UpscalerQueueEntry(
            frames=torch.tensor(entry.frames, dtype=torch.float32), audio_segment=entry.audio_segment, step=self.frame_step
        ))
        self.frame_step += 1

    def upscaler_on_queue(self, entry:UpscalerQueueEntry):
        print(f'TwitchUpscalerPostStreamer: upscaled, tensor{entry.frames[0].shape}[{len(entry.frames)}], {entry.step}')
    
    def start(self):
        self.upscaler.start()
        self.recoder.start()
    
    def stop(self):
        self.recoder.stop()
        self.upscaler.stop()

    def join(self):
        self.upscaler.join()
        self.recoder.join()

if __name__ == '__main__':
    pipeline = TwitchUpscalerPostStreamer(
        url = TW_PIANOCAT, fps = 24
    )
    pipeline.start()
    pipeline.join()