import time, torch, cv2, gc, tqdm
from wsgiref.headers import tspecials
from matplotlib import pyplot as plt
import numpy as np
from upscale.model.fsrcnn.factory import build_model

torch.hub._validate_not_a_forked_repo=lambda a,b,c: True
torch.backends.cudnn.benchmark = True

from .upscaler_base import BaseUpscalerService, UpscalerQueueEntry

def log(*args, **kwargs):
    print(f"FsrcnnUpscalerService: {' '.join([str(a) for a in args])}", **kwargs)

class FsrcnnUpscalerService(BaseUpscalerService):
    def __init__(self, lr_level=3, device=0, on_queue=None):
        self.lr_shape = [
            (360, 640),
            (540, 960),
            (630, 1120),
            (720, 1280),
        ][lr_level]
        self.scale = 4
        self.hr_shape = tuple([i * self.scale for i in self.lr_shape])
        self.hr_shape = (1440, 2560)
        self.device = device
        self.on_queue = on_queue
        self.output_shape = None

        super().__init__()
    
    def proc_init(self):
        log('proc init')
        self.model = build_model(
            factor=self.scale, device=self.device
        ).eval()
        log('model loaded')
    
    def proc_cleanup(self):
        pass
    
    def upscale(self, frames: torch.Tensor):
        assert isinstance(frames, torch.Tensor)
        frames = frames.to(self.device, non_blocking=True)
        if frames.ndim == 4:
            assert frames.shape[-1] == 3
            N, H, W, C = frames.shape
            hrs = []
            for i in range(N):
                hr = self.upscale_single(frames[i])
                hrs.append(hr)
            hrs = torch.stack(hrs, dim=0).detach()
            return hrs
        else: 
            raise Exception(frames.shape)
    
    def upscale_single(self, img:torch.Tensor):
        with torch.no_grad():
            img = img.permute(2,0,1).unsqueeze(1)
            img = img / 255.0
            lr_curr = torch.nn.functional.interpolate(
                img, size=self.lr_shape, mode='area'
            )
            
            hr_curr = self.model(lr_curr)

            _hr_curr = torch.clamp(hr_curr, 0, 1)
            if self.output_shape is not None:
                _hr_curr = torch.nn.functional.interpolate(
                    _hr_curr, size=self.output_shape, mode='area'
                )
            return (_hr_curr * 255)[:,0,:,:].permute(1,2,0).to(torch.uint8)