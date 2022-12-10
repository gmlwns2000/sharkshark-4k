import time, torch, cv2, gc, tqdm, math
from wsgiref.headers import tspecials
from matplotlib import pyplot as plt
import numpy as np
from upscale.model.fsrcnn.factory import build_model as build_model_fsrcnn
from upscale.model.realesrgan.factory import build_model as build_model_esrgan
from upscale.model.bsvd.factory import build_model as build_denoise_model
from util.profiler import Profiler

torch.hub._validate_not_a_forked_repo=lambda a,b,c: True
torch.backends.cudnn.benchmark = True

from .upscaler_base import BaseUpscalerService, UpscalerQueueEntry

def log(*args, **kwargs):
    print(f"FsrcnnUpscalerService: {' '.join([str(a) for a in args])}", **kwargs)

def blur_ker(channels=1, kernel_size = 3, sigma = 0.5):
    # Set these to whatever you want for your gaussian filter

    # Create a x, y coordinate grid of shape (kernel_size, kernel_size, 2)
    x_cord = torch.arange(kernel_size)
    x_grid = x_cord.repeat(kernel_size).view(kernel_size, kernel_size)
    y_grid = x_grid.t()
    xy_grid = torch.stack([x_grid, y_grid], dim=-1)

    mean = (kernel_size - 1)/2.
    variance = sigma**2.

    # Calculate the 2-dimensional gaussian kernel which is
    # the product of two gaussian distributions for two different
    # variables (in this case called x and y)
    gaussian_kernel = (1./(2.*math.pi*variance)) *\
                    torch.exp(
                        -torch.sum((xy_grid - mean)**2., dim=-1) /\
                        (2*variance)
                    )
    # Make sure sum of values in gaussian kernel equals 1.
    gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)

    # Reshape to 2d depthwise convolutional weight
    gaussian_kernel = gaussian_kernel.view(1, 1, kernel_size, kernel_size)
    gaussian_kernel = gaussian_kernel.repeat(channels, 1, 1, 1)

    gaussian_filter = torch.nn.Conv2d(in_channels=channels, out_channels=channels,
                                kernel_size=kernel_size, groups=channels, bias=False, padding=kernel_size//2, padding_mode='reflect')

    gaussian_filter.weight.data = gaussian_kernel
    gaussian_filter.weight.requires_grad = False
    return gaussian_filter

def sharpen_ker(channels=1, strength=1.0):
    kernel_size = 3

    assert channels == 1
    gaussian_kernel_sharp = torch.tensor([
        [-1,-1,-1],
        [-1, 9,-1],
        [-1,-1,-1],
    ])

    gaussian_kernel_id = torch.tensor([
        [0,0,0],
        [0,1,0],
        [0,0,0],
    ])

    gaussian_kernel = gaussian_kernel_sharp * strength + (1-strength) * gaussian_kernel_id

    gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)
    gaussian_kernel = gaussian_kernel.view(1, 1, kernel_size, kernel_size)
    gaussian_kernel = gaussian_kernel.repeat(channels, 1, 1, 1)

    gaussian_filter = torch.nn.Conv2d(
        in_channels=channels, out_channels=channels,
        kernel_size=kernel_size, groups=channels, bias=False, 
        padding=kernel_size//2, padding_mode='reflect'
    )

    gaussian_filter.weight.data = gaussian_kernel
    gaussian_filter.weight.requires_grad = False
    return gaussian_filter

class FsrcnnUpscalerService(BaseUpscalerService):
    profiler: Profiler

    def __init__(self, lr_level=3, device=0, on_queue=None, denoising=True, denoise_rate=1.0, upscaler_model='realesrgan', batch_size=1):
        self.lr_shape = [
            (360, 640),
            (540, 960),
            (630, 1120),
            (720, 1280),
            (900, 1600),
            (1080, 1920),
        ][lr_level]
        self.scale = 4
        self.denoise_rate = denoise_rate
        self.hr_shape = tuple([i * self.scale for i in self.lr_shape])
        self.hr_shape = (1440, 2560)
        self.device = device
        self.on_queue = on_queue
        self.output_shape = None
        self.upscaler_model = upscaler_model
        self.single_mode = upscaler_model != 'realesrgan'

        self.denoising = denoising
        self.batch_size = batch_size

        super().__init__()
    
    def proc_init(self):
        log('proc init')
        self.lr_prev = None
        if self.upscaler_model == 'fsrcnn':
            self.model = build_model_fsrcnn(
                factor=self.scale, device=self.device, input_shape=self.lr_shape
            ).eval()
        elif self.upscaler_model == 'realesrgan':
            self.model = build_model_esrgan(
                factor=self.scale, device=self.device, input_shape=self.lr_shape, 
                batch_size=self.batch_size, denoise_rate=self.denoise_rate
            ).eval()
        else: raise Exception(self.upscaler_model)
        if self.denoising:
            self.denoise_model = build_denoise_model(
                device=self.device, input_shape=self.lr_shape
            )
            self.denoise_blur = blur_ker().half().to(self.device)
            self.denoise_sharpen = sharpen_ker(strength=0.00002).half().to(self.device)
            self.denoise_sharpen_hr = sharpen_ker(strength=0.00007).half().to(self.device)
        log('model loaded')
    
    def proc_cleanup(self):
        pass
    
    def upscale(self, frames: torch.Tensor):
        assert isinstance(frames, torch.Tensor)
        if frames.device != self.device:
            frames = frames.to(self.device, non_blocking=True)
        if frames.ndim == 4:
            assert frames.shape[-1] == 3
            N, H, W, C = frames.shape
            if self.single_mode:
                hrs = []
                for i in range(N):
                    hr = self.upscale_single(frames[i])
                    hrs.append(hr)
                hrs = torch.stack(hrs, dim=0).detach()
            else:
                hrs = self.upscale_multi(frames)
            return hrs
        else: 
            raise Exception(frames.shape)
    
    def upscale_multi(self, img:torch.Tensor):
        with torch.no_grad(), torch.cuda.amp.autocast():
            img = img.permute(0,3,1,2)
            img = img / 255.0
            lr_curr_before = lr_curr = img
            if img.shape[-1] > self.lr_shape[-1] or img.shape[-2] > self.lr_shape[-2]:
                lr_curr_before = lr_curr = torch.nn.functional.interpolate(
                    img, size=self.lr_shape, mode='area'
                )
        
        with torch.no_grad(), torch.cuda.amp.autocast():
            self.profiler.start('fsrcnn.model')
            if self.upscaler_model == 'realesrgan':
                hr_curr = self.model(lr_curr)
                # hr_curr = lr_curr
            else:
                raise Exception()
            
            self.profiler.end('fsrcnn.model')
        
        with torch.no_grad(), torch.cuda.amp.autocast():
            N, C, H, W = hr_curr.shape
            hr_curr = hr_curr.view(N, C, H, W)
            hr_curr_mean = hr_curr.view(N, C, H*W).mean(dim=-1).view(N, C, 1, 1)
            hr_curr_std = hr_curr.view(N, C, H*W).std(dim=-1).view(N, C, 1, 1)
            N, C, LH, LW = lr_curr_before.shape
            img_mean = lr_curr_before.view(N, C, LH*LW).mean(dim=-1).view(N, C, 1, 1)
            img_std = lr_curr_before.view(N, C, LH*LW).std(dim=-1).view(N, C, 1, 1)
            hr_curr = (hr_curr - hr_curr_mean) / (hr_curr_std + 1e-8)
            hr_curr = hr_curr * img_std + img_mean
            hr_curr = hr_curr.view(N, C, H, W)
            
            _hr_curr = torch.clamp(hr_curr, 0, 1)
            if self.output_shape is not None:
                if self.output_shape[0] >= _hr_curr.shape[0]:
                    _hr_curr = torch.nn.functional.interpolate(
                        _hr_curr, size=self.output_shape, mode='bicubic'
                    )
                else:
                    _hr_curr = torch.nn.functional.interpolate(
                        _hr_curr, size=self.output_shape, mode='area'
                    )
            _hr_curr = torch.clamp(_hr_curr, 0, 1)
            return (_hr_curr * 255).permute(0,2,3,1).to(torch.uint8, non_blocking=True)
    
    def upscale_single(self, img:torch.Tensor):
        with torch.cuda.amp.autocast():
            img = img.permute(2,0,1).unsqueeze(0)
            img = img / 255.0
            lr_curr_before = lr_curr = torch.nn.functional.interpolate(
                img, size=self.lr_shape, mode='area'
            ).squeeze(0)

        diff_map = None

        if self.denoising:
            __lr_curr = lr_curr
            _lr_curr = lr_curr.unsqueeze(0).unsqueeze(1)
            N, F, C, H, W = _lr_curr.shape
            lr_curr = torch.empty((N, 2, 4, H, W), dtype=_lr_curr.dtype, device=_lr_curr.device)
            if self.lr_prev is not None:
                with torch.no_grad(), torch.cuda.amp.autocast():
                    diff_map = torch.mean(torch.abs(__lr_curr - self.lr_prev), dim=0)
                    diff_map = self.denoise_blur(diff_map.unsqueeze(0).unsqueeze(0)).squeeze(0).squeeze(0) * 10
                    diff_map = torch.clamp(diff_map, 0.00, 0.1) * 0.35 * self.denoise_rate
                    #diff_map.fill_(0.1)
                # lr_curr[:,:,:3,:,:] = _lr_curr
                # lr_curr[0,0,0,:,:] = diff_map
                # lr_curr[0,0,1,:,:] = diff_map
                # lr_curr[0,0,2,:,:] = diff_map
                # lr_curr[0,0,3,:,:] = 0.0
                
                diff_map.fill_(0.1 * self.denoise_rate)
                
                lr_curr[0,0,3,:,:] = self.lr_prev_diff_map
                lr_curr[0,0,:3,:,:] = self.lr_prev
                lr_curr[0,1,3,:,:] = diff_map
                lr_curr[0,1,:3,:,:] = _lr_curr
            else:
                lr_curr.fill_(0.05)
                lr_curr[0,1,:3,:,:] = _lr_curr
                diff_map = lr_curr[0,1,3,:,:]
            with torch.no_grad(), torch.cuda.amp.autocast():
                denoise_opacity = 0.8

                self.profiler.start('fsrcnn.denoise')
                before_denoise = lr_curr
                lr_curr = self.denoise_model(lr_curr[:,1:2,:,:,:])[:,-1,:,:,:].squeeze(0).squeeze(0)
                C, H, W = lr_curr.shape
                lr_curr = torch.clamp(self.denoise_sharpen(lr_curr.view(C, 1, H, W)).view(C, H, W), 0, 1)

                lr_curr = lr_curr * denoise_opacity + (1-denoise_opacity) * __lr_curr
                self.profiler.end('fsrcnn.denoise')
            self.lr_prev = lr_curr.clone()
            self.lr_prev_diff_map = diff_map.clone()
        
        # if diff_map is not None:
        #     lr_curr[:,:,:] *= diff_map.unsqueeze(0) * 10
        lr_curr = lr_curr.unsqueeze(1)
        
        with torch.no_grad(), torch.cuda.amp.autocast():
            self.profiler.start('fsrcnn.model')
            if self.upscaler_model == 'realesrgan':
                lr_curr = lr_curr.permute(1,0,2,3)
                hr_curr = self.model(lr_curr.float())
                hr_curr = hr_curr.permute(1,0,2,3)
            else:
                hr_curr = self.model(lr_curr.float())
            if self.denoising:
                hr_curr = torch.clamp(self.denoise_sharpen_hr(hr_curr), 0, 1)
            self.profiler.end('fsrcnn.model')

        with torch.no_grad(), torch.cuda.amp.autocast():
            #color distribution match
            C, F, H, W = hr_curr.shape
            hr_curr = hr_curr.view(C, H, W)
            hr_curr_mean = hr_curr.view(C, H*W).mean(dim=-1).view(C, 1, 1)
            hr_curr_std = hr_curr.view(C, H*W).std(dim=-1).view(C, 1, 1)
            C, LH, LW = lr_curr_before.shape
            img_mean = lr_curr_before.view(C, LH*LW).mean(dim=-1).view(C, 1, 1)
            img_std = lr_curr_before.view(C, LH*LW).std(dim=-1).view(C, 1, 1)
            hr_curr = (hr_curr - hr_curr_mean) / (hr_curr_std + 1e-8)
            hr_curr = hr_curr * img_std + img_mean
            hr_curr = hr_curr.view(C, F, H, W)

            _hr_curr = torch.clamp(hr_curr, 0, 1)
            if self.output_shape is not None:
                if self.output_shape[0] >= _hr_curr.shape[0]:
                    _hr_curr = torch.nn.functional.interpolate(
                        _hr_curr, size=self.output_shape, mode='bicubic'
                    )
                else:
                    _hr_curr = torch.nn.functional.interpolate(
                        _hr_curr, size=self.output_shape, mode='area'
                    )
            _hr_curr = torch.clamp(_hr_curr, 0, 1)
            return (_hr_curr * 255)[:,0,:,:].permute(1,2,0).to(torch.uint8, non_blocking=True)