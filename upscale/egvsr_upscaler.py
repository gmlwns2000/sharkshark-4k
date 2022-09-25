import time, torch, cv2, gc, tqdm
from wsgiref.headers import tspecials
from matplotlib import pyplot as plt
import numpy as np
from upscale.model.egvsr.egvsr import FRNet

torch.hub._validate_not_a_forked_repo=lambda a,b,c: True
torch.backends.cudnn.benchmark = True

from .upscaler_base import BaseUpscalerService, UpscalerQueueEntry

def build_egvsr_model(
    img, device=0, lr_shape=(630, 1120), bsize=2, 
    jit_mode='tensorrt', do_benchmark=True
):
    with torch.no_grad():
        # device = 0
        # do_benchmark = True
        # lr_shape = (540, 960)
        # lr_shape = (630, 1120)
        # lr_shape = (720, 1280)
        # bsize = 2
        # jit_mode = 'tensorrt'

        state = torch.load('./saves/models/EGVSR_iter420000.pth', map_location='cpu')
        model = FRNet(in_nc=3, out_nc=3, nf=64, nb=10, degradation='BD', scale=4)
        model.load_state_dict(state)
        del state
        model = model.to(device).eval()

        batch = torch.tensor(img) * (1/255.0)
        batch = batch.permute(2,0,1).unsqueeze(0)
        #4096x2160
        batch_lr = torch.nn.functional.interpolate(batch, size=lr_shape, mode='area')\
            .to(device)
        batch_hr = torch.nn.functional.interpolate(batch, size=tuple([i*4 for i in lr_shape]), mode='bicubic')\
            .to(device)
        lr_curr = batch_lr
        lr_prev = batch_lr
        hr_prev = batch_hr

        #jit
        amp_enabled = True
        half_convert = False
        skip_repeat = False
        if jit_mode == 'torch':
            model_ft = model
            traced_model = torch.jit.trace(model_ft, (lr_curr, lr_prev, hr_prev))
            model = traced_model
            if do_benchmark:
                with torch.no_grad():
                    model(lr_curr, lr_prev, hr_prev) #run jit
        elif jit_mode == 'deepspeed':
            import deepspeed

            # Initialize the DeepSpeed-Inference engine
            ds_engine = deepspeed.init_inference(
                model,
                mp_size=1,
                dtype=torch.half,
                checkpoint=None,
                replace_method='auto',
                replace_with_kernel_inject=True
            )
            model = ds_engine.module
            amp_enabled = False
            if do_benchmark:
                with torch.no_grad():
                    model(lr_curr.half(), lr_prev.half(), hr_prev.half()) #run jit
            half_convert = True
        elif jit_mode == 'tensorrt':
            import torch_tensorrt, os
            bsize = 1
            version = '0'

            lr_curr = batch_lr.repeat(bsize, 1, 1, 1)
            lr_prev = batch_lr.repeat(bsize, 1, 1, 1)
            hr_prev = batch_hr.repeat(bsize, 1, 1, 1)
            N, _, H, W = lr_curr.shape

            ts_path = f"./saves/models/egvsr_{version}_{N}x3x{W}x{H}.pts"

            if os.path.exists(ts_path):
                model = torch.jit.load(ts_path)
            else:
                print('EgvsrUpscaler.build_egvsr_model: Compiling...')
                trt_model = torch_tensorrt.compile(model, 
                    inputs= [
                        torch_tensorrt.Input(lr_curr.shape), 
                        torch_tensorrt.Input(lr_prev.shape),
                        torch_tensorrt.Input(hr_prev.shape),
                    ],
                    enabled_precisions= { torch_tensorrt.dtype.half } # Run with FP16
                )
                model = trt_model
                torch.jit.save(model, ts_path)

            skip_repeat = True
            amp_enabled = False
            half_convert = False
            
            if half_convert:
                lr_curr = lr_curr.half()
                lr_prev = lr_prev.half()
                hr_prev = hr_prev.half()
            if do_benchmark:
                with torch.no_grad():
                    model(lr_curr, lr_prev, hr_prev) #run jit

        if do_benchmark:
            if not skip_repeat:
                lr_curr = batch_lr.repeat(bsize, 1, 1, 1)
                lr_prev = batch_lr.repeat(bsize, 1, 1, 1)
                hr_prev = batch_hr.repeat(bsize, 1, 1, 1)
            if half_convert:
                lr_curr = lr_curr.half()
                lr_prev = lr_prev.half()
                hr_prev = hr_prev.half()
            
            count = 100
            print('warmup')
            for i in range(10):
                with torch.no_grad(), torch.cuda.amp.autocast(enabled=amp_enabled):
                    output = model(lr_curr, lr_prev, hr_prev)
            del output
            torch.cuda.synchronize()
            # gc.collect()
            # torch.cuda.empty_cache()
            torch.cuda.synchronize()
            t = time.time()
            for i in tqdm.tqdm(range(count)):
                with torch.no_grad(), torch.cuda.amp.autocast(enabled=amp_enabled):
                    output = model(lr_curr, lr_prev, hr_prev)
                    torch.cuda.synchronize()
            t = time.time() - t
            print(f'took {t / (count*bsize) * 1000} ms per img, {torch.cuda.memory_allocated() / (1024**2)}MB used')

            uimg = torch.clamp(output[0].squeeze(0).permute(1,2,0)*255, 0, 255)
            uimg = uimg.detach().cpu().numpy().astype(np.uint8)
            plt.imshow(uimg)
            cv2.imwrite(f'./samples/images/shark1_hr_{lr_shape}.jpg', cv2.cvtColor(uimg, cv2.COLOR_BGR2RGB))
    
    return model

class EgvsrUpscalerService(BaseUpscalerService):
    def __init__(self, lr_level=1, device=0, on_queue=None):
        self.lr_shape = [
            (540, 960),
            (630, 1120),
            (720, 1280)
        ][lr_level]
        self.scale = 4
        self.hr_shape = tuple([i * self.scale for i in self.lr_shape])
        self.device = device
        self.on_queue = on_queue

        super().__init__()

    def proc_init(self):
        print('EgvsrUpscalerService: proc init')
        self.model = build_egvsr_model(
            np.empty((*self.lr_shape, 3), dtype=np.uint8), lr_shape=self.lr_shape,
            device=self.device, bsize=1, jit_mode='tensorrt', do_benchmark=False
        ).eval()
        self.lr_prev = None
        self.hr_prev = None
        print('EgvsrUpscalerService: model loaded')
    
    def proc_cleanup(self):
        pass
    
    def upscale(self, img: torch.Tensor):
        assert isinstance(img, torch.Tensor)
        with torch.no_grad():
            img = img.to(self.device, non_blocking=True).permute(2,0,1).unsqueeze(0)
            img = img / 255.0
            lr_curr = torch.nn.functional.interpolate(img, size=self.lr_shape, mode='area')
            if self.lr_prev is None:
                self.lr_prev = lr_curr
            if self.hr_prev is None:
                self.hr_prev = torch.nn.functional.interpolate(img, size=self.hr_shape, mode='bicubic')
            
            hr_curr = self.model(lr_curr, self.lr_prev, self.hr_prev)

            self.hr_prev = hr_curr
            self.lr_prev = lr_curr

            return (torch.clamp(hr_curr, 0, 1) * 255).detach()[0].permute(1,2,0).to('cpu', non_blocking=True)
        
if __name__ == '__main__':
    import os
    def handler(entry: UpscalerQueueEntry):
        print('upscaled', entry.img.shape, entry.step)
    
    service = EgvsrUpscalerService(
        lr_level=0,
        on_queue=handler
    )
    service.start()
    for i, filename in enumerate(os.listdir('./saves/frames/')):
        if filename.endswith('.png'):
            frame = cv2.imread(f'./saves/frames/{filename}')
            frame = torch.tensor(frame, dtype=torch.float32)
            service.push_frame(frame, i, timeout=30)
    service.wait_for_job_clear()
    service.stop()
    #service.join(timeout=None)