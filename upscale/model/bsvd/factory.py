import cv2
import torch
from torch import nn
import time
import numpy as np
from upscale.model.bsvd.model import *
from matplotlib import pyplot as plt

class JitWrapper(nn.Module):
    def __init__(self, module, half_convert) -> None:
        super().__init__()
        self.module = module
        self.half_convert = half_convert
    
    def forward(self, *args):
        if self.half_convert:
            args = [a.half() for a in args]
        return self.module(*args)

def build_model(device=0):
    model = BSVD(
        chns=[64,128,256], mid_ch=64, shift_input=False, 
        norm='none', interm_ch=64, act='relu6', 
        pretrain_ckpt='./upscale/model/bsvd/bsvd-64.pth'
    )
    model = model.to(device).eval()

    jit_mode = 'ds'
    if jit_mode == 'jit':
        model_ft = model
        lr_curr = torch.empty((1, 1, 4, 540, 960), dtype=torch.float32, device=device)
        traced_model = torch.jit.script(model_ft, (lr_curr))
        model = traced_model
        f_s_m = torch._C._freeze_module(traced_model._c)
        f_s_m.dump()
    elif jit_mode == 'ds':
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
        model = JitWrapper(model, half_convert=True)
        amp_enabled = False
        half_convert = True
    elif jit_mode == 't2trt':
        from torch2trt import torch2trt

        lr_curr = torch.empty((1, 1, 4, 540, 960), dtype=torch.float32, device=device)
        model_trt = torch2trt(model, [lr_curr])
        model = model_trt
    elif jit_mode == 'trt':
        import torch_tensorrt, os
        torch_tensorrt.logging.set_reportable_log_level(torch_tensorrt.logging.Level.Debug)
        version = '0'

        lr_curr = torch.empty((1, 1, 4, 720, 1280), dtype=torch.float32, device=device)
        N, F, C, H, W = lr_curr.shape

        ts_path = f"./saves/models/bsvd_{version}_{N}x{F}x{C}x{W}x{H}.pts"

        if os.path.exists(ts_path):
            model = torch.jit.load(ts_path)
        else:
            print('Bsvd.build_model: Compiling...')
            trt_model = torch_tensorrt.compile(model, 
                inputs= [
                    torch_tensorrt.Input(lr_curr.shape),
                ],
                enabled_precisions= { torch_tensorrt.dtype.half } # Run with FP16
            )
            model = trt_model
            torch.jit.save(model, ts_path)
        torch_tensorrt.logging.set_reportable_log_level(torch_tensorrt.logging.Level.Warning)

    skip_repeat = True
    amp_enabled = False
    half_convert = False

    return model

if __name__ == '__main__':
    cmodel = build_model()

    frame = cv2.imread('./samples/images/shana.png')
    frame = cv2.resize(frame, (640, 360))
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    inp = torch.tensor(frame,device=0,dtype=torch.float32).permute(2,0,1).unsqueeze(0) / 255.0

    inp_ = torch.empty((inp.shape[0], 4, inp.shape[2], inp.shape[3]), device=0, dtype=torch.float32)
    inp_.fill_(0.03)
    inp_[:,:3,:,:] = inp
    inp = inp_
    inp = inp.unsqueeze(1)
    n_clips = 1
    inp = inp.repeat(1, n_clips, 1, 1, 1)
    print(inp.shape)

    for i in range(50):
        with torch.no_grad(), torch.cuda.amp.autocast():
            output = cmodel(inp)
        if (i%10)==0: print('warm', i)

    t = time.time()
    for i in range(100):
        with torch.no_grad(), torch.cuda.amp.autocast():
            output = cmodel(inp)
        if (i%10)==0: print(i)
    print(time.time()-t)

    denoise = (torch.clamp(output[0,output.shape[1]//2,:,:,:],0,1).permute(1,2,0)*255).cpu().numpy().astype(np.uint8)
    plt.imshow(denoise)
    denoise = cv2.cvtColor(denoise, cv2.COLOR_RGB2BGR)
    cv2.imwrite('output.png', denoise)

    diff = ((denoise.astype(np.float32) - cv2.cvtColor(frame, cv2.COLOR_RGB2BGR).astype(np.float32))*10 + 127).astype(np.uint8)
    cv2.imwrite('output_diff.png', diff)
