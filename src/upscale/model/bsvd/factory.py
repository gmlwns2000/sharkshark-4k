import cv2
from pkg_resources import working_set
import torch
from torch import nn
import time
import numpy as np
from . import model as bsvd
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

def build_model(device=0, input_shape=(360, 640), jit_mode='ds'):
    if jit_mode == 'trt_vol':
        return __build_model_volatile(device=device, input_shape=input_shape)
    
    # bsvd.set_res(input_shape)
    # model = bsvd.BSVD(
    #     chns=[64,128,256], mid_ch=64, shift_input=False, 
    #     norm='none', interm_ch=64, act='relu6', 
    #     pretrain_ckpt='./upscale/model/bsvd/bsvd-64.pth'
    # )
    model = bsvd.BSVD(
        chns=[32,64,128], mid_ch=32, shift_input=False, 
        norm='none', interm_ch=30, act='relu6', 
        pretrain_ckpt='./upscale/model/bsvd/bsvd-32.pth'
    )
    model = model.to(device).eval()
    
    if jit_mode == 'jit':
        model_ft = model
        lr_curr = torch.empty((1, 1, 4, *input_shape), dtype=torch.float32, device=device)
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
    elif jit_mode == 'trt':
        import torch_tensorrt, os
        torch_tensorrt.logging.set_reportable_log_level(torch_tensorrt.logging.Level.Debug)
        version = '1'

        lr_curr = torch.empty((1, 1, 4, *input_shape), dtype=torch.float32, device=device)
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

    return model

def __build_model_volatile(device=0, input_shape=(360, 640)):
    import torch_tensorrt, os
    import upscale.model.bsvd.model_volatile as bsvd_vol

    torch_tensorrt.logging.set_reportable_log_level(torch_tensorrt.logging.Level.Warning)

    H, W = input_shape
    bsvd_vol.set_res((H, W))

    model = bsvd_vol.BSVD(
        chns=[64,128,256], mid_ch=64, shift_input=False, 
        norm='none', interm_ch=64, act='relu6', 
        pretrain_ckpt='./upscale/model/bsvd/bsvd-64.pth'
    )
    model = model.to(device).eval()

    def jit(model, inp_shape=input_shape, save_path=None):
        if os.path.exists(save_path):
            return torch.jit.load(save_path)
        else:
            trt_model = torch_tensorrt.compile(model, 
                inputs = [
                    torch_tensorrt.Input(inp_shape, dtype=torch.half),
                ],
                enabled_precisions= { torch_tensorrt.dtype.half },
                require_full_compilation = False,
                min_block_size = 3,
                workspace_size = 1<<30,
            )
            if save_path is not None:
                torch.jit.save(trt_model, save_path)
            return trt_model
    
    def jit_bsvd(bsvd_model, image_shape=(720,1280)):
        H, W = image_shape
        F = 1
        inp_shape = (1,F,4,H,W)

        version = '0'
        name = f'saves/models/bsvd_{version}_({F}x{H}x{W})'
        
        model = bsvd_model.half()
        
        model.temp1.inc = jit(model.temp1.inc, inp_shape=(1,4,H,W), save_path=name+'_t1inc.pth')
        model.temp1.downc0 = jit(model.temp1.downc0, inp_shape=(1,64,H,W), save_path=name+'_t1d0.pth')
        model.temp1.downc1 = jit(model.temp1.downc1, inp_shape=(1,128,H//2,W//2), save_path=name+'_t1d1.pth')
        model.temp1.upc2 = jit(model.temp1.upc2, inp_shape=(1,256,H//4,W//4), save_path=name+'_t1u2.pth')
        model.temp1.upc1 = jit(model.temp1.upc1, inp_shape=(1,128,H//2,W//2), save_path=name+'_t1u1.pth')
        model.temp1.outc = jit(model.temp1.outc, inp_shape=(1,64,H,W), save_path=name+'_t1out.pth')

        model.temp2.inc = jit(model.temp2.inc, inp_shape=(1,64,H,W), save_path=name+'_t2inc.pth')
        model.temp2.downc0 = jit(model.temp2.downc0, inp_shape=(1,64,H,W), save_path=name+'_t2d0.pth')
        model.temp2.downc1 = jit(model.temp2.downc1, inp_shape=(1,128,H//2,W//2), save_path=name+'_t2d1.pth')
        model.temp2.upc2 = jit(model.temp2.upc2, inp_shape=(1,256,H//4,W//4), save_path=name+'_t2u2.pth')
        model.temp2.upc1 = jit(model.temp2.upc1, inp_shape=(1,128,H//2,W//2), save_path=name+'_t2u1.pth')
        model.temp2.outc = jit(model.temp2.outc, inp_shape=(1,64,H,W), save_path=name+'_t2out.pth')

        return model
    
    model = JitWrapper(jit_bsvd(model, (H, W)), True)

    torch_tensorrt.logging.set_reportable_log_level(torch_tensorrt.logging.Level.Warning)

    return model

if __name__ == '__main__':
    input_shape = (1080, 1920)
    # bsvd.set_res(input_shape)

    cmodel = build_model(input_shape=input_shape)

    frame = cv2.imread('./samples/images/shana.png')
    frame = cv2.resize(frame, (input_shape[1], input_shape[0]))
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

    N = 5
    for i in range(N):
        with torch.no_grad(), torch.cuda.amp.autocast():
            output = cmodel(inp)
        if (i%10)==0: print('warm', i)

    t = time.time()
    N = 100
    for i in range(N):
        with torch.no_grad(), torch.cuda.amp.autocast():
            output = cmodel(inp)
        if (i%10)==0: print(i)
    print((time.time()-t)/N)

    denoise = (torch.clamp(output[0,output.shape[1]//2,:,:,:],0,1).permute(1,2,0)*255).cpu().numpy().astype(np.uint8)
    #print(output, denoise, denoise.shape, np.max(denoise.reshape(input_shape[0]*input_shape[1],3), axis=0))
    #plt.imshow(denoise)
    denoise = cv2.cvtColor(denoise, cv2.COLOR_RGB2BGR)
    cv2.imwrite('output.png', denoise)

    diff = ((denoise.astype(np.float32) - cv2.cvtColor(frame, cv2.COLOR_RGB2BGR).astype(np.float32))*10 + 127).astype(np.uint8)
    cv2.imwrite('output_diff.png', diff)
