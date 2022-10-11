import cv2
import torch
from torch import nn
import time
import numpy as np
import upscale.model.bsvd.model as bsvd
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

def build_model(device=0, input_shape=(360, 640)):
    model = bsvd.BSVD(
        chns=[64,128,256], mid_ch=64, shift_input=False, 
        norm='none', interm_ch=64, act='relu6', 
        pretrain_ckpt='./upscale/model/bsvd/bsvd-64.pth'
    )
    model = model.to(device).eval()

    jit_mode = 'ds'
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
        amp_enabled = False
        half_convert = True
    elif jit_mode == 't2trt':
        from torch2trt import torch2trt
        from torch2trt import tensorrt_converter
        from torch2trt.torch2trt import add_missing_trt_tensors

        @tensorrt_converter('torch.nn.PixelShuffle.forward')
        def convert_PixelShuffle(ctx):

            input = ctx.method_args[1]
            module = ctx.method_args[0]
            scale_factor =  module.upscale_factor

            input_trt = add_missing_trt_tensors(ctx.network, [input])[0]
            output = ctx.method_return

            batch_size, in_channels, in_height, in_width = input.shape

            assert scale_factor >= 1

            out_channels = in_channels // (scale_factor * scale_factor)
            out_height = in_height * scale_factor
            out_width = in_width * scale_factor

            layer_1 = ctx.network.add_shuffle(input_trt)
            layer_1.reshape_dims = (out_channels, scale_factor, scale_factor, in_height, in_width)

            layer_2 = ctx.network.add_shuffle(layer_1.get_output(0))
            layer_2.first_transpose = (0, 3, 1, 4, 2)
            layer_2.reshape_dims = (out_channels, out_height, out_width)

            output._trt = layer_2.get_output(0)

        lr_curr = torch.empty((1, 1, 4, *input_shape), dtype=torch.float32, device=device)
        model_trt = torch2trt(model, [lr_curr], fp16_mode=True)
        model = model_trt
    elif jit_mode == 'trt':
        import torch_tensorrt, os
        torch_tensorrt.logging.set_reportable_log_level(torch_tensorrt.logging.Level.Debug)
        version = '0'

        lr_curr = torch.empty((1, 1, 4, *input_shape), dtype=torch.float32, device=device)
        N, F, C, H, W = lr_curr.shape
        # with torch.no_grad():
        #     model(lr_curr)

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
    input_shape = (720, 1280)
    bsvd.set_res(input_shape)

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
