import torch
from torch import nn
import upscale.model.fsrcnn.model as models

def build_model(factor=4, device=0, input_shape=(720,1280)):
    model = models.FSRCNN(4)
    if factor == 4:
        state = torch.load('upscale/model/fsrcnn/fsrcnn_x4-T91.pth', map_location='cpu')
    elif factor == 2:
        state = torch.load('upscale/model/fsrcnn/fsrcnn_x2-T91.pth', map_location='cpu')
    else:
        raise Exception()
    model.load_state_dict(state['state_dict'])
    del state
    model = model.eval().to(device)

    jit_mode = 'trt'
    if jit_mode == 'jit':
        model_ft = model
        lr_curr = torch.empty((3, 1, 720, 1280), dtype=torch.float32, device=device)
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

        lr_curr = torch.empty((3, 1, 720, 1280), dtype=torch.float32, device=device)
        model_trt = torch2trt(model, [lr_curr])
        model = model_trt
    elif jit_mode == 'trt':
        import torch_tensorrt, os
        torch_tensorrt.logging.set_reportable_log_level(torch_tensorrt.logging.Level.Debug)
        version = '2'

        # lr_curr = torch.empty((3, 1, 720, 1280), dtype=torch.half, device=device)
        N, C, H, W = (3, 1, *input_shape)

        ts_path = f"./saves/models/fsrcnn_{version}_{N}x{C}x{W}x{H}.pts"

        if os.path.exists(ts_path):
            model = torch.jit.load(ts_path)
        else:
            print('FsrcnnUpscaler.build_model: Compiling...')
            trt_model = torch_tensorrt.compile(model, 
                inputs= [
                    torch_tensorrt.Input((N, C, H, W)),
                ],
                enabled_precisions= { torch_tensorrt.dtype.half } # Run with FP16
            )
            model = trt_model
            torch.jit.save(model, ts_path)
        torch_tensorrt.logging.set_reportable_log_level(torch_tensorrt.logging.Level.Warning)

    return model

class JitWrapper(nn.Module):
    def __init__(self, module, half_convert) -> None:
        super().__init__()
        self.module = module
        self.half_convert = half_convert
    
    def forward(self, *args):
        if self.half_convert:
            args = [a.half() for a in args]
        return self.module(*args)

if __name__ == '__main__':
    import torch, time, tqdm

    inp_shape = (1080,1920)

    model = build_model(input_shape=inp_shape)
    batch = torch.zeros((3,1,*inp_shape), dtype=torch.float32, device=0)
    
    def run(n=100):
        t = time.time()
        for i in tqdm.tqdm(range(n)):
            with torch.no_grad(), torch.cuda.amp.autocast():
                output = model(batch)
        print(output.shape)
        return time.time() - t
    
    run(30)
    print(run()/100*1000)