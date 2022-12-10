from dataclasses import dataclass
import torch
import os
from torch import nn

from basicsr.archs.rrdbnet_arch import RRDBNet
from basicsr.utils.download_util import load_file_from_url

from realesrgan import RealESRGANer
# from realesrgan.archs.srvgg_arch import SRVGGNetCompact

# from basicsr.utils.registry import ARCH_REGISTRY
from torch import nn as nn
from torch.nn import functional as F


# @ARCH_REGISTRY.register()
class SRVGGNetCompact(nn.Module):
    """A compact VGG-style network structure for super-resolution.

    It is a compact network structure, which performs upsampling in the last layer and no convolution is
    conducted on the HR feature space.

    Args:
        num_in_ch (int): Channel number of inputs. Default: 3.
        num_out_ch (int): Channel number of outputs. Default: 3.
        num_feat (int): Channel number of intermediate features. Default: 64.
        num_conv (int): Number of convolution layers in the body network. Default: 16.
        upscale (int): Upsampling factor. Default: 4.
        act_type (str): Activation type, options: 'relu', 'prelu', 'leakyrelu'. Default: prelu.
    """

    def __init__(self, num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=16, upscale=4, act_type='prelu'):
        super(SRVGGNetCompact, self).__init__()
        self.num_in_ch = num_in_ch
        self.num_out_ch = num_out_ch
        self.num_feat = num_feat
        self.num_conv = num_conv
        self.upscale = upscale
        self.act_type = act_type

        self.body = nn.ModuleList()
        # the first conv
        self.body.append(nn.Conv2d(num_in_ch, num_feat, 3, 1, 1))
        # the first activation
        if act_type == 'relu':
            activation = nn.ReLU(inplace=True)
        elif act_type == 'prelu':
            activation = nn.PReLU(num_parameters=num_feat)
        elif act_type == 'leakyrelu':
            activation = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.body.append(activation)

        # the body structure
        for _ in range(num_conv):
            self.body.append(nn.Conv2d(num_feat, num_feat, 3, 1, 1))
            # activation
            if act_type == 'relu':
                activation = nn.ReLU(inplace=True)
            elif act_type == 'prelu':
                activation = nn.PReLU(num_parameters=num_feat)
            elif act_type == 'leakyrelu':
                activation = nn.LeakyReLU(negative_slope=0.1, inplace=True)
            self.body.append(activation)

        # the last conv
        self.body.append(nn.Conv2d(num_feat, num_out_ch * upscale * upscale, 3, 1, 1))
        # upsample
        self.upsampler = nn.PixelShuffle(upscale)

    def forward(self, x):
        out = x
        # for i in range(0, len(self.body)):
        #     out = self.body[i](out)
        for i, module in enumerate(self.body):
            out = module(out)

        out = self.upsampler(out)
        # add the nearest upsampled image, so that the network learns the residual
        base = F.interpolate(x, scale_factor=float(self.upscale), mode='nearest')
        out += base
        return out


@dataclass
class ArgsData:
    # model_name = 'realesr-animevideov3'
    model_name = 'realesr-general-x4v3'
    denoise_strength = 0.5
    outscale = 4
    model_path = None
    suffix = 'out'
    tile = 0
    tile_pad = 10
    pre_pad = 0
    face_enhance = False
    alpha_upscaler = 'realesrgan'
    ext = 'auto'

class RealESRGANWrapper(nn.Module):
    def __init__(self, up, outscale):
        super().__init__()
        self.up = up
        self.scale = outscale
    def forward(self, x):
        return self.up.enhance(x, outscale=self.scale)[0]

def build_model(factor=4, device=0, input_shape=(720,1280), batch_size=8, denoise_rate=0.5):
    args = ArgsData()
    args.denoise_strength = denoise_rate
    args.model_name = args.model_name.split('.')[0]
    if args.model_name == 'RealESRGAN_x4plus':  # x4 RRDBNet model
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
        netscale = 4
        file_url = ['https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth']
    elif args.model_name == 'RealESRNet_x4plus':  # x4 RRDBNet model
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
        netscale = 4
        file_url = ['https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.1/RealESRNet_x4plus.pth']
    elif args.model_name == 'RealESRGAN_x4plus_anime_6B':  # x4 RRDBNet model with 6 blocks
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=6, num_grow_ch=32, scale=4)
        netscale = 4
        file_url = ['https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth']
    elif args.model_name == 'RealESRGAN_x2plus':  # x2 RRDBNet model
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2)
        netscale = 2
        file_url = ['https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth']
    elif args.model_name == 'realesr-animevideov3':  # x4 VGG-style model (XS size)
        model = SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=16, upscale=4, act_type='prelu')
        netscale = 4
        file_url = ['https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-animevideov3.pth']
    elif args.model_name == 'realesr-general-x4v3':  # x4 VGG-style model (S size)
        model = SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=32, upscale=4, act_type='prelu')
        netscale = 4
        file_url = [
            'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-wdn-x4v3.pth',
            'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth'
        ]

    # determine model paths
    if args.model_path is not None:
        model_path = args.model_path
    else:
        model_path = os.path.join('weights', args.model_name + '.pth')
        if not os.path.isfile(model_path):
            ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
            for url in file_url:
                # model_path will be updated
                model_path = load_file_from_url(
                    url=url, model_dir=os.path.join(ROOT_DIR, 'weights'), progress=True, file_name=None)

    # use dni to control the denoise strength
    dni_weight = None
    if args.model_name == 'realesr-general-x4v3' and args.denoise_strength != 1:
        wdn_model_path = model_path.replace('realesr-general-x4v3', 'realesr-general-wdn-x4v3')
        model_path = [model_path, wdn_model_path]
        dni_weight = [args.denoise_strength, 1 - args.denoise_strength]

    # restorer
    upsampler = RealESRGANer(
        scale=netscale,
        model_path=model_path,
        dni_weight=dni_weight,
        model=model,
        tile=args.tile,
        tile_pad=args.tile_pad,
        pre_pad=args.pre_pad,
        half=True,
        gpu_id=device)
    model = upsampler.model
    # model = RealESRGANWrapper(upsampler, args.outscale)
    
    model = model.eval().to(device)

    half_convert = False
    jit_mode = 'trt'
    if jit_mode == 'jit':
        model_ft = model
        lr_curr = torch.empty((batch_size, 3, 720, 1280), dtype=torch.float32, device=device)
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

        lr_curr = torch.empty((batch_size, 3, 720, 1280), dtype=torch.float32, device=device)
        model_trt = torch2trt(model, [lr_curr])
        model = model_trt
    elif jit_mode == 'trt':
        import torch_tensorrt
        torch_tensorrt.logging.set_reportable_log_level(torch_tensorrt.logging.Level.Debug)
        version = '0'

        # lr_curr = torch.empty((3, 1, 720, 1280), dtype=torch.half, device=device)
        N, C, H, W = (batch_size, 3, *input_shape)

        ts_path = f"./saves/models/realesrcnn_{version}_{N}x{C}x{W}x{H}.pts"

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
        amp_enabled = False
        half_convert = True

    if half_convert:
        model = JitWrapper(model, half_convert)
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
    import torch, time, tqdm, cv2
    import numpy as np

    input_shape = (720,1280)
    frame = cv2.imread('./samples/images/shana.png')
    frame = cv2.resize(frame, (input_shape[1], input_shape[0]))
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    inp = torch.tensor(frame,device=0,dtype=torch.float32).permute(2,0,1).unsqueeze(0) / 255.0

    model = build_model(input_shape=input_shape)
    batch = inp #(1,3,*input_shape), dtype=torch.float32, device=0)
    
    def run(n=100):
        t = time.time()
        for i in tqdm.tqdm(range(n)):
            with torch.no_grad(), torch.cuda.amp.autocast():
                output = model(batch)
        print(output.shape)
        return time.time() - t, output
    
    N = 10
    _, output = run(5)
    print(run(N)[0]/N*1000, 'ms')
    
    denoise = (torch.clamp(output[0,:,:,:],0,1).permute(1,2,0)*255).cpu().numpy().astype(np.uint8)
    #print(output, denoise, denoise.shape, np.max(denoise.reshape(input_shape[0]*input_shape[1],3), axis=0))
    #plt.imshow(denoise)
    denoise = cv2.cvtColor(denoise, cv2.COLOR_RGB2BGR)
    cv2.imwrite('output_sr.png', denoise)
    