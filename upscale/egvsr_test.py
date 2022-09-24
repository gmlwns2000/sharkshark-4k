import time, torch, cv2, gc, tqdm
from wsgiref.headers import tspecials
from matplotlib import pyplot as plt
import numpy as np
from upscale.model.egvsr.egvsr import FRNet

torch.hub._validate_not_a_forked_repo=lambda a,b,c: True
torch.backends.cudnn.benchmark = True

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

if __name__ == '__main__':
    import os

    img = cv2.imread('./samples/images/shark1.png')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    lr_shape = (540, 960) #25ms
    #lr_shape = (630, 1120) #35ms
    #lr_shape = (720, 1280) #45ms
    model = build_egvsr_model(
        img, device=0, lr_shape=lr_shape, bsize=1, jit_mode='tensorrt', do_benchmark=False
    )
    gc.collect()
    torch.cuda.empty_cache()
    
    height, width = tuple([i * 4 for i in lr_shape])

    from PIL import Image
    from subprocess import Popen, PIPE
    from imutils.video import VideoStream
    from imutils.object_detection import non_max_suppression
    from imutils import paths
    # fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    path = 'saves/egvsr_test.mp4'
    # out = cv2.VideoWriter(', fourcc, 24.0, (int(width), int(height)))

    # ffmpeg setup
    with Popen(['ffmpeg', '-y', '-f', 'image2pipe', '-vcodec', 'mjpeg', '-r', '24', '-i', '-',
        '-vcodec', 'mpeg4', '-qscale', '5', '-r', '24', path], stdin=PIPE) as p:

        frames_dir = './saves/frames/'
        hr_prev = None
        lr_prev = None
        def batch_to_cv2mat(t, idx=0):
            return torch.clamp(t[idx].detach().permute(1,2,0) * 255, 0, 255).to('cpu', non_blocking=True).numpy().astype(np.uint8)
        for file_name in tqdm.tqdm(sorted(os.listdir(frames_dir))):
            if file_name.endswith('.png'):
                fname = os.path.join(frames_dir, file_name)

                img = cv2.cvtColor(cv2.imread(fname), cv2.COLOR_BGR2RGB)
                img = torch.tensor(img, dtype=torch.float32).to(0, non_blocking=True).permute(2,0,1).unsqueeze(0) / 255.0
                lr_curr = torch.nn.functional.interpolate(img, size=lr_shape, mode='area')
                if lr_prev is None:
                    lr_prev = lr_curr
                if hr_prev is None:
                    hr_prev = torch.nn.functional.interpolate(img, size=tuple([i * 4 for i in lr_shape]), mode='bicubic')
                
                with torch.no_grad():
                    hr_curr = model(lr_curr, lr_prev, hr_prev)
                
                frame = batch_to_cv2mat(hr_curr)
                #print(frame.shape, height, width)
                #out.write(frame)
                #frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                im = Image.fromarray(frame)
                im.save(p.stdin, 'JPEG')
                hr_prev = hr_prev
                lr_prev = lr_curr
        
        #out.release()
            