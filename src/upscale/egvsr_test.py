from .egvsr_upscaler import *

if __name__ == '__main__':
    import os

    img = cv2.imread('./samples/images/shark1.png')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    #lr_shape = (540, 960) #25ms
    #lr_shape = (630, 1120) #35ms
    lr_shape = (720, 1280) #45ms
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
            