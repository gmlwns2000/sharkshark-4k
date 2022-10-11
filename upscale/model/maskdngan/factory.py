import cv2, time
import torch
from torch import nn
import numpy as np
from upscale.model.maskdngan.model import DenoiseNet
from data.dataloader import backWarp

def forward(flowComp, denoiser, trainBackWarp, input):
    input_raw_left  = input[:,   :4]
    input_raw_centr = input[:,  4:8]
    input_raw_right = input[:, 8:12]

    with torch.no_grad():
        # Calculate optical flows - RAFT
        print('hye', input_raw_left.shape)
        flow_left, context_centr = flowComp(input_raw_centr, input_raw_left, 32)
        flow_right, _ = flowComp(input_raw_centr, input_raw_right, 32)
        input_raw_left  = trainBackWarp(input_raw_left, flow_left[-1])
        input_raw_right = trainBackWarp(input_raw_right, flow_right[-1])

        # Denoise
        input_raw = torch.cat((input_raw_left, input_raw_centr, input_raw_right, context_centr), 1)
        out_raw = input_raw_centr + denoiser(input_raw)
    
    return out_raw

import argparse
from core.raft import RAFT

class DenoiseWrapper(nn.Module):
    def __init__(self, model, input_shape) -> None:
        super().__init__()
        parser = argparse.ArgumentParser(description='Testing')
        parser.add_argument('--model', dest='model', type=str, default='final', help='model type')
        parser.add_argument('--gpu_id', dest='gpu_id', type=int, default=0, help='gpu id')
        parser.add_argument('--output_dir', type=str, default='./results/indoor/', help='output path')
        parser.add_argument('--vis_data', type=bool, default=True, help='whether to visualize noisy and gt data')
        parser.add_argument('--small', action='store_true', help='use small model')
        parser.add_argument('--epsilon', type=float, default=1e-8)
        parser.add_argument('--dropout', type=float, default=0.0)
        parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
        args = parser.parse_args("")

        self.device = 0
        self.model = model
        self.padder = nn.ZeroPad2d((0, 0, 18, 18))
        self.flowComp = RAFT(args)
        self.flowComp = self.flowComp.to(self.device).eval()
        self.flowComp.load_state_dict(torch.load('core/407000_raft.pth'), strict=False)
        for param in self.flowComp.parameters():
            param.requires_grad = False
        
        self.trainBackWarp = backWarp((input_shape[1], input_shape[0]), self.device)
        self.trainBackWarp = self.trainBackWarp.to(self.device)
    
    def forward(self, x0, x1, x2):
        #x : 1 3 H W
        N, C, H , W = x0.shape
        assert N == 1 and C == 3

        inp = torch.empty((1, 12, H, W), dtype=x0.dtype, device=x0.device)
        inp[:,:3,:,:] = x0
        inp[:,3:4,:,:] = x0[:,0:1,:,:]
        inp[:,4:7,:,:] = x1
        inp[:,7:8,:,:] = x1[:,0:1,:,:]
        inp[:,8:11,:,:] = x2
        inp[:,11:12,:,:] = x2[:,0:1,:,:]

        #inp = self.padder(inp)

        output = forward(self.flowComp, self.model, self.trainBackWarp, inp)

        return output[:,:3, :, :]

def build_model(input_shape=(720,1280), device=0):
    model = DenoiseNet(140)
    state = torch.load('./upscale/model/maskdngan/final.ckpt', map_location='cpu')
    model.load_state_dict(state['state_dict'])
    del state

    #jit

    model = DenoiseWrapper(model, input_shape=input_shape)
    return model.to(device).eval()

if __name__ == '__main__':
    input_shape = (360, 640)
    cmodel = build_model(input_shape = input_shape)

    frame = cv2.imread('./samples/images/shark1.png')
    frame = cv2.resize(frame, (input_shape[1], input_shape[0]))
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    inp = torch.tensor(frame,device=0,dtype=torch.float32).permute(2,0,1).unsqueeze(0) / 255.0

    # inp_ = torch.empty((inp.shape[0], 4, inp.shape[2], inp.shape[3]), device=0, dtype=torch.float32)
    # inp_.fill_(0.03)
    # inp_[:,:3,:,:] = inp
    # inp = inp_
    # inp = inp.unsqueeze(1)
    # n_clips = 1
    # inp = inp.repeat(1, n_clips, 1, 1, 1)
    print(inp.shape)

    N = 3
    for i in range(N):
        with torch.no_grad(), torch.cuda.amp.autocast():
            output = cmodel(inp, inp, inp)
        if (i%10)==0: print('warm', i)

    N = 10
    t = time.time()
    for i in range(N):
        with torch.no_grad(), torch.cuda.amp.autocast():
            output = cmodel(inp, inp, inp)
        if (i%10)==0: print(i)
    print((time.time()-t)/N)

    denoise = (torch.clamp(output[0],0,1).permute(1,2,0)*255).cpu().numpy().astype(np.uint8)
    #plt.imshow(denoise)
    denoise = cv2.cvtColor(denoise, cv2.COLOR_RGB2BGR)
    cv2.imwrite('output.png', denoise)

    diff = ((denoise.astype(np.float32) - cv2.cvtColor(frame, cv2.COLOR_RGB2BGR).astype(np.float32))*10 + 127).astype(np.uint8)
    cv2.imwrite('output_diff.png', diff)