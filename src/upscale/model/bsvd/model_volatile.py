import time
from typing import List, Optional, Type, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def extract_dict(ckpt_state, string_name='base_model.temp1.', replace_name=''):
    m_dict = {}
    for k, v in ckpt_state.items():
        if string_name in k:
            m_dict[k.replace(string_name, replace_name)] = v
    return m_dict
        
def replace_dict(ckpt_state, string_name='base_model.temp1.', replace_name=''):
    m_dict = {}
    for k, v in ckpt_state.items():
        # if string_name in k:
        m_dict[k.replace(string_name, replace_name)] = v
    return m_dict

class ShiftConv(nn.Module):
    def __init__(self,
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            bias
        ) -> None:
        super(ShiftConv, self).__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            bias=bias
        )
    def forward(self, left_fold_2fold:torch.Tensor, center:torch.Tensor, right:torch.Tensor):
        #assert not(left_fold_2fold is None or center is None)
        fold_div = 8
        n, c, h, w = center.size()
        fold = c//fold_div
        #assert left_fold_2fold.size()[1] == fold
        return  self.conv(torch.cat([ right[:, :fold, :, :],
                                     left_fold_2fold, 
                                     center[:, 2*fold:, :, :]], dim=1))
        # return  self.conv(torch.cat([left[:, fold: 2*fold, :, :], center[:, 2*fold:, :, :], right[:, :fold, :, :]], dim=1))

def slice(x: torch.Tensor, fold:int):
    #assert x is not None
    return x[:, fold:2*fold, :, :]

bsvd_input_res = (720, 1280)

def set_res(v):
    global bsvd_input_res
    bsvd_input_res = v

biconv_idx = 0
biconv_expected = {
    (720, 1280): [
        (1, 128, 360, 640),
        (1, 128, 360, 640),
        (1, 256, 180, 320),
        (1, 256, 180, 320),
        (1, 256, 180, 320),
        (1, 256, 180, 320),
        (1, 128, 360, 640),
        (1, 128, 360, 640),
        (1, 128, 360, 640),
        (1, 128, 360, 640),
        (1, 256, 180, 320),
        (1, 256, 180, 320),
        (1, 256, 180, 320),
        (1, 256, 180, 320),
        (1, 128, 360, 640),
        (1, 128, 360, 640),
    ],
    (360, 640): [
        (1, 128, 180, 320),
        (1, 128, 180, 320),
        (1, 256, 90, 160),
        (1, 256, 90, 160),
        (1, 256, 90, 160),
        (1, 256, 90, 160),
        (1, 128, 180, 320),
        (1, 128, 180, 320),
        (1, 128, 180, 320),
        (1, 128, 180, 320),
        (1, 256, 90, 160),
        (1, 256, 90, 160),
        (1, 256, 90, 160),
        (1, 256, 90, 160),
        (1, 128, 180, 320),
        (1, 128, 180, 320),
    ],
}
def add_res_expected(data, from_res, to_res):
    d720 = data[from_res]
    d1080 = []
    for it in d720:
        shape = list(it[:-2])
        shape.append(int(it[-2]*(to_res[0]/from_res[0])))
        shape.append(int(it[-1]*(to_res[0]/from_res[0])))
        d1080.append(tuple(shape))
    data[to_res] = d1080
add_res_expected(biconv_expected, (720,1280), (1080,1920))
add_res_expected(biconv_expected, (720,1280), (900,1600))

class BiBufferConvVolatile(nn.Module):
    left_fold_2fold: torch.Tensor
    center: torch.Tensor

    def __init__(self,
            in_channels,
            out_channels,
            kernel_size,
            stride=1,
            padding=0,
            bias=True
        ) -> None:
        global biconv_idx, biconv_expected, bsvd_input_res

        super().__init__()
        self.op = ShiftConv(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            bias
        )
        self.out_channels = out_channels

        self.idx = biconv_idx
        self.last_shape = None
        self.expected_shape = biconv_expected[bsvd_input_res][self.idx]
        biconv_idx += 1

        self.n, self.c, self.h, self.w = self.expected_shape
        self.fold_div = 8
        self.fold = self.c // self.fold_div
        
        self.register_buffer('left_fold_2fold', torch.zeros((self.n, self.fold, self.h, self.w), dtype=torch.float32), False)
        self.register_buffer('center', torch.zeros((self.n, self.c, self.h, self.w), dtype=torch.float32), False)
        
    def forward(self, 
        input_right: torch.Tensor,
    ):
        output =  self.op(self.left_fold_2fold, self.center, input_right)
        
        self.left_fold_2fold.copy_(self.center[:, self.fold:2*self.fold, :, :])
        return output


class MemCvBlockVolatile(nn.Module):
    '''(Conv2d => BN => ReLU) x 2'''

    def __init__(self, in_ch, out_ch, norm='bn', bias=True, act='relu'):
        super().__init__()
        norm_fn = get_norm_function(norm)
        act_fn = get_act_function(act)
        self.c1 = BiBufferConvVolatile(in_ch, out_ch, kernel_size=3,
                            padding=1,bias=bias)
        self.b1 = norm_fn(out_ch)
        self.relu1 = act_fn(inplace=True)
        self.c2 = BiBufferConvVolatile(out_ch, out_ch, kernel_size=3,
                            padding=1,bias=bias)
        self.b2 = norm_fn(out_ch)
        self.relu2 = act_fn(inplace=True)


    def forward(self, x:torch.Tensor):
        x = self.c1(x)
        if x is not None:
            x = self.b1(x)
            x = self.relu1(x)
        x = self.c2(x)
        if x is not None:
            x = self.b2(x)
            x = self.relu2(x)
        return x
    
    def load(self, state_dict):
        state_dict = replace_dict(state_dict, 'net.', 'op.conv.')
        try:
            self.load_state_dict(state_dict)
        except RuntimeError as ex:
            print(ex)
    
    # def reset(self):
    #     self.c1.reset()
    #     self.c2.reset()
    
class CvBlock(nn.Module):
    '''(Conv2d => BN => ReLU) x 2'''

    def __init__(self, in_ch, out_ch, norm='bn', bias=True, act='relu'):
        super(CvBlock, self).__init__()
        norm_fn = get_norm_function(norm)
        act_fn = get_act_function(act)
        self.c1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=bias)
        self.b1 = norm_fn(out_ch)
        self.relu1 = act_fn(inplace=True)
        self.c2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=bias)
        self.b2 = norm_fn(out_ch)
        self.relu2 = act_fn(inplace=True)

    def forward(self, x:torch.Tensor):
        x = self.c1(x)
        x = self.b1(x)
        x = self.relu1(x)
        x = self.c2(x)
        x = self.b2(x)
        x = self.relu2(x)
        return x

def get_norm_function(norm):
    if norm == "bn":
        norm_fn = nn.BatchNorm2d
    elif norm == "in":
        norm_fn = nn.InstanceNorm2d
    elif norm == 'none':
        norm_fn =nn.Identity
    return norm_fn

def get_act_function(act):
    if act == "relu":
        act_fn = nn.ReLU
    elif act == "relu6":
        act_fn = nn.ReLU6
    elif act == 'none':
        act_fn =nn.Identity
    return act_fn

class InputCvBlockVolatile(nn.Module):
    '''(Conv with num_in_frames groups => BN => ReLU) + (Conv => BN => ReLU)'''

    def __init__(self, num_in_frames, out_ch, in_ch=4, norm='bn', bias=True, act='relu', interm_ch = 30, blind=False):
        super().__init__()
        # self.interm_ch = 30
        # if with_sigma: channel_per_frame = 4
        # else: channel_per_frame = 3
        self.interm_ch = interm_ch
        norm_fn = get_norm_function(norm)
        act_fn = get_act_function(act)
        if blind:
            in_ch = 3
        self.convblock = nn.Sequential(
            nn.Conv2d(num_in_frames*in_ch, num_in_frames*self.interm_ch,
                      kernel_size=3, padding=1, groups=num_in_frames, bias=bias),
            norm_fn(num_in_frames*self.interm_ch),
            act_fn(inplace=True),
            nn.Conv2d(num_in_frames*self.interm_ch, out_ch,
                      kernel_size=3, padding=1, bias=bias),
            norm_fn(out_ch),
            act_fn(inplace=True)
        )

    def forward(self, x:torch.Tensor):
        # if x is not None:
        #     self.n, self.in_channels, self.h, self.w = x.size()
        return self.convblock(x)

        # if x is None:
        #     return None
        # else:
        #     return self.convblock(x)
    def load(self, state_dict):
        self.load_state_dict(state_dict)


class DownBlock(nn.Module):
    '''Downscale + (Conv2d => BN => ReLU)*2'''

    def __init__(self, in_ch, out_ch, norm='bn', bias=True, act='relu'):
        super(DownBlock, self).__init__()
        self.out_channels = out_ch
        norm_fn = get_norm_function(norm)
        act_fn = get_act_function(act)
        self.convblock = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3,
                      padding=1, stride=2, bias=bias),
            norm_fn(out_ch),
            act_fn(inplace=True),
        )
        self.memconv = MemCvBlockVolatile(out_ch, out_ch, norm=norm, bias=bias, act=act)
    def reset(self):
        self.memconv.reset()
    def forward(self, x: torch.Tensor):
        if x is not None: 
            #self.n, self.in_channels, self.h, self.w = x.size()
            x = self.convblock(x)
        return self.memconv(x)

    def load(self, ckpt_state):
        self.convblock[0].load_state_dict(extract_dict(ckpt_state,string_name='convblock.0.'))
        self.convblock[1].load_state_dict(extract_dict(ckpt_state,string_name='convblock.1.'))
        self.memconv.load(extract_dict(ckpt_state, string_name='convblock.3.'))


class UpBlock(nn.Module):
    '''(Conv2d => BN => ReLU)*2 + Upscale'''

    def __init__(self, in_ch, out_ch, norm='bn', bias=True, act='relu'):
        super(UpBlock, self).__init__()
        self.memconv = MemCvBlockVolatile(in_ch, in_ch, norm=norm, bias=bias, act=act)
        self.convblock = nn.Sequential(
            nn.Conv2d(in_ch, out_ch*4, kernel_size=3, padding=1, bias=bias),
            nn.PixelShuffle(2)
        )
        self.out_channels = out_ch
    def reset(self):
        self.memconv.reset()
    def forward(self, x: torch.Tensor):
        # if x is None: return None
        # if x is not None:
        #     self.n, self.in_channels, self.h, self.w = x.size()
        x = self.memconv(x)
        if x is not None:
            x = self.convblock(x)
        return x

    def load(self, ckpt_state):
        self.convblock[0].load_state_dict(extract_dict(ckpt_state,string_name='convblock.1.'))
        self.memconv.load(extract_dict(ckpt_state, string_name='convblock.0.'))
        



class OutputCvBlock(nn.Module):
    '''Conv2d => BN => ReLU => Conv2d'''

    def __init__(self, in_ch, out_ch, norm='bn', bias=True, act='relu'):
        super(OutputCvBlock, self).__init__()
        norm_fn = get_norm_function(norm)
        act_fn = get_act_function(act)
        self.convblock = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, kernel_size=3, padding=1, bias=bias),
            norm_fn(in_ch),
            act_fn(inplace=True),
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=bias)
        )

    def forward(self, x: torch.Tensor):
        if x is None: return None
        if x is not None:
            return self.convblock(x)
    def load(self, state_dict):
        self.load_state_dict(state_dict)


mem_idx = 0
mem_shapes = {
    (360, 640) : [
        (1, 3, 360, 640),
        (1, 64, 360, 640),
        (1, 128, 180, 320),
        (1, 3, 360, 640),
        (1, 64, 360, 640),
        (1, 128, 180, 320),
    ],
    (720, 1280) : [
        (1, 3, 720, 1280),
        (1, 64, 720, 1280),
        (1, 128, 360, 640),
        (1, 3, 720, 1280),
        (1, 64, 720, 1280),
        (1, 128, 360, 640),
    ]
}
add_res_expected(mem_shapes, (720,1280), (1080,1920))
add_res_expected(mem_shapes, (720,1280), (900,1600))

class MemSkip(nn.Module):
    #mem_list: List[torch.Tensor]

    def __init__(self):
        global mem_idx, mem_shapes, bsvd_input_res

        super(MemSkip, self).__init__()
        #self.mem_list = []

        self.idx = mem_idx
        mem_idx += 1
        #self.last_shape = None
        self.expected_shape = mem_shapes[bsvd_input_res][self.idx]

        #self.buffer_size = [4]
        self.register_buffer('buffer_size', torch.empty((1,), dtype=torch.int32).fill_(16), False)
        
        #self.count = [0]
        self.register_buffer('count', torch.zeros((1,), dtype=torch.int32), False)

        #self.tail_idx = [0]
        self.register_buffer('tail_idx', torch.zeros((1,), dtype=torch.int32), False)
        
        #self.head_idx = [0]
        self.register_buffer('head_idx', torch.zeros((1,), dtype=torch.int32), False)
        
        self.register_buffer('buffer', torch.empty((self.buffer_size[0], *self.expected_shape)), False)
    
    def push(self, x: torch.Tensor):
        if x is not None:
            #print(self.idx, x.shape, len(self.mem_list))
            # if self.last_shape is not None:
            #     assert self.last_shape == x.shape
            # self.last_shape = x.shape

            #self.mem_list.insert(0,x)
            #return 1

            hey = self.tail_idx[0]
            self.buffer[hey, :, :, :, :] = x
            self.tail_idx[0] = (self.tail_idx[0] + 1) % self.buffer_size[0]
            self.count[0] += 1

            assert self.count[0] < self.buffer_size[0]

            return 1
        else:
            return 0
    
    def pop(self, x: torch.Tensor):
        if x is not None:
            #return self.mem_list.pop()

            item = self.buffer[self.head_idx[0]].clone()
            
            self.head_idx[0] = (self.head_idx[0] + 1) % self.buffer_size[0]
            self.count[0] -= 1
            assert self.count[0] >= 0

            return item
        else:
            return None
            

class DenBlock(nn.Module):
    """ Definition of the denosing block
    Inputs of constructor:
        num_input_frames: int. number of input frames
    Inputs of forward():
        xn: input frames of dim [N, C, H, W], (C=3 RGB)
        noise_map: array with noise map of dim [N, 1, H, W]
    """

    def __init__(self, chns=[32, 64, 128], out_ch=3, in_ch=4, shift_input=False, norm='bn', bias=True,  act='relu', interm_ch=30, blind=False):
        super(DenBlock, self).__init__()
        self.chs_lyr0, self.chs_lyr1, self.chs_lyr2 = chns
        if shift_input:
            self.inc = CvBlock(in_ch=in_ch, out_ch=self.chs_lyr0, norm=norm, bias=bias, act=act)
        else:
            self.inc = InputCvBlockVolatile(
                num_in_frames=1, out_ch=self.chs_lyr0, in_ch=in_ch, norm=norm, bias=bias, act=act, interm_ch=interm_ch, blind=blind)

        self.downc0 = DownBlock(in_ch=self.chs_lyr0, out_ch=self.chs_lyr1, norm=norm, bias=bias, act=act)
        self.downc1 = DownBlock(in_ch=self.chs_lyr1, out_ch=self.chs_lyr2, norm=norm, bias=bias, act=act)
        self.upc2 = UpBlock(in_ch=self.chs_lyr2, out_ch=self.chs_lyr1, norm=norm, bias=bias,    act=act)
        self.upc1 = UpBlock(in_ch=self.chs_lyr1, out_ch=self.chs_lyr0, norm=norm, bias=bias,    act=act)
        self.outc = OutputCvBlock(in_ch=self.chs_lyr0, out_ch=out_ch, norm=norm, bias=bias,     act=act)
        # self.skip1  = MemSkip()
        # self.skip2  = MemSkip()
        # self.skip3  = MemSkip()
        self.reset_params()
    def reset(self):
        self.downc0.reset()
        self.downc1.reset()
        self.upc2.reset()
        self.upc1.reset()
    def load_from(self, ckpt_state):
        self.inc.load(extract_dict(ckpt_state, string_name='inc.'))
        self.downc0.load(extract_dict(ckpt_state, string_name='downc0.'))
        self.downc1.load(  extract_dict(ckpt_state, string_name='downc1.'))
        self.upc2.load(  extract_dict(ckpt_state, string_name='upc2.'))
        self.upc1.load(  extract_dict(ckpt_state, string_name='upc1.'))
        self.outc.load( extract_dict(ckpt_state, string_name='outc.'))

    @staticmethod
    def weight_init(m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, nonlinearity='relu')

    def reset_params(self):
        for _, m in enumerate(self.modules()):
            self.weight_init(m)

    def forward(self, in1:torch.Tensor):
        '''Args:
            inX: Tensor, [N, C, H, W] in the [0., 1.] range
            noise_map: Tensor [N, 1, H, W] in the [0., 1.] range
        '''
        # Input convolution block
        #self.skip1.push(self.non_slice(in1))
        skip1 = self.non_slice(in1)
        x0 = self.inc(in1)
        x0 = x0.half()
        #self.skip2.push(x0)
        skip2 = x0
        # Downsampling
        #print(x0.dtype, x0.device, x0.shape)
        x1 = self.downc0(x0)
        x1 = x1.half()
        #self.skip3.push(x1)
        skip3 = x1
        x2 = self.downc1(x1)
        x2 = x2.half()
        # Upsampling
        x2 = self.upc2(x2)
        x2 = x2.half()
        #x1 = self.upc1(self.none_add(x2, self.skip3.pop(x2)))
        x1 = self.upc1(self.none_add(x2, skip3))
        x1 = x1.half()
        # Estimation
        #x = self.outc(self.none_add(x1, self.skip2.pop(x1)))
        x = self.outc(self.none_add(x1, skip2))
        x = x.half()

        # Residual
        #x = self.none_minus(self.skip1.pop(x), x)
        x = self.none_minus(skip1, x)

        return x
    def non_slice(self, x:torch.Tensor):
        return x[:, 0:3, :, :]
        # if x is None:
        #     return None
        # else:
        #     return x[:, 0:3, :, :]
    def none_add(self, x1:torch.Tensor, x2:torch.Tensor):
        return x1+x2
        # if x1 is None or x2 is None:
        #     return None
        # else: 
        #     return x1+x2
        
    def none_minus(self, x1:torch.Tensor, x2:torch.Tensor):
        x_out = x2
        x_out[:, :3, :, :] = x1[:, :3, :, :] - x_out[:, :3, :, :]
        return x_out
        # if x1 is None or x2 is None:
        #     return None
        # else: 
        #     x_out = x2
        #     x_out[:, :3, :, :] = x1[:, :3, :, :] - x_out[:, :3, :, :]
        #     return x_out
        

class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        self.temp1 = DenBlock()

        # Init weights
        self.reset_params()

    @staticmethod
    def weight_init(m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, nonlinearity='relu')

    def reset_params(self):
        for _, m in enumerate(self.modules()):
            self.weight_init(m)

    def forward(self, x):
        x = self.temp1(x)
        return x


class BSVD(nn.Module):
    """
        Bidirection-buffer based framework with pipeline-style inference
    """
    def __init__(self, chns=[32, 64, 128], mid_ch=3, shift_input=False, in_ch=4, out_ch=3, norm='bn', act='relu', interm_ch=30, blind=False, 
                 pretrain_ckpt='./experiments/pretrained_ckpt/bsvd-64.pth'):
        super(BSVD, self).__init__()
        self.temp1 = DenBlock(chns=chns, out_ch=mid_ch, in_ch=in_ch,  shift_input=shift_input, norm=norm, act=act, blind=blind, interm_ch=interm_ch)
        self.temp2 = DenBlock(chns=chns, out_ch=out_ch, in_ch=mid_ch, shift_input=shift_input, norm=norm, act=act, blind=blind, interm_ch=interm_ch)

        self.shift_num = self.count_shift()
        # Init weights
        self.reset_params()
        if pretrain_ckpt is not None:
            self.load(pretrain_ckpt)
        # self.shift_num = 
        # self.shift_num = 
    # def reset(self):
    #     self.temp1.reset()
    #     self.temp2.reset()
    def load(self, path):
        ckpt = torch.load(path)
        print("load from %s"%path)
        ckpt_state = ckpt['params']
        # split the dict here
        if 'module' in list(ckpt_state.keys())[0]:
            base_name = 'module.base_model.'
        else:
            base_name = 'base_model.'
        ckpt_state_1 = extract_dict(ckpt_state, string_name=base_name+'nets_list.0.')
        ckpt_state_2 = extract_dict(ckpt_state, string_name=base_name+'nets_list.1.')
        self.temp1.load_from(ckpt_state_1)
        self.temp2.load_from(ckpt_state_2)
            
    @staticmethod
    def weight_init(m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, nonlinearity='relu')

    def reset_params(self):
        for _, m in enumerate(self.modules()):
            self.weight_init(m)

    def feedin_one_element(self, x:torch.Tensor):
        x   = self.temp1(x)
        x   = self.temp2(x)
        return x
    
    def forward(self, input):#, noise_map=None):
        # N, F, C, H, W -> (N*F, C, H, W)
        # noise_map = None
        # if noise_map is not None:
        #     input = torch.cat([input, noise_map], dim=2)
        N, F, C, H, W = input.shape
        input = input.reshape(N*F, C, H, W)
        base_out = self.streaming_forward(input)
        NF, C, H, W = base_out.shape
        base_out = base_out.reshape(N, F, C, H, W)
        return base_out
    
    def streaming_forward(self, input_seq):
        """
        pipeline-style inference

        Args:
            Noisy video stream

        Returns:
            Denoised video stream
        """
        out_seq: List[torch.Tensor] = []
        
        n,c,h,w = input_seq.shape
        #input_seq = [input_seq[i:i+1, ...] for i in range(n)]

        #_,c,h,w = input_seq[0].shape
        #with torch.no_grad():
        for i in range(n):
            x = input_seq[i].unsqueeze(0)
            x_cuda = x#.clone()
            #print('hey', x_cuda.shape)
            x_cuda = self.feedin_one_element(x_cuda)
            #print('hey in', x_cuda.shape)
            out_seq.append(x_cuda)
        #print('hey', x_cuda.shape)
        x_cuda = input_seq[-1].unsqueeze(0)

        # end_out = self.feedin_one_element(x_cuda)
        # out_seq.append(end_out)
        # end_out = self.feedin_one_element(0)
        # end stage
        # while 1:
        #     # print("feed in none")
        #     end_out = self.feedin_one_element(x_cuda)
        #     # if end_out is not None: end_out = end_out.cpu()
            
        #     if len(out_seq) == (self.shift_num+len(input_seq)):
        #         break
        #     # if isinstance(end_out, torch.Tensor): end_out = end_out.cpu()\
        #     #print('hey in', end_out.shape)
        #     out_seq.append(end_out)
        #     # max_mem = torch.cuda.max_memory_allocated()/1024/1024/1024
        #     # print("max memory required \t\t %.2fGB"%max_mem)
        #     # print("*****************************************************************************")
        # number of temporal shift is 2, last element is 0
        # TODO fix init and end frames
        out_seq_clip = []
        for item in out_seq[-len(input_seq):]:#out_seq[self.shift_num:]:
            assert item is not None
            out_seq_clip.append(item)
        #self.reset()
        return torch.cat(out_seq_clip, dim=0)

    def count_shift(self):
        count = 0
        for name, module in self.named_modules():
            # print(type(module))
            if "BiBufferConv" in str(type(module)):
                count+=1
        return count


if __name__ == '__main__':
    H, W = 720, 1280
    set_res((H, W))
    model = BSVD(
        chns=[64,128,256], mid_ch=64, shift_input=False, 
        norm='none', interm_ch=64, act='relu6', 
        pretrain_ckpt='./upscale/model/bsvd/bsvd-64.pth'
    )
    model = model
    inp_shape = (1,1,4,H,W)
    inp = torch.zeros(inp_shape)
    inp = inp.to(0)
    model = model.to(0).eval()

    def test_model():
        inp = torch.rand(inp_shape).to(0)
        torch.cuda.synchronize()
        t = time.time()
        inp = inp.half()
        N = 30
        for i in range(N):
            with torch.no_grad(), torch.cuda.amp.autocast():
                output = model(inp)
                #print(output)
                print(output if output is None else output.shape)#, (inp[:,:,:3,:,:]-output).mean() if output is not None else None)
        print((time.time()-t)/N)
    test_model()

    import torch_tensorrt, os
    torch_tensorrt.logging.set_reportable_log_level(torch_tensorrt.logging.Level.Warning)
    
    print('Bsvd.build_model: Compiling...')
    
    def jit(model, inp_shape=inp_shape):
        trt_model = torch_tensorrt.compile(model, 
            inputs = [
                torch_tensorrt.Input(inp_shape, dtype=torch.half),
            ],
            enabled_precisions= { torch_tensorrt.dtype.half },
            require_full_compilation = False,
            min_block_size = 3,
        )
        return trt_model
    
    def jit_bsvd(bsvd_model, image_shape=(720,1280)):
        H, W = image_shape
        model = bsvd_model.half()
        
        model.temp1.inc = jit(model.temp1.inc, inp_shape=(1,4,H,W))
        model.temp1.downc0 = jit(model.temp1.downc0, inp_shape=(1,64,H,W))
        model.temp1.downc1 = jit(model.temp1.downc1, inp_shape=(1,128,H//2,W//2))
        model.temp1.upc2 = jit(model.temp1.upc2, inp_shape=(1,256,H//4,W//4))
        model.temp1.upc1 = jit(model.temp1.upc1, inp_shape=(1,128,H//2,W//2))
        model.temp1.outc = jit(model.temp1.outc, inp_shape=(1,64,H,W))

        model.temp2.inc = jit(model.temp2.inc, inp_shape=(1,64,H,W))
        model.temp2.downc0 = jit(model.temp2.downc0, inp_shape=(1,64,H,W))
        model.temp2.downc1 = jit(model.temp2.downc1, inp_shape=(1,128,H//2,W//2))
        model.temp2.upc2 = jit(model.temp2.upc2, inp_shape=(1,256,H//4,W//4))
        model.temp2.upc1 = jit(model.temp2.upc1, inp_shape=(1,128,H//2,W//2))
        model.temp2.outc = jit(model.temp2.outc, inp_shape=(1,64,H,W))

        return model
    
    # skip1 = self.non_slice(in1)
    # x0 = self.inc(in1)
    # skip2 = x0
    # x1 = self.downc0(x0)
    # skip3 = x1
    # x2 = self.downc1(x1)
    # x2 = self.upc2(x2)
    # x1 = self.upc1(self.none_add(x2, skip3))
    # x = self.outc(self.none_add(x1, skip2))
    # x = self.none_minus(skip1, x)

    # return x
    # model = model.half()
    # #model.temp1.upc2(torch.zeros((1,256,90,160),device=0))
    # model.temp1.inc = jit(model.temp1.inc, inp_shape=(1,4,H,W))
    # model.temp1.downc0 = jit(model.temp1.downc0, inp_shape=(1,64,H,W))
    # model.temp1.downc1 = jit(model.temp1.downc1, inp_shape=(1,128,H//2,W//2))
    # model.temp1.upc2 = jit(model.temp1.upc2, inp_shape=(1,256,H//4,W//4))
    # model.temp1.upc1 = jit(model.temp1.upc1, inp_shape=(1,128,H//2,W//2))
    # model.temp1.outc = jit(model.temp1.outc, inp_shape=(1,64,H,W))
    # #model.temp1 = jit(model.temp1, inp_shape=(1,4,360,640))
    # #model.temp2 = jit(model.temp2, inp_shape=(1,64,360,640))
    # model.temp2.inc = jit(model.temp2.inc, inp_shape=(1,64,H,W))
    # model.temp2.downc0 = jit(model.temp2.downc0, inp_shape=(1,64,H,W))
    # model.temp2.downc1 = jit(model.temp2.downc1, inp_shape=(1,128,H//2,W//2))
    # model.temp2.upc2 = jit(model.temp2.upc2, inp_shape=(1,256,H//4,W//4))
    # model.temp2.upc1 = jit(model.temp2.upc1, inp_shape=(1,128,H//2,W//2))
    # model.temp2.outc = jit(model.temp2.outc, inp_shape=(1,64,H,W))
    #model = jit(model)
    model = jit_bsvd(model, (H, W))

    torch_tensorrt.logging.set_reportable_log_level(torch_tensorrt.logging.Level.Warning)

    def eval_img(imgpath = './samples/images/shana.png', name='bsvd', image_shape=(720,1280)):
        import cv2
        H, W = image_shape
        frame = cv2.imread(imgpath)
        frame = cv2.resize(frame, (W,H))
        img = torch.empty((1,1,4,H,W), device=0)
        img.fill_(0.03)
        img[0,0,:3,:,:] = torch.tensor(frame, device=0, dtype=torch.float32).permute(2,0,1) / 255.0

        with torch.no_grad(), torch.cuda.amp.autocast():
            output = model(img.half())
            img = img[0,0,:3,:,:]
            output = output[0,0,:,:,:]
            C, H, W = img.shape
            img = img.view(C, H*W)
            output = output.view(C, H*W)
            output = ((output - torch.mean(output, keepdim=True, dim=-1)) / torch.std(output, keepdim=True, dim=-1) * torch.std(img, keepdim=True, dim=-1)) + torch.mean(img, keepdim=True, dim=-1)
            output = torch.clamp(output.view(C, H, W) * 255, 0,255)
        print(output, output.shape)
        cv2.imwrite(f'samples/images/{name}.png', output.permute(1,2,0).cpu().numpy().astype(np.uint8))
    
    eval_img('./samples/images/shana.png', 'bsvd_shana', (H, W))
    eval_img('./samples/images/shark1.png', 'bsvd_sh1', (H, W))
    eval_img('./samples/images/shark2.png', 'bsvd_sh2', (H, W))
    eval_img('./samples/images/shark3.png', 'bsvd_sh3', (H, W))

    test_model()


# if __name__ == '__main__':
#     model = BSVD(
#         chns=[64,128,256], mid_ch=64, shift_input=False, 
#         norm='none', interm_ch=64, act='relu6', 
#         pretrain_ckpt='./upscale/model/bsvd/bsvd-64.pth'
#     )
#     model = model
#     inp_shape = (1,10,4,720,1280)
#     inp = torch.zeros(inp_shape)
#     inp = inp.to(0)

#     model = model.to(0).eval()
#     def test_model():
#         inp = torch.rand(inp_shape).to(0)
#         torch.cuda.synchronize()
#         t = time.time()
#         for i in range(3):
#             with torch.no_grad(), torch.cuda.amp.autocast():
#                 output = model(inp)
#                 print(output if output is None else output.shape)#, (inp[:,:,:3,:,:]-output).mean() if output is not None else None)
#         print((time.time()-t)/50)
#     test_model()

#     import torch_tensorrt, os
#     torch_tensorrt.logging.set_reportable_log_level(torch_tensorrt.logging.Level.Info)
    
#     print('Bsvd.build_model: Compiling...')
#     traced_model = torch.jit.script(model, (inp, None, None))
#     trt_model = torch_tensorrt.compile(traced_model, 
#         inputs = [
#             torch_tensorrt.Input(inp.shape),
#             # torch_tensorrt.Input(inp.shape),
#             # torch_tensorrt.Input([model.n, model.fold, model.h, model.w]),
#         ],
#         enabled_precisions= { torch_tensorrt.dtype.half },
#         require_full_compilation = False
#     )
#     model = trt_model

#     torch_tensorrt.logging.set_reportable_log_level(torch_tensorrt.logging.Level.Warning)

#     test_model()