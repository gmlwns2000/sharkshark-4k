import torch
import upscale.model.fsrcnn.model as models

def build_model(factor=4, device=0):
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

    return model