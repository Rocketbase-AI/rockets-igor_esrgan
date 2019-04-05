import os
from .architecture import RRDB_Net
import torch
from torchvision import transforms
import types


def postprocess(self, x: torch.Tensor):
    """Converts pytorch tensor into PIL Image

    Handles all the steps for postprocessing of the raw output of the model.
    Depending on the rocket family there might be additional options.

    Args:
        x (Tensor): Output Tensor to postprocess
    """
    output_transform = transforms.ToPILImage()
    out = []
    x = x.clamp_(0, 1).data.cpu()
    for elem in x:
        out.append(output_transform(elem))

    return out


def preprocess(self, x):
    """Converts PIL Image or Array into pytorch tensor specific to this model

    Handles all the necessary steps for preprocessing such as resizing, normalization.
    Works with both single images and list/batch of images. Input image file is expected
    to be a `PIL.Image` object with 3 color channels.

    Args:
        x (list or PIL.Image): input image or list of images.
    """

    input_transform = transforms.Compose([
        transforms.ToTensor()
    ])

    if type(x) == list:
        out_tensor = None
        for elem in x:
            out = input_transform(elem).unsqueeze(0)
            if out_tensor is not None:
                torch.cat((out_tensor, out), 0)
            else:
                out_tensor = out
    else:
        out_tensor = input_transform(x).unsqueeze(0)

    return out_tensor


def build():
    model = RRDB_Net(3, 3, 64, 23, gc=32, upscale=4, norm_type=None, act_type='leakyrelu',
                     mode='CNA', res_scale=1, upsample_mode='upconv')
    model.load_state_dict(torch.load(os.path.join(os.path.realpath(os.path.dirname(__file__)),
                                                  "weights.pth")),
                          strict=True)

    model.postprocess = types.MethodType(postprocess, model)
    model.preprocess = types.MethodType(preprocess, model)

    return model

