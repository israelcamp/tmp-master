from typing import List
from PIL import Image


import torchvision as tv
import torch


class MaxPoolImagePad():

    def __init__(self):
        self.pool = torch.nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
    
    def __call__(self, x):
        return self.pool(self.pool(x))

def expand_image(img, h, w):
    expanded = Image.new('RGB', (w, h), color=3*(255,))  # white
    expanded.paste(img)
    return expanded

def get_image(
    image_path: str,
    height: int,
    min_width: int = 40
) -> Image:
    image = Image.open(image_path).convert('RGB')
    
    w, h = image.size
    ratio = height / h  # how the height will change
    nw = round(w * ratio)

    image = image.resize((nw, height))

    if nw < min_width:
        image = expand_image(image, height, 40)
    
    return image



def get_inputs_from_image(
    image: Image,
    pooler: MaxPoolImagePad,
):

    max_width = image.width
    attention_image = torch.tensor(
        [1] * max_width
    ).unsqueeze(0)
    attention = pooler(attention_image.float()).long() if pooler is not None else None

    tensor_image = tv.transforms.ToTensor()(image).unsqueeze(0)

    return tensor_image, attention
    

def image_processor(
    image_path: str
):
    image = get_image(image_path, height=32)
    return get_inputs_from_image(image, MaxPoolImagePad())