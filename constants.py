import torch
from torchvision.transforms import v2

IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)
IMAGENET_INCEPTION_MEAN = (0.5, 0.5, 0.5)
IMAGENET_INCEPTION_STD = (0.5, 0.5, 0.5)
IMAGENET_DPN_MEAN = (124 / 255, 117 / 255, 104 / 255)
IMAGENET_DPN_STD = tuple([1 / (.0167 * 255)] * 3)
OPENAI_CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
OPENAI_CLIP_STD = (0.26862954, 0.26130258, 0.27577711)

TRANSFORMS = {
    'imagenet': v2.Compose([
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Resize((224, 224)),
            # v2.ToTensor(),
            v2.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD)
        ]),
    'clip': v2.Compose([
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Resize((224, 224)),
            # v2.ToTensor(),
            v2.Normalize(mean=OPENAI_CLIP_MEAN, std=OPENAI_CLIP_STD)
        ]),
}

