from torch import nn

from .utils import Swish


class ImageFeatureExtractor(nn.Module):
    def __init__(self, imgH=32, nc=3, vocab_size=100):
        super(ImageFeatureExtractor, self).__init__()
        assert imgH % 16 == 0, "imgH has to be a multiple of 16"

        ks = [3, 3, 3, 3, 3, 3, 2]
        ps = [1, 1, 1, 1, 1, 1, 0]
        ss = [1, 1, 1, 1, 1, 1, 1]
        nm = [64, 128, 256, 256, 512, 512, 512]

        cnn = nn.Sequential()

        def convSwish(i, batchNormalization=False):
            nIn = nc if i == 0 else nm[i - 1]
            nOut = nm[i]
            cnn.add_module(
                "conv{0}".format(i), nn.Conv2d(nIn, nOut, ks[i], ss[i], ps[i])
            )
            if batchNormalization:
                cnn.add_module("batchnorm{0}".format(i), nn.BatchNorm2d(nOut))
            cnn.add_module("swish{0}".format(i), Swish())

        """
            MaxPool Kernel per Height:
            32: (2,2) -> (2,2) -> (2,1) -> (2,1)
            64: (4,4) -> (2,2) -> (2,1) -> (2,1)
            128: (4,4) -> (4,4) -> (2,1) -> (2,1)
            256: (4,4) -> (4,4) -> (4,2) -> (2,1)
        """
        maxpoolargs_per_size = {
            16: ((2, 2), (2, (2, 1)), (2, 1), (1, 1)),
            32: ((2, 2), (2, 2), (2, 1), (2, 1)),
            64: ((4, 4), (2, 2), (2, 1), (2, 1)),
            128: ((4, 4), (4, 4), (2, 1), (2, 1)),
            256: ((4, 4), (4, 4), (4, 2), (2, 1)),
        }
        args = maxpoolargs_per_size[imgH]
        convSwish(0)
        cnn.add_module("pooling{0}".format(0), nn.MaxPool2d(*args[0]))  # 64x16x64
        convSwish(1)
        cnn.add_module("pooling{0}".format(1), nn.MaxPool2d(*args[1]))  # 128x8x32
        convSwish(2, False)
        convSwish(3)
        cnn.add_module(
            "pooling{0}".format(2), nn.MaxPool2d(args[2], args[2])
        )  # , (0, 1)))  # 256x4x16
        convSwish(4, True)
        convSwish(5, True)
        cnn.add_module(
            "pooling{0}".format(3), nn.MaxPool2d(args[3], args[3])
        )  # , (0, 1)))  # 512x2x16
        convSwish(6, False)  # 512x1x16

        self.cnn = cnn
        self.lm_head = nn.Linear(512, vocab_size, bias=False)

    def forward(self, input, *args, **kwargs):
        # conv features
        image_features = self.image_features(input)
        return image_features

    def lm(self, image_features):
        return self.lm_head(image_features)

    def image_features(self, input, *args, **kwargs):
        conv = self.cnn(input)
        b, c, h, w = conv.size()
        assert h == 1, f"the height of conv must be 1, shape is {conv.shape}"
        conv = conv.squeeze(2).permute(0, 2, 1)
        return conv
