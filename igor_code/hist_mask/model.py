import torch
from torch import nn
from torch.nn import functional as F
from torchvision import models
import math


class UNet(nn.Module):
    """
    TODO: 8 points

    A standard UNet network (with padding in covs).

    For reference, see the scheme in materials/unet.png
    - Use batch norm between conv and relu
    - Use max pooling for downsampling
    - Use conv transpose with kernel size = 3, stride = 2, padding = 1, and output padding = 1 for upsampling
    - Use 0.5 dropout after concat

    Args:
      - num_classes: number of output classes
      - min_channels: minimum number of channels in conv layers
      - max_channels: number of channels in the bottleneck block
      - num_down_blocks: number of blocks which end with downsampling

    The full architecture includes downsampling blocks, a bottleneck block and upsampling blocks

    You also need to account for inputs which size does not divide 2**num_down_blocks:
    interpolate them before feeding into the blocks to the nearest size which divides 2**num_down_blocks,
    and interpolate output logits back to the original shape
    """
    def __init__(self, 
                 num_classes,
                 min_channels=32,
                 max_channels=512, 
                 num_down_blocks=4):
        super(UNet, self).__init__()
        self.num_classes = num_classes
        # TODO

        self.down_blocks = []
        self.pool_layers = []
        for i in range(num_down_blocks):
            if i == 0:
                channels_in = 3
                channels_1 = min_channels
            else:
                channels_in = min_channels * 2**i
                channels_1 = min_channels * 2**(i+1)
            channels_out = min_channels * 2**(i+1)
            down_block = nn.Sequential(
                nn.Conv2d(channels_in, channels_1, 3, padding=1, bias=False),
                nn.BatchNorm2d(channels_1),
                nn.ReLU(inplace=True),
                nn.Conv2d(channels_1, channels_out, 3, padding=1, bias=False),
                nn.BatchNorm2d(channels_out),
                nn.ReLU(inplace=True),
            )
            self.down_blocks.append(down_block)
            self.pool_layers.append(nn.MaxPool2d(2, 2))

        self.bottleneck = nn.Sequential(
            nn.Conv2d(max_channels, max_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(max_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(max_channels, max_channels, 3, padding=1),
            nn.BatchNorm2d(max_channels),
            nn.ReLU(inplace=True),
        )

        self.up_blocks = []
        self.upsample_layers = []
        for i in range(num_down_blocks):
            channels_in = max_channels // 2**i
            if i == num_down_blocks - 1:
                channels_1 = min_channels * 2
            else:
                channels_1 = max_channels // 2**(i+1)
            channels_out = max_channels // 2**(i+1)
            self.upsample_layers.append(
                nn.ConvTranspose2d(channels_in, channels_in, 3, 
                    stride=2, padding=1, output_padding=1))
            up_block = nn.Sequential(
                nn.Dropout2d(p=0.5),
                nn.Conv2d(channels_in*2, channels_1, 3, padding=1, bias=False),
                nn.BatchNorm2d(channels_1),
                nn.ReLU(inplace=True),
                nn.Conv2d(channels_1, channels_out, 3, padding=1, bias=False),
                nn.BatchNorm2d(channels_out),
                nn.ReLU(inplace=True),
            )
            self.up_blocks.append(up_block)
                
        self.out = nn.Sequential(
            nn.Conv2d(min_channels, num_classes, 1),
            nn.Sigmoid(),
        )

        self.down_blocks = nn.ModuleList(self.down_blocks)
        self.pool_layers = nn.ModuleList(self.pool_layers)
        self.up_blocks = nn.ModuleList(self.up_blocks)
        self.upsample_layers = nn.ModuleList(self.upsample_layers)

    

    def forward(self, inputs):
        # logits = None # TODO

        input_shape = (inputs.shape[2], inputs.shape[3])
        input_shape_corrected = (
            int(math.ceil(input_shape[0]/2**len(self.down_blocks)))*2**len(self.down_blocks),
            int(math.ceil(input_shape[1]/2**len(self.down_blocks)))*2**len(self.down_blocks),
        )
        if input_shape[0] != input_shape_corrected[0] or input_shape[1] != input_shape_corrected[1]:
            x = torch.nn.functional.interpolate(inputs, size=input_shape_corrected, mode="bilinear")
            interpolated = True
        else:
            x = inputs
            interpolated = False

        downs = []
        for i in range(len(self.down_blocks)):
            x = self.down_blocks[i](x)
            downs.append(x)
            x = self.pool_layers[i](x)

        for i in range(len(self.up_blocks)):
            x = self.upsample_layers[i](x)
            x = torch.cat([downs[len(downs)-i-1], x], dim=1)
            x = self.up_blocks[i](x)
        
        logits = self.out(x)

        if interpolated:
            logits = torch.nn.functional.interpolate(logits, size=input_shape, mode="bilinear")

        assert logits.shape == (inputs.shape[0], self.num_classes, inputs.shape[2], inputs.shape[3]), 'Wrong shape of the logits'
        return logits


class DeepLab(nn.Module):
    """
    TODO: 6 points

    (simplified) DeepLab segmentation network.
    
    Args:
      - backbone: ['resnet18', 'vgg11_bn', 'mobilenet_v3_small'],
      - aspp: use aspp module
      - num classes: num output classes

    During forward pass:
      - Pass inputs through the backbone to obtain features
      - Apply ASPP (if needed)
      - Apply head
      - Upsample logits back to the shape of the inputs
    """
    def __init__(self, backbone, aspp, num_classes):
        super(DeepLab, self).__init__()
        self.backbone = backbone
        self.init_backbone()
        self.num_classes = num_classes

        if aspp:
            self.aspp = ASPP(self.out_features, 256, [12, 24, 36])
        else:
            self.aspp = None

        self.head = DeepLabHead(self.out_features, num_classes)

    def init_backbone(self):
        # TODO: initialize an ImageNet-pretrained backbone
        if self.backbone == 'resnet18':
            # self.out_features = None # TODO: number of output features in the backbone
            model = models.resnet18(pretrained=True, progress=False)
            self.model_backbone = nn.Sequential(
                model.conv1,
                model.bn1,
                model.relu,
                model.maxpool,
                model.layer1,
                model.layer2,
                model.layer3,
                model.layer4,
            )
            self.out_features = 512

        elif self.backbone == 'vgg11_bn':
            # pass
            # self.out_features = None # TODO
            self.model_backbone = models.vgg11_bn(pretrained=True, progress=False).features
            self.out_features = 512

        elif self.backbone == 'mobilenet_v3_small':
            # pass
            # self.out_features = None # TODO
            self.model_backbone = models.mobilenet_v3_small(pretrained=True, progress=False).features
            self.out_features = 576

    def _forward(self, x):
        # TODO: forward pass through the backbone
        if self.backbone == 'resnet18':
            x = self.model_backbone(x)

        elif self.backbone == 'vgg11_bn':
            x = self.model_backbone(x)

        elif self.backbone == 'mobilenet_v3_small':
            x = self.model_backbone(x)

        return x

    def forward(self, inputs):
        # pass # TODO
        x = self._forward(inputs)
        if self.aspp is not None:
            x = self.aspp(x)
        x = self.head(x)
        logits = F.upsample_bilinear(x, size=(inputs.shape[2], inputs.shape[3]))

        assert logits.shape == (inputs.shape[0], self.num_classes, inputs.shape[2], inputs.shape[3]), 'Wrong shape of the logits'
        return logits


class DeepLabHead(nn.Sequential):
    def __init__(self, in_channels, num_classes):
        super(DeepLabHead, self).__init__(
            nn.Conv2d(in_channels, in_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels, num_classes, 1)
        )


class ASPP(nn.Module):
    """
    TODO: 8 points

    Atrous Spatial Pyramid Pooling module
    with given atrous_rates and out_channels for each head
    Description: https://paperswithcode.com/method/aspp
    
    Detailed scheme: materials/deeplabv3.png
      - "Rates" are defined by atrous_rates
      - "Conv" denotes a Conv-BN-ReLU block
      - "Image pooling" denotes a global average pooling, followed by a 1x1 "conv" block and bilinear upsampling
      - The last layer of ASPP block should be Dropout with p = 0.5

    Args:
      - in_channels: number of input and output channels
      - num_channels: number of output channels in each intermediate "conv" block
      - atrous_rates: a list with dilation values
    """
    def __init__(self, in_channels, num_channels, atrous_rates):
        super(ASPP, self).__init__()
        self.conv_blocks = []
        self.conv_blocks.append(nn.Sequential(
            nn.Conv2d(in_channels, num_channels, 1, bias=False),
            nn.BatchNorm2d(num_channels),
            nn.ReLU(inplace=True)
        ))
        for rate in atrous_rates:
            self.conv_blocks.append(nn.Sequential(
                nn.Conv2d(in_channels, num_channels, 3, padding=rate, dilation=rate, bias=False),
                nn.BatchNorm2d(num_channels),
                nn.ReLU(inplace=True),
            ))
        self.image_pooling = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, num_channels, 1, bias=False),
            nn.BatchNorm2d(num_channels),
            nn.ReLU(inplace=True),
        )
        self.last = nn.Sequential(
            nn.Conv2d(num_channels*(len(atrous_rates)+2), in_channels, 1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.5),
        )
        self.conv_blocks = nn.ModuleList(self.conv_blocks)


    def forward(self, x):
        # TODO: forward pass through the ASPP module

        outs = []
        for layer in self.conv_blocks:
            outs.append(layer(x))
        pooled = self.image_pooling(x)
        pooled = F.upsample_bilinear(pooled, size=(x.shape[2], x.shape[3]))
        outs.append(pooled)
        out = torch.cat(outs, dim=1)
        res = self.last(out)
        
        assert res.shape[1] == x.shape[1], 'Wrong number of output channels'
        assert res.shape[2] == x.shape[2] and res.shape[3] == x.shape[3], 'Wrong spatial size'
        return res