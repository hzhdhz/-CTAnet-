import torch
import torch.nn as nn
import math

from einops import rearrange
from einops.layers.torch import Rearrange

import torch.nn.functional as F
import scipy.io as sio
def conv_3x3_bn(inp, oup, image_size, downsample=False):
    stride = 1 if downsample == False else 2
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        h_swish()
    )


# Channel Attention (CA) Layer
class CALayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
            nn.ReLU(inplace=False),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y

class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6

class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)
# Residual Channel Attention Block (RCAB)nn.GELU()
class RCAB(nn.Module):
    def __init__(self, n_feat, kernel_size, reduction, bias=True, bn=False, act=h_swish(), res_scale=1):

        super(RCAB, self).__init__()
        modules_body = []
        for i in range(2):
            modules_body.append(
                nn.Conv2d(n_feat, n_feat, kernel_size=kernel_size, stride=1, padding=kernel_size // 2, bias=bias))
            modules_body.append(nn.BatchNorm2d(n_feat))
            modules_body.append(act)
            # if bn:
            #     modules_body.append(nn.BatchNorm2d(n_feat))
            # if i == 0:
            #     modules_body.append(act)
        # modules_body.append(CALayer(n_feat, reduction))
        self.body = nn.Sequential(*modules_body)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x).mul(self.res_scale)
        res = res + x
        return res



# Residual Group (RG)
#class ResidualGroup(nn.Module):
class block_hz(nn.Module):
    def __init__(self, inp, oup, image_size=1, downsample=False, kernel_size=3,
                 reduction=16, res_scale=1, n_resblocks=2, act=nn.ReLU(False)):
        super(block_hz, self).__init__()
        n_feat = inp
        modules_body = [
            RCAB(n_feat, kernel_size, reduction, bias=True, bn=False, act=act, res_scale=res_scale)
            for _ in range(n_resblocks)]
        modules_body.append(nn.Conv2d(n_feat, n_feat, kernel_size=kernel_size, stride=1, padding=kernel_size // 2))
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        res = self.body(x)
        res = res + x
        return res


def convT_3x3_bn(inp, oup, image_size, downsample=False):
    #stride = 1 if downsample == False else 2
    return nn.Sequential(
        nn.ConvTranspose2d(inp, oup, kernel_size=3, bias=False),
        nn.BatchNorm2d(oup),
        #nn.GELU()
        h_swish()
    )


class PreNorm(nn.Module):
    def __init__(self, dim, fn, norm):
        #PreNorm(inp, self.attn, nn.LayerNorm),
        # self.conv = PreNorm(inp, self.conv, nn.BatchNorm2d)
        # PreNorm(inp, self.attn, nn.LayerNorm),
        super().__init__()
        self.norm = norm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class SE(nn.Module):
    def __init__(self, inp, oup, expansion=0.25):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(oup, int(inp * expansion), bias=False),
            nn.GELU(),
            nn.Linear(int(inp * expansion), oup, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class MBConv(nn.Module):
    def __init__(self, inp, oup, image_size, downsample=False, expansion=4):
        super().__init__()
        self.downsample = downsample
        stride = 1 if self.downsample == False else 2
        hidden_dim = int(inp * expansion)

        if self.downsample:
            self.pool = nn.MaxPool2d(3, 2, 1)
            self.proj = nn.Conv2d(inp, oup, 1, 1, 0, bias=False)

        if expansion == 1:
            self.conv = nn.Sequential(
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride,
                          1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.GELU(),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )
        else:
            self.conv = nn.Sequential(
                # pw
                # down-sample in the first conv
                nn.Conv2d(inp, hidden_dim, 1, stride, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.GELU(),
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, 1, 1,
                          groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.GELU(),
                SE(inp, hidden_dim),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )

        self.conv = PreNorm(inp, self.conv, nn.BatchNorm2d)

    def forward(self, x):
        if self.downsample:
            return self.proj(self.pool(x)) + self.conv(x)
        else:
            # print('x + self.conv(x)==========================', x.shape, self.conv(x).shape)
            # x + self.conv(x)========================== torch.Size([1, 64, 28, 28]) torch.Size([1, 96, 28, 28])
            return x + self.conv(x)

class Attention(nn.Module):
    def __init__(self, inp, oup, image_size, heads=8, dim_head=32, dropout=0.):
        # self.attn = Attention(inp, oup, image_size, heads, dim_head, dropout)
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == inp)

        self.ih, self.iw = image_size

        self.heads = heads
        self.scale = dim_head ** -0.5

        # parameter table of relative position bias
        self.relative_bias_table = nn.Parameter(
            torch.zeros((2 * self.ih - 1) * (2 * self.iw - 1), heads))

        coords = torch.meshgrid((torch.arange(self.ih), torch.arange(self.iw)))
        coords = torch.flatten(torch.stack(coords), 1)
        relative_coords = coords[:, :, None] - coords[:, None, :]

        relative_coords[0] += self.ih - 1
        relative_coords[1] += self.iw - 1
        relative_coords[0] *= 2 * self.iw - 1
        relative_coords = rearrange(relative_coords, 'c h w -> h w c')
        relative_index = relative_coords.sum(-1).flatten().unsqueeze(1)
        self.register_buffer("relative_index", relative_index)

        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(inp, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, oup),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):

        qkv = self.to_qkv(x).chunk(3, dim=-1)


        q, k, v = map(lambda t: rearrange(
            t, 'b n (h d) -> b h n d', h=self.heads), qkv)


        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale


        # Use "gather" for more efficiency on GPUs
        relative_bias = self.relative_bias_table.gather(
            0, self.relative_index.repeat(1, self.heads))
        relative_bias = rearrange(
            relative_bias, '(h w) c -> 1 c h w', h=self.ih * self.iw, w=self.ih * self.iw)
        dots = dots + relative_bias


        attn = self.attend(dots)
        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        return out


class Transformer(nn.Module):
    def __init__(self, inp, oup, image_size, heads=8, dim_head=32, downsample=False, dropout=0.):
        super().__init__()
        hidden_dim = int(inp * 4)

        self.ih, self.iw = image_size
        self.downsample = downsample

        if self.downsample:
            self.pool1 = nn.MaxPool2d(3, 2, 1)
            self.pool2 = nn.MaxPool2d(3, 2, 1)
            self.proj = nn.Conv2d(inp, oup, 1, 1, 0, bias=False)

        self.attn = Attention(inp, oup, image_size, heads, dim_head, dropout)
        self.ff = FeedForward(oup, hidden_dim, dropout)

        self.attn = nn.Sequential(
            Rearrange('b c ih iw -> b (ih iw) c'),
            PreNorm(inp, self.attn, nn.LayerNorm),
            Rearrange('b (ih iw) c -> b c ih iw', ih=self.ih, iw=self.iw)
        )

        self.ff = nn.Sequential(
            Rearrange('b c ih iw -> b (ih iw) c'),
            PreNorm(oup, self.ff, nn.LayerNorm),
            Rearrange('b (ih iw) c -> b c ih iw', ih=self.ih, iw=self.iw)
        )

    def forward(self, x):
        # print('x------transf', x.shape)
        if self.downsample:
            x = self.proj(self.pool1(x)) + self.attn(self.pool2(x))
        else:
            x = x + self.attn(x)
        x = x + self.ff(x)
        return x



class CTANet(nn.Module):
    def __init__(self, image_size, in_channels, num_blocks, channels, num_classes=900,
                 block_types=['C', 'C', 'T', 'T']):
        super().__init__()
        ih, iw = image_size
        block = {'C': MBConv, 'T': Transformer}
        block_types_decode = ['T', 'T', 'C', 'C']
        channels_d = [channels[4], channels[3], channels[2], channels[1], channels[0]]
        self.s0 = self._make_layer(
            conv_3x3_bn, in_channels, channels[0], num_blocks[0], (ih // 1, iw // 1))
        self.s1 = self._make_layer(
            block_hz, channels[0], channels[1], num_blocks[1], (ih // 4, iw // 4))
        self.s2 = self._make_layer(
            block_hz, channels[1], channels[2], num_blocks[2], (ih // 8, iw // 8))
        self.s3 = self._make_layer(
            block[block_types[2]], channels[2], channels[3], num_blocks[3], (ih // 1, iw // 1))
        self.s4 = self._make_layer(
            block[block_types[3]], channels[3], channels[4], num_blocks[4], (ih // 1, iw // 1))

        cc = num_classes // 9
        self.ds2 = self._make_layer(
            convT_3x3_bn,                 cc, channels_d[3], 1, (ih // 1, iw // 1))
        self.ds2_add = self._make_layer(
            block_hz,                 channels_d[3], channels_d[3], num_blocks[2]-1, (ih // 1, iw // 1))

        self.ds3 = self._make_layer(
            convT_3x3_bn,                 channels_d[3], channels_d[4], 1, (ih // 1, iw // 1))
        self.ds3_add = self._make_layer(
            block_hz,                 channels_d[4], channels_d[4], num_blocks[1]-1, (ih // 1, iw // 1))
        ############################################################################################airport
        self.ds4 = self._make_layer(
            convT_3x3_bn,                 channels_d[4], 205, 1, (ih // 1, iw // 1))
        self.ds4_add = self._make_layer(
            conv_3x3_bn,                 205, 205, num_blocks[0]-1, (ih // 1, iw // 1))
        ############################################################################################airport
        ############################################################################################sandiego
        # self.ds4 = self._make_layer(
        #     convT_3x3_bn, channels_d[4], 189, 1, (ih // 1, iw // 1))
        # self.ds4_add = self._make_layer(
        #     conv_3x3_bn, 189, 189, num_blocks[0] - 1, (ih // 1, iw // 1))
        ############################################################################################sandiego
        ############################################################################################SZ
        # self.ds4 = self._make_layer(
        #     convT_3x3_bn, channels_d[4], 32, 1, (ih // 1, iw // 1))
        # self.ds4_add = self._make_layer(
        #     conv_3x3_bn, 32, 32, num_blocks[0] - 1, (ih // 1, iw // 1))
        ############################################################################################SZ

        # self.pool = nn.AvgPool2d(ih // 32, 1)
        self.pool = nn.AvgPool2d(ih, 1)
        self.fc = nn.Linear(channels[-1], num_classes, bias=False)

    def forward(self, x):
        x = self.s0(x)

        x = self.s1(x)
        x = self.s2(x)
        x = self.s3(x)
        x = self.s4(x)
        #############################################%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        x_feature = x.detach().cpu().numpy()
        ###############################################################%%%%%%%%%%%%%%%%%%
        x = self.pool(x).view(-1, x.shape[1])
        x = self.fc(x)
        x_encoder = x
        x = x.view(int(x.size(0)), -1, 3, 3)
        #############################################%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        x = self.ds2(x)
        x = self.ds2_add(x)
        x = self.ds3(x)
        x = self.ds3_add(x)
        x = self.ds4(x)
        x = self.ds4_add(x)
        x = torch.sigmoid(x)
        return x_encoder, x

    def _make_layer(self, block, inp, oup, depth, image_size):
        layers = nn.ModuleList([])
        for i in range(depth):
            if i == 0:
                layers.append(block(inp, oup, image_size))  # , downsample=True))
            else:
                layers.append(block(oup, oup, image_size))
        return nn.Sequential(*layers)


def ctanet_hz():
    num_blocks = [2, 2, 3, 5, 2]  # L
    channels = [64, 64, 64, 64, 64]  # D
    return CTANet((9,9), 205, num_blocks, channels, num_classes=999)#ariport
    # return CTANet((9,9), 189, num_blocks, channels, num_classes=999)#Sandiego
    # return CTANet((9,9), 32, num_blocks, channels, num_classes=999)#SZ


# def count_parameters(model):
#     return sum(p.numel() for p in model.parameters() if p.requires_grad)
