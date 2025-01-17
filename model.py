import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import seed_everything
seed_everything(0)

class WSConv2d(nn.Module):
    # Weighted Scaled Convolutional layer
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, gain=2):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.scale = (gain / (kernel_size * kernel_size * in_channels)) ** 0.5
        self.bias = self.conv.bias
        self.conv.bias = None

        # initialize conv layer
        nn.init.normal_(self.conv.weight)
        nn.init.zeros_(self.bias)

    def forward(self, x):
        return self.conv(x * self.scale) + self.bias.view(1, self.bias.shape[0], 1, 1)

class PixelNorm(nn.Module):
    def __init__(self):
        super().__init__()
        self.epsilon = 1e-8

    def forward(self, x):
        return x / torch.sqrt(torch.mean(x**2, dim=1, keepdim=True) + self.epsilon)

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, use_pixelnorm=True):
        super().__init__()
        self.conv1 = WSConv2d(in_channels, out_channels)
        self.conv2 = WSConv2d(out_channels, out_channels)
        self.leaky = nn.LeakyReLU(0.2)
        self.pn = PixelNorm()
        self.use_pn = use_pixelnorm

    def forward(self, x):
        x = self.leaky(self.conv1(x))
        x = self.pn(x) if self.use_pn else x
        x = self.leaky(self.conv2(x))
        x = self.pn(x) if self.use_pn else x
        return x

class Generator(nn.Module):
    def __init__(
        self,
        z_dim: int,                # Dimension of the latent space of the generator
        embedding_dim: int,        # Dimension of the embedding space of the generator
        class_size: int,           # Number of classes
        img_channels: int,         # Number of channels of the images (1 for grayscale, 3 for RGB)
        in_channels: int,          # Number of channels of the input of the generator
        factors: list              # List of factors used for scaling in the progressive growing structure (e.g. [1, 1, 1/2, 1/4, 1/8, 1/16, 1/32])
    ):
        super().__init__()

        self.label_embedding = nn.Embedding(class_size, embedding_dim)

        self.initial_block = nn.Sequential(
            PixelNorm(),
            nn.ConvTranspose2d(z_dim + embedding_dim, in_channels, 4, 1, 0), # 1x1 -> 4x4
            nn.LeakyReLU(0.2),
            WSConv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2),
            PixelNorm(),
        )
        self.initial_layer = WSConv2d(in_channels, img_channels, kernel_size=1, stride=1, padding=0)

        self.prog_blocks = nn.ModuleList()
        self.last_layer = nn.ModuleList([self.initial_layer])

        for i in range(len(factors)-1):
            conv_in_c = int(in_channels * factors[i])
            conv_out_c = int(in_channels * factors[i+1])
            self.prog_blocks.append(ConvBlock(conv_in_c, conv_out_c))
            self.last_layer.append(WSConv2d(conv_out_c, img_channels, kernel_size=1, stride=1, padding=0))

    def fade_in(self, alpha, upscaled, generated):
        return torch.tanh(alpha * generated + (1 - alpha) * upscaled)

    def forward(self, x, label, alpha, steps): # steps=0 (4x4), steps=1 (8x8), ...
        x = x.float()
        label_embed = self.label_embedding(label)
        label_embed = label_embed.view(label_embed.shape[0], label_embed.shape[1], 1, 1)
        x = torch.cat((x, label_embed), dim=1)  # b_size + (z_dim + embedding_dim) x 1x1
        out = self.initial_block(x)             # b_size x (512) x 4x4

        if steps == 0:
            return torch.tanh(self.initial_layer(out))

        for step in range(steps):
            upscaled = F.interpolate(out, scale_factor=2, mode='nearest')
            out = self.prog_blocks[step](upscaled)

        final_upscaled = self.last_layer[steps-1](upscaled)
        final_out = self.last_layer[steps](out)
        return self.fade_in(alpha, final_upscaled, final_out)

class Discriminator(nn.Module):
    def __init__(
        self,
        img_channels: int,                   # Number of channels of the images (1 for grayscale, 3 for RGB)
        in_channels: int,                    # Number of channels of the input of the generator
        factors: list                        # List of factors used for scaling in the progressive growing structure (e.g. [1, 1, 1/2, 1/4, 1/8, 1/16, 1/32])
    ):
        super().__init__()
        self.prog_blocks = nn.ModuleList()
        self.from_img = nn.ModuleList()
        self.leaky = nn.LeakyReLU(0.2)

        for i in range(len(factors) - 1, 0, -1):
            conv_in_c = int(in_channels * factors[i])
            conv_out_c = int(in_channels * factors[i-1])
            self.prog_blocks.append(ConvBlock(conv_in_c, conv_out_c, use_pixelnorm=False))
            self.from_img.append(WSConv2d(img_channels, conv_in_c, kernel_size=1, stride=1, padding=0))

        # this is for 4x4 img resolution
        self.from_img4 = WSConv2d(img_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.from_img.append(self.from_img4)
        self.avg_pool = nn.AvgPool2d(kernel_size=2, stride=2)

        # block for 4x4 resolution
        self.final_block = nn.Sequential(
            WSConv2d(in_channels+1, in_channels, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2),
            WSConv2d(in_channels, in_channels, kernel_size=4, stride=1, padding=0),
            nn.LeakyReLU(0.2),
        )
        
        self.adv_layer = WSConv2d(in_channels, 1, kernel_size=1, stride=1, padding=0)
        
        aux_channels = 150
        self.aux_FC1 = WSConv2d(in_channels, aux_channels, kernel_size=1, stride=1, padding=0)
        self.aux_layer = WSConv2d(aux_channels, 2, kernel_size=1, stride=1, padding=0)

    def fade_in(self, alpha, downscaled, out):
        return alpha * out + (1-alpha) * downscaled

    def minibatch_std(self, x):
        batch_statistics = torch.std(x, dim=0).mean().repeat(x.shape[0], 1, x.shape[2], x.shape[3])
        return torch.cat([x, batch_statistics], dim=1)

    def forward(self, x, alpha, steps):  # steps=0 (4x4), steps=1 (8x8), ...
        x = x.float()
        cur_step = len(self.prog_blocks) - steps
        out = self.leaky(self.from_img[cur_step](x))

        if steps == 0:
            out = self.final_block(self.minibatch_std(out))
            out1 = self.adv_layer(out)
            out2 = self.aux_layer(self.aux_FC1(out))
            return out1.view(out1.shape[0], -1), out2.view(out2.shape[0], out2.shape[1])

        downscaled = self.leaky(self.from_img[cur_step + 1](self.avg_pool(x)))
        out = self.avg_pool(self.prog_blocks[cur_step](out))
        out = self.fade_in(alpha, downscaled, out)

        for step in range(cur_step + 1, len(self.prog_blocks)):
            out = self.prog_blocks[step](out)
            out = self.avg_pool(out)

        out = self.final_block(self.minibatch_std(out))
        out1 = self.adv_layer(out)
        out2 = self.aux_layer(self.aux_FC1(out))
        return out1.view(out1.shape[0], -1), out2.view(out2.shape[0], out2.shape[1])

# ----------------------------------------------------------------------------------------------

# # Count parameters:

# from pytorch_model_summary import summary

# alpha = 1
# step = 6
# size = 256
# z_dim = 512
# embedding_dim = 3
# class_size = 2
# img_channels = 1
# in_channels = 512
# factors = [1, 1, 1/2, 1/4, 1/8, 1/16, 1/32]
# z = torch.randn(1, z_dim, 1, 1)
# label = torch.tensor([1])
# print(summary(Generator(z_dim, embedding_dim, class_size, img_channels, in_channels, factors), z, label, alpha, step, show_input=True, show_hierarchical=True))
# print(summary(Discriminator(img_channels, in_channels, factors), torch.zeros((1, 1, size, size)), alpha, step, show_input=True, show_hierarchical=True))
