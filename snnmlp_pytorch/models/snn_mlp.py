# 2022.06.27-Changed for building SNN-MLP
#            Huawei Technologies Co., Ltd. <foss@huawei.com>
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath, to_2tuple, trunc_normal_ 
from .snn_cuda import LIFSpike

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1, 1)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1, 1)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class LIFModule(nn.Module):
    def __init__(self, dim, lif_bias=True, proj_drop=0.,
                 lif=-1, lif_fix_tau=False, lif_fix_vth=False, lif_init_tau=0.25, lif_init_vth=-1.):
        super().__init__()
        self.dim = dim
        self.lif = lif
        self.conv1 = nn.Conv2d(dim, dim, 1, 1, 0, groups=1, bias=lif_bias)
        self.conv2_1 = nn.Conv2d(dim, dim, 1, 1, 0, groups=1, bias=lif_bias)
        self.conv2_2 = nn.Conv2d(dim, dim, 1, 1, 0, groups=1, bias=lif_bias)
        self.conv3 = nn.Conv2d(dim, dim, 1, 1, 0, groups=1, bias=lif_bias)

        self.actn = nn.GELU()

        self.norm1 = MyNorm(dim)
        self.norm2 = MyNorm(dim)
        self.norm3 = MyNorm(dim)

        self.lif1 = LIFSpike(lif=lif, fix_tau=lif_fix_tau, fix_vth=lif_fix_vth,
                               init_tau=lif_init_tau, init_vth=lif_init_vth, dim=2)
        self.lif2 = LIFSpike(lif=lif, fix_tau=lif_fix_tau, fix_vth=lif_fix_vth,
                               init_tau=lif_init_tau, init_vth=lif_init_vth, dim=3)
        self.dw1 = nn.Conv2d(dim, dim, 3, 1, 1, groups=dim, bias=lif_bias)
        

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.actn(x)
        x = self.dw1(x)
        x = self.norm2(x)
        x = self.actn(x)
        
        x_lr = self.lif1(x)
        x_td = self.lif2(x)
        x_lr = self.conv2_1(x_lr)
        x_td = self.conv2_2(x_td)
        x_lr = self.actn(x_lr)
        x_td = self.actn(x_td)

        x = x_lr + x_td
        x = self.norm3(x)
        x = self.conv3(x)

        return x

    def extra_repr(self) -> str:
        return f'dim={self.dim}, lif={self.lif}'

    def flops(self, N):
        # calculate flops for 1 window with token length of N
        flops = 0
        # conv1 
        flops += N * self.dim * self.dim
        # norm 1
        flops += N * self.dim
        # dw 1
        flops += N * self.dim * 3 * 3
        # norm 2
        flops += N * self.dim
        # conv2_1 conv2_2
        flops += N * self.dim * self.dim * 2
        # x_lr + x_td
        flops += N * self.dim
        # norm2
        flops += N * self.dim
        # norm3
        flops += N * self.dim * self.dim
        return flops


class LIFBlock(nn.Module):
    def __init__(self, dim, input_resolution,
                 mlp_ratio=4., lif_bias=True, drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 lif=-1, lif_fix_tau=False, lif_fix_vth=False, lif_init_tau=0.25, lif_init_vth=-1.):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.mlp_ratio = mlp_ratio
        self.lif = lif

        self.norm1 = norm_layer(dim)
        self.lif_module = LIFModule(dim, lif_bias=lif_bias, proj_drop=drop,
                               lif=lif, lif_fix_tau=lif_fix_tau, lif_fix_vth=lif_fix_vth,
                               lif_init_tau=lif_init_tau, lif_init_vth=lif_init_vth)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        shortcut = x
        x = self.norm1(x)

        # lif block
        x = self.lif_module(x)

        # FFN
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, " \
               f"lif={self.lif}, mlp_ratio={self.mlp_ratio}"

    def flops(self):
        flops = 0
        H, W = self.input_resolution
        # norm1
        flops += self.dim * H * W
        # lif module
        flops += self.lif_module.flops(H * W)
        # mlp
        flops += 2 * H * W * self.dim * self.dim * self.mlp_ratio
        # norm2
        flops += self.dim * H * W
        return flops


class PatchMerging(nn.Module):
    def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.reduction = nn.Conv2d(4 * dim, 2 * dim, 1, 1, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x):
        x0 = x[:, :, 0::2, 0::2]
        x1 = x[:, :, 1::2, 0::2]
        x2 = x[:, :, 0::2, 1::2]
        x3 = x[:, :, 1::2, 1::2]
        x = torch.cat([x0, x1, x2, x3], 1)

        x = self.norm(x)
        x = self.reduction(x)

        return x

    def extra_repr(self) -> str:
        return f"input_resolution={self.input_resolution}, dim={self.dim}"

    def flops(self):
        H, W = self.input_resolution
        flops = H * W * self.dim
        flops += (H // 2) * (W // 2) * 8 * self.dim * self.dim
        return flops


class BasicLayer(nn.Module):
    def __init__(self, dim, input_resolution, depth,
                 mlp_ratio=4., lif_bias=True, drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False,
                 lif=-1, lif_fix_tau=False, lif_fix_vth=False, lif_init_tau=0.25, lif_init_vth=-1.):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        # build blocks
        self.blocks = nn.ModuleList([
            LIFBlock(dim=dim, input_resolution=input_resolution,
                              mlp_ratio=mlp_ratio,
                              lif_bias=lif_bias,
                              drop=drop, 
                              drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                              norm_layer=norm_layer,
                              lif=lif, lif_fix_tau=lif_fix_tau, lif_fix_vth=lif_fix_vth,
                              lif_init_tau=lif_init_tau, lif_init_vth=lif_init_vth)
            for i in range(depth)])

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(input_resolution, dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x):
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)
        if self.downsample is not None:
            x = self.downsample(x)
        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, depth={self.depth}"

    def flops(self):
        flops = 0
        for blk in self.blocks:
            flops += blk.flops()
        if self.downsample is not None:
            flops += self.downsample.flops()
        return flops


class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None


    def forward(self, x):
        x = self.proj(x)
        if self.norm is not None:
            x = self.norm(x)
        return x

    def flops(self):
        Ho, Wo = self.patches_resolution
        flops = Ho * Wo * self.embed_dim * self.in_chans * (self.patch_size[0] * self.patch_size[1])
        if self.norm is not None:
            flops += Ho * Wo * self.embed_dim
        return flops


def MyNorm(dim):
    return nn.GroupNorm(1, dim)


class SNNMLP(nn.Module):
    def __init__(self, img_size=224, patch_size=4, in_chans=3, num_classes=1000,
                 embed_dim=96, depths=[2, 2, 6, 2], 
                 mlp_ratio=4., lif_bias=True, 
                 drop_rate=0., drop_path_rate=0.1,
                 norm_layer=MyNorm, patch_norm=True,
                 use_checkpoint=False, lif=-1, lif_fix_tau=False, lif_fix_vth=False,
                 lif_init_tau=0.25, lif_init_vth=-1., **kwargs):
        super().__init__()

        self.num_classes = num_classes
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.embed_dim_list = []
        for i in range(self.num_layers):
            self.embed_dim_list.append(int(embed_dim * 2 ** i))
        self.patch_norm = patch_norm
        self.num_features = self.embed_dim_list[-1]
        self.mlp_ratio = mlp_ratio

        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=self.embed_dim_list[0],
            norm_layer=norm_layer if self.patch_norm else None)
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution

        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        # build layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(dim=self.embed_dim_list[i_layer],
                               input_resolution=(patches_resolution[0] // (2 ** i_layer),
                                                 patches_resolution[1] // (2 ** i_layer)),
                               depth=depths[i_layer],
                               mlp_ratio=self.mlp_ratio,
                               lif_bias=lif_bias,
                               drop=drop_rate,
                               drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                               norm_layer=norm_layer,
                               downsample=PatchMerging if (i_layer < self.num_layers - 1) else None,
                               use_checkpoint=use_checkpoint,
                               lif=lif, lif_fix_tau=lif_fix_tau, lif_fix_vth=lif_fix_vth,
                               lif_init_tau=lif_init_tau, lif_init_vth=lif_init_vth)
            self.layers.append(layer)

        self.norm = norm_layer(self.num_features)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_features(self, x):
        x = self.patch_embed(x)
        x = self.pos_drop(x)

        for layer in self.layers:
            x = layer(x)

        x = self.norm(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x

    def flops(self):
        flops = 0
        flops += self.patch_embed.flops()
        for i, layer in enumerate(self.layers):
            flops += layer.flops()
        flops += self.num_features * self.patches_resolution[0] * self.patches_resolution[1] // (2 ** self.num_layers)
        flops += self.num_features * self.num_classes
        return flops
