import torch
from torch import nn
import torch.nn.functional as F
import math
from libs.unet import *


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class EuclideanPositionEncoding(nn.Module):
    def __init__(self, dmodel,
                 coords_dim=2,
                 trainable=False,
                 debug=False):
        super(EuclideanPositionEncoding, self).__init__()
        """
        channel expansion for input
        """
        self.pos_proj = nn.Conv2d(coords_dim, dmodel, kernel_size=1)
        self.trainable = trainable
        self.debug = debug

    def _get_position(self, x):
        bsz, _, h, w = x.size()  # x is bsz, channel, h, w
        grid_x, grid_y = torch.linspace(0, 1, h), torch.linspace(0, 1, w)
        mesh = torch.stack(torch.meshgrid(
            [grid_x, grid_y], indexing='ij'), dim=0)
        return mesh

    def forward(self, x):
        pos = self._get_position(x).to(x.device)
        pos = pos.unsqueeze(0).repeat(x.size(0), 1, 1, 1)
        pos = self.pos_proj(pos)

        x = x + pos
        return x

class PositionalEncoding2D(nn.Module):
    def __init__(self, channels, 
                       channel_last=False, 
                       trainable=False, 
                       pos_cache=None,
                       debug=False):
        """
        modified from https://github.com/tatp22/multidim-positional-encoding
        :param channels: The last dimension of the tensor you want to apply pos emb to.
        """
        super(PositionalEncoding2D, self).__init__()
        channels = int(math.ceil(channels / 2))
        self.channels = channels
        inv_freq = 1. / (10000
                         ** (torch.arange(0, channels, 2).float() / channels))
        self.register_buffer('inv_freq', inv_freq)
        self.channel_last = channel_last
        self.trainable = trainable
        self.debug = debug
        self.pos_cache = pos_cache

    def forward(self, x):
        """
        :param x: A 4d tensor of size (batch_size, C, h, w)
        :return: Positional Encoding Matrix of size (batch_size, C, x, y)
        """
        if self.channel_last:
            x = x.permute(0, 3, 1, 2)

        if self.pos_cache is not None and self.pos_cache.shape == x.shape:
            return self.pos_cache + x

        bsz, n_channel, h, w = x.shape
        pos_x = torch.arange(h, device=x.device, dtype=self.inv_freq.dtype)
        pos_y = torch.arange(w, device=x.device, dtype=self.inv_freq.dtype)
        sin_inp_x = torch.einsum("i,j->ij", self.inv_freq, pos_x)
        sin_inp_y = torch.einsum("i,j->ij", self.inv_freq, pos_y)

        emb_x = torch.cat((sin_inp_x.sin(), sin_inp_x.cos()),
                          dim=0).unsqueeze(-1)
        emb_y = torch.cat((sin_inp_y.sin(), sin_inp_y.cos()),
                          dim=0).unsqueeze(-2)

        emb = torch.zeros((self.channels * 2, h, w),
                          device=x.device, dtype=x.dtype)
        emb[:self.channels, ...] = emb_x
        emb[self.channels:2 * self.channels, ...] = emb_y

        emb = emb[:n_channel, ...].unsqueeze(0).repeat(bsz, 1, 1, 1)

        if self.channel_last:
            emb = emb.permute(0, 2, 3, 1)

        self.pos_cache = emb
        return self.pos_cache + x


class Attention(nn.Module):
    def __init__(self, dim,
                 heads=4,
                 dim_head=None,
                 attn_drop=0.,
                 proj_drop=0.,
                 bias=False,
                 norm_type='layer',
                 skip_connection=True,
                 return_attention=False,
                 softmax=True,
                 sinosoidal_pe=False,
                 pe_trainable=False,
                 debug=False):
        super(Attention, self).__init__()

        self.heads = heads
        self.dim_head = dim // heads * 2 if dim_head is None else dim_head
        self.inner_dim = self.dim_head * heads  # like dim_feedforward
        self.attn_factor = self.dim_head ** (-0.5)
        self.bias = bias
        self.softmax = softmax
        self.skip_connection = skip_connection
        self.return_attention = return_attention
        self.debug = debug

        self.to_qkv = depthwise_separable_conv(
            dim, self.inner_dim*3, bias=self.bias)
        self.to_out = depthwise_separable_conv(
            self.inner_dim, dim, bias=self.bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)

        PE = PositionalEncoding2D if sinosoidal_pe else EuclideanPositionEncoding
        self.pe = PE(dim, trainable=pe_trainable)
        self.norm_type = norm_type
        self.norm_q = self._get_norm(self.dim_head, self.heads,
                                     eps=1e-6)

        self.norm_k = self._get_norm(self.dim_head, self.heads,
                                     eps=1e-6)

        self.norm_out = self._get_norm(self.dim_head, self.heads,
                                       eps=1e-6)
        self.norm = LayerNorm2d(dim, eps=1e-6)

    def _get_norm(self, dim, n_head, **kwargs):
        if self.norm_type == 'layer':
            norm = nn.LayerNorm
        elif self.norm_type == "batch":
            norm = nn.BatchNorm1d
        elif self.norm_type == "instance":
            norm = nn.InstanceNorm1d
        else:
            norm = Identity
        return nn.ModuleList(
            [copy.deepcopy(norm(dim, **kwargs)) for _ in range(n_head)])

    def forward(self, x):

        _, _, h, w = x.size()
        x = self.pe(x)

        #B, inner_dim, H, W
        qkv = self.to_qkv(x)
        q, k, v = qkv.chunk(3, dim=1)

        q, k, v = map(lambda t: rearrange(t, 'b (dim_head heads) h w -> b heads (h w) dim_head',
                                          dim_head=self.dim_head,
                                          heads=self.heads,
                                          h=h, w=w), (q, k, v))

        q = torch.stack(
            [norm(x) for norm, x in
             zip(self.norm_q, (q[:, i, ...] for i in range(self.heads)))], dim=1)
        k = torch.stack(
            [norm(x) for norm, x in
             zip(self.norm_k, (k[:, i, ...] for i in range(self.heads)))], dim=1)

        q_k_attn = torch.einsum('bhid,bhjd->bhij', q, k)

        q_k_attn *= self.attn_factor
        if self.softmax:
            q_k_attn = F.softmax(q_k_attn, dim=-1)
        else:
            q_k_attn /= (h*w)

        q_k_attn = self.attn_drop(q_k_attn)

        out = torch.einsum('bhij,bhjd->bhid', q_k_attn, v)

        if self.skip_connection:
            out = out + v

        out = torch.stack(
            [norm(x) for norm, x in
             zip(self.norm_out, (out[:, i, ...] for i in range(self.heads)))], dim=1)

        out = rearrange(out, 'b heads (h w) dim_head -> b (dim_head heads) h w',
                        h=h, w=w, dim_head=self.dim_head, heads=self.heads)

        out = self.to_out(out)

        out = self.proj_drop(out)

        out = self.norm(out)

        if self.return_attention:
            return out, q_k_attn
        else:
            return out


class CrossConv(nn.Module):
    def __init__(self, dim, dim_c,
                 scale_factor=2):
        super(CrossConv, self).__init__()

        self.dim = dim  # dim = C
        self.dim_c = dim_c  # dim_c = 2*C
        self.convt = nn.ConvTranspose2d(
            dim_c, dim, scale_factor, stride=scale_factor)

    def forward(self, xf, xc):
        x = self.convt(xc)
        x = torch.cat([xf, x], dim=1)
        return x


class CrossAttention(nn.Module):
    def __init__(self, dim,
                 dim_c,
                 scale_factor=[2, 2],
                 heads=4,
                 dim_head=64,
                 attn_drop=0.1,
                 proj_drop=0.1,
                 skip_connection=False,
                 hadamard=False,
                 softmax=True,
                 pe_trainable=False,
                 sinosoidal_pe=False,
                 bias=False,
                 return_attn=False,
                 debug=False):
        super(CrossAttention, self).__init__()

        self.heads = heads
        self.dim_head = dim // heads * 2 if dim_head is None else dim_head
        self.inner_dim = self.dim_head * heads  # like dim_feedforward
        self.c2f_factor = scale_factor
        self.f2c_factor = [1/s for s in scale_factor]
        self.attn_factor = self.dim_head ** (-0.5)
        self.bias = bias
        self.hadamard = hadamard
        self.softmax = softmax
        self.skip_connection = skip_connection
        self.return_attn = return_attn
        self.debug = debug
        self.dim = dim
        self.dim_c = dim_c

        self.q_proj = depthwise_separable_conv(
            self.dim_c, self.inner_dim, bias=self.bias)
        self.k_proj = depthwise_separable_conv(
            self.dim_c, self.inner_dim, bias=self.bias)
        self.v_proj = depthwise_separable_conv(
            self.dim, self.inner_dim, bias=self.bias)
        self.out_proj = depthwise_separable_conv(
            self.inner_dim, self.dim, bias=self.bias)
        self.skip_proj = depthwise_separable_conv(
            self.dim_c, self.dim, bias=self.bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)

        PE = PositionalEncoding2D if sinosoidal_pe else EuclideanPositionEncoding
        self.pe = PE(self.dim, trainable=pe_trainable)
        self.pe_c = PE(self.dim_c, trainable=pe_trainable)
        self.norm_k = self._get_norm(self.dim_head, self.heads, eps=1e-6)
        self.norm_q = self._get_norm(self.dim_head, self.heads, eps=1e-6)
        self.norm_out = LayerNorm2d(2*self.dim, eps=1e-6)

    def _get_norm(self, dim, n_head, norm=None, **kwargs):
        norm = nn.LayerNorm if norm is None else norm
        return nn.ModuleList(
            [copy.deepcopy(norm(dim, **kwargs)) for _ in range(n_head)])

    def forward(self, xf, xc):

        _, _, hf, wf = xf.size()
        xf = self.pe(xf)

        _, _, ha, wa = xc.size()
        xc = self.pe_c(xc)

        #B, inner_dim, H, W
        q = self.q_proj(xc)
        k = self.k_proj(xc)
        v = self.v_proj(xf)

        res = self.skip_proj(xc)
        res = F.interpolate(res, scale_factor=self.c2f_factor,
                            mode='bilinear',
                            align_corners=True,
                            recompute_scale_factor=True)

        q, k = map(lambda t: rearrange(t, 'b (dim_head heads) h w -> b heads (h w) dim_head',
                   dim_head=self.dim_head, heads=self.heads, h=ha, w=wa), (q, k))

        v = F.interpolate(v, scale_factor=self.f2c_factor,
                          mode='bilinear',
                          align_corners=True,
                          recompute_scale_factor=True)
        v = rearrange(v, 'b (dim_head heads) h w -> b heads (h w) dim_head',
                      dim_head=self.dim_head, heads=self.heads, h=ha, w=wa)

        q = torch.stack(
            [norm(x) for norm, x in
             zip(self.norm_q, (q[:, i, ...] for i in range(self.heads)))], dim=1)
        k = torch.stack(
            [norm(x) for norm, x in
             zip(self.norm_k, (k[:, i, ...] for i in range(self.heads)))], dim=1)

        q_k_attn = torch.einsum('bhid,bhjd->bhij', q, k)

        q_k_attn *= self.attn_factor

        if self.softmax:
            q_k_attn = F.softmax(q_k_attn, dim=-1)
        else:
            q_k_attn /= (ha*wa)
        q_k_attn = self.attn_drop(q_k_attn)

        out = torch.einsum('bhij,bhjd->bhid', q_k_attn, v)
        out = rearrange(out, 'b heads (h w) dim_head -> b (dim_head heads) h w',
                        h=ha, w=wa, dim_head=self.dim_head, heads=self.heads)

        out = self.out_proj(out)
        out = self.proj_drop(out)

        out = F.interpolate(out, scale_factor=self.c2f_factor,
                            mode='bilinear',
                            align_corners=True,
                            recompute_scale_factor=True)

        if self.hadamard:
            out = torch.sigmoid(out)
            out = out*xf

        if self.skip_connection:
            out = out+xf

        out = torch.cat([out, res], dim=1)

        out = self.norm_out(out)

        if self.return_attn:
            return out, q_k_attn
        else:
            return out


class DownBlock(nn.Module):
    """Downscaling with interp then double conv"""

    def __init__(self, in_channels,
                 out_channels,
                 scale_factor=[0.5, 0.5],
                 batch_norm=True,
                 activation='relu'):
        super().__init__()
        self.scale_factor = scale_factor
        self.conv = DoubleConv(in_channels, out_channels,
                               batch_norm=batch_norm,
                               activation=activation)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=self.scale_factor,
                          mode='bilinear',
                          align_corners=True,
                          recompute_scale_factor=True)
        return self.conv(x)


class UpBlock(nn.Module):
    def __init__(self, nc_coarse, nc_fine,
                 heads=4,
                 activation='relu',
                 hadamard=False,
                 attention=True,
                 softmax=True,
                 skip_connection=False,
                 sinosoidal_pe=False,
                 pe_trainable=False,
                 batch_norm=False,
                 debug=False):
        super(UpBlock, self).__init__()
        if attention:
            self.up = CrossAttention(nc_fine,
                                     nc_coarse,
                                     heads=heads,
                                     dim_head=nc_coarse//2,
                                     skip_connection=skip_connection,
                                     hadamard=hadamard,
                                     softmax=softmax,
                                     sinosoidal_pe=sinosoidal_pe,
                                     pe_trainable=pe_trainable)
        else:
            self.up = CrossConv(nc_fine, nc_coarse)

        self.conv = DoubleConv(nc_coarse, nc_fine,
                               batch_norm=batch_norm,
                               activation=activation)
        self.debug = debug

    def forward(self, xf, xc):
        x = self.up(xf, xc)
        x = self.conv(x)
        return x


class UTransformer(nn.Module):
    def __init__(self, in_channels,
                 out_channels,
                 dim=64,
                 heads=4,
                 input_size=(224, 224),
                 activation='gelu',
                 attention_coarse=True,
                 attention_up=True,
                 batch_norm=False,
                 attn_norm_type='layer',
                 skip_connection=True,
                 softmax=True,
                 pe_trainable=False,
                 hadamard=False,
                 sinosoidal_pe=False,
                 add_grad_channel=True,
                 out_latent=False,
                 debug=False,
                 **kwargs):
        super(UTransformer, self).__init__()

        self.inc = DoubleConv(in_channels, dim,
                              activation=activation,
                              batch_norm=batch_norm)
        self.down1 = DownBlock(dim, 2*dim,
                               activation=activation,
                               batch_norm=batch_norm)
        self.down2 = DownBlock(2*dim, 4*dim,
                               activation=activation,
                               batch_norm=batch_norm)
        self.down3 = DownBlock(4*dim, 8*dim,
                               activation=activation,
                               batch_norm=batch_norm)
        if attention_coarse:
            self.up0 = Attention(8*dim, heads=heads,
                                 softmax=softmax,
                                 norm_type=attn_norm_type,
                                 sinosoidal_pe=sinosoidal_pe,
                                 pe_trainable=pe_trainable,
                                 skip_connection=skip_connection)
        else:
            self.up0 = DoubleConv(8*dim, 8*dim,
                                  activation=activation,
                                  batch_norm=batch_norm)
        self.up1 = UpBlock(8*dim, 4*dim,
                           heads=heads,
                           batch_norm=batch_norm,
                           hadamard=hadamard,
                           attention=attention_up,
                           softmax=softmax,
                           sinosoidal_pe=sinosoidal_pe,
                           pe_trainable=pe_trainable)
        self.up2 = UpBlock(4*dim, 2*dim,
                           heads=heads,
                           batch_norm=batch_norm,
                           hadamard=hadamard,
                           attention=attention_up,
                           softmax=softmax,
                           sinosoidal_pe=sinosoidal_pe,
                           pe_trainable=pe_trainable)
        self.up3 = UpBlock(2*dim, dim,
                           heads=heads,
                           batch_norm=batch_norm,
                           hadamard=hadamard,
                           attention=attention_up,
                           softmax=softmax,
                           sinosoidal_pe=sinosoidal_pe,
                           pe_trainable=pe_trainable,)
        self.out = OutConv(dim, out_channels)
        self.out_latent = out_latent
        self.add_grad_channel = add_grad_channel
        self.input_size = input_size
        self.debug = debug

    def forward(self, x, gradx, *args, **kwargs):
        "input dim: bsz, n, n, C"
        if gradx.ndim == x.ndim and self.add_grad_channel:
            x = torch.cat([x, gradx], dim=-1)
        x = x.permute(0, 3, 1, 2)

        if self.input_size:
            _, _, *size = x.size()
            x = F.interpolate(x, size=self.input_size,
                              mode='bilinear',
                              align_corners=True)

        x1 = self.inc(x)

        x2 = self.down1(x1)

        x3 = self.down2(x2)

        x4 = self.down3(x3)

        x4 = self.up0(x4)

        x = self.up1(x3, x4)

        x = self.up2(x2, x)

        x = self.up3(x1, x)

        out = self.out(x)

        if self.input_size:
            out = F.interpolate(out, size=size,
                                mode='bilinear',
                                align_corners=True)

            out = out.permute(0, 2, 3, 1)

        if self.out_latent:
            return dict(preds=out,
                        preds_latent=[x2, x3, x4])
        else:
            return dict(preds=out)
