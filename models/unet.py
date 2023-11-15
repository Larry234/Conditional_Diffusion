import math
from telnetlib import PRAGMA_HEARTBEAT
import torch
from torch import nn
from torch.nn import init
from torch.nn import functional as F
from .embedding import TimeEmbedding, ConditionalEmbedding, LabelEmbedding
from .attention import SpatialTransformer, AttentionBlock

def drop_connect(x, drop_ratio):
    keep_ratio = 1.0 - drop_ratio
    mask = torch.empty([x.shape[0], 1, 1, 1], dtype=x.dtype, device=x.device)
    mask.bernoulli_(p=keep_ratio)
    x.div_(keep_ratio)
    x.mul_(mask)
    return x

class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

class UpSample(nn.Module):
    """
    ### Up-sampling layer
    """

    def __init__(self, channels: int):
        """
        :param channels: is the number of channels
        """
        super().__init__()
        # $3 \times 3$ convolution mapping
        self.conv = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x: torch.Tensor):
        """
        :param x: is the input feature map with shape `[batch_size, channels, height, width]`
        """
        # Up-sample by a factor of $2$
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        # Apply convolution
        return self.conv(x)


    
class DownSample(nn.Module):
    """
    ## Down-sampling layer
    """

    def __init__(self, channels: int):
        """
        :param channels: is the number of channels
        """
        super().__init__()
        # $3 \times 3$ convolution with stride length of $2$ to down-sample by a factor of $2$
        self.op = nn.Conv2d(channels, channels, 3, stride=2, padding=1)

    def forward(self, x: torch.Tensor):
        """
        :param x: is the input feature map with shape `[batch_size, channels, height, width]`
        """
        # Apply convolution
        return self.op(x)



class AttnBlock(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        self.group_norm = nn.GroupNorm(32, in_ch)
        self.proj_q = nn.Conv2d(in_ch, in_ch, 1, stride=1, padding=0)
        self.proj_k = nn.Conv2d(in_ch, in_ch, 1, stride=1, padding=0)
        self.proj_v = nn.Conv2d(in_ch, in_ch, 1, stride=1, padding=0)
        self.proj = nn.Conv2d(in_ch, in_ch, 1, stride=1, padding=0)

    def forward(self, x):
        B, C, H, W = x.shape
        h = self.group_norm(x)
        q = self.proj_q(h)
        k = self.proj_k(h)
        v = self.proj_v(h)

        q = q.permute(0, 2, 3, 1).view(B, H * W, C)
        k = k.view(B, C, H * W)
        w = torch.bmm(q, k) * (int(C) ** (-0.5))
        assert list(w.shape) == [B, H * W, H * W]
        w = F.softmax(w, dim=-1)

        v = v.permute(0, 2, 3, 1).view(B, H * W, C)
        h = torch.bmm(w, v)
        assert list(h.shape) == [B, H * W, C]
        h = h.view(B, H, W, C).permute(0, 3, 1, 2)
        h = self.proj(h)

        return x + h


class TimestepEmbedSequential(nn.Sequential):
    """
    A sequential module that passes timestep embeddings to the children that
    support it as an extra input.
    """

    def forward(self, x, emb, condition=None, context=None):
        for layer in self:
            if isinstance(layer, ResBlockCond):
                x = layer(x, emb, condition)
            elif isinstance(layer, ResBlock):
                x = layer(x, emb)
            elif isinstance(layer, SpatialTransformer):
                x = layer(x, context)
            else:
                x = layer(x)
        return x

    
class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, tdim, dropout, attn=False):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.GroupNorm(32, in_ch),
            Swish(),
            nn.Conv2d(in_ch, out_ch, 3, stride=1, padding=1),
        )
        self.temb_proj = nn.Sequential(
            Swish(),
            nn.Linear(tdim, out_ch),
        )
        self.block2 = nn.Sequential(
            nn.GroupNorm(32, out_ch),
            Swish(),
            nn.Dropout(dropout),
            nn.Conv2d(out_ch, out_ch, 3, stride=1, padding=1),
        )
        if in_ch != out_ch:
            self.shortcut = nn.Conv2d(in_ch, out_ch, 1, stride=1, padding=0)
        else:
            self.shortcut = nn.Identity()
        if attn:
            self.attn = AttnBlock(out_ch)
        else:
            self.attn = nn.Identity()
        
    def forward(self, x, temb, conditioned=None):
        h = self.block1(x)
        h += self.temb_proj(temb)[:, :, None, None]
        h = self.block2(h)

        h = h + self.shortcut(x)
        h = self.attn(h)
        return h
        
class ResBlockCond(ResBlock):
    def __init__(self, in_ch, out_ch, tdim, dropout, attn=False):
        super().__init__(in_ch, out_ch, tdim, dropout, attn)
        self.cond_proj = nn.Sequential(
            Swish(),
            nn.Linear(tdim, out_ch),
        )

    def forward(self, x, temb, cond):
        h = self.block1(x)
        h += self.temb_proj(temb)[:, :, None, None]
        h += self.cond_proj(cond)[:, :, None, None]
        h = self.block2(h)

        h = h + self.shortcut(x)
        h = self.attn(h)
        return h
    
class ResBlockDualCondition(ResBlock):
    def __init__(self, in_ch, out_ch, tdim, dropout, attn=False):
        super().__init__(in_ch, out_ch, tdim, dropout, attn)
        self.cond_proj = nn.Sequential(
            Swish(),
            nn.Linear(tdim, out_ch // 2),
        )
        self.com_proj = nn.Sequential(
            Swish(),
            nn.Linear(tdim, out_ch // 2),
        )

    def forward(self, x, temb, labels, labels_1):
        h = self.block1(x)
        h += self.temb_proj(temb)[:, :, None, None]
        a = self.cond_proj(labels)[:, :, None, None]
        b = self.com_proj(labels_1)[:, :, None, None]
        h += torch.cat((a, b), dim = 1)
        h = self.block2(h)

        h = h + self.shortcut(x)
        h = self.attn(h)
        return h
    
    
class ResBlockImageClassConcat(ResBlock):
    def __init__(self, in_ch: int, out_ch: int, tdim: int, dropout: float, attn=False):
        
        super().__init__(in_ch, out_ch, tdim, dropout, attn)
        self.cond_proj = nn.Sequential(
            Swish(),
            nn.Linear(tdim, out_ch // 2),
        )
        self.com_proj = nn.Sequential(
            Swish(),
            nn.Linear(tdim, out_ch // 2),
        )
        self.bottleneck = nn.Conv2d(out_ch*2, out_ch, 1, stride=1, padding=0)

    def forward(self, x, temb, c1, c2):
        
        h = self.block1(x)
        temb = self.temb_proj(temb)[:, :, None, None]
        c1 = self.cond_proj(c1)[:, :, None, None]
        c2 = self.com_proj(c2)[:, :, None, None]
        c = torch.cat([c1, c2], dim=1).expand(-1, -1, h.shape[-2], h.shape[-1])
        h = torch.cat([h, c], dim=1)
        h = self.bottleneck(h)
        h += temb
        h = self.block2(h)
        h = h + self.shortcut(x)
        h = self.attn(h)
        return h
    
class ResBlockImageClassConcatTri(ResBlock):
    def __init__(self, in_ch: int, out_ch: int, tdim: int, dropout: float, attn=False):
        
        super().__init__(in_ch, out_ch, tdim, dropout, attn)
        
        self.size_proj = nn.Sequential(
            Swish(),
            nn.Linear(tdim, out_ch),
        )

        self.atr_proj = nn.Sequential(
            Swish(),
            nn.Linear(tdim, out_ch),
        )
        self.obj_proj = nn.Sequential(
            Swish(),
            nn.Linear(tdim, out_ch),
        )
        self.bottleneck = nn.Conv2d(out_ch*4, out_ch, 1, stride=1, padding=0)

    def forward(self, x, temb, size, atr, obj):
        
        h = self.block1(x)
        temb = self.temb_proj(temb)[:, :, None, None]
        c1 = self.size_proj(size)[:, :, None, None]
        c2 = self.atr_proj(atr)[:, :, None, None]
        c3 = self.obj_proj(obj)[:, :, None, None]
        c = torch.cat([c1, c2, c3], dim=1).expand(-1, -1, h.shape[-2], h.shape[-1])
        h = torch.cat([h, c], dim=1)
        h = self.bottleneck(h)
        h += temb
        h = self.block2(h)
        h = h + self.shortcut(x)
        h = self.attn(h)
        return h

    
class UNetTriple(nn.Module):
    def __init__(self, T, num_size, num_atr, num_obj, model_channels, ch_mult, num_res_blocks, dropout, drop_prob=0.1, only_table=False):
        super().__init__()
        tdim = model_channels * 4
        self.time_embedding = TimeEmbedding(T, model_channels, tdim)
        if only_table:
            self.size_embedding = LabelEmbedding(num_size, model_channels, drop_prob)
            self.atr_embedding = LabelEmbedding(num_atr, model_channels, drop_prob)
            self.obj_embedding = LabelEmbedding(num_obj, model_channels, drop_prob)
        else:
            self.size_embedding = ConditionalEmbedding(num_size, model_channels, tdim, drop_prob)
            self.atr_embedding = ConditionalEmbedding(num_atr, model_channels, tdim, drop_prob)
            self.obj_embedding = ConditionalEmbedding(num_obj, model_channels, tdim, drop_prob)

        self.input_blocks = nn.ModuleList(
            [
                TimestepEmbedSequential(
                    nn.Conv2d(3, model_channels, 3, stride=1, padding=1)
                )
            ]
        )
        self._feature_size = model_channels
        input_block_chans = [model_channels]
        ch = model_channels
        ds = 1
        for level, mult in enumerate(ch_mult):
            for _ in range(num_res_blocks):
                layers = [
                    ResBlock(
                        in_ch=ch,
                        out_ch=mult * model_channels,
                        tdim=tdim,
                        dropout=dropout,
                    )
                ]
                ch = mult * model_channels
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch
                input_block_chans.append(ch)
            if level != len(ch_mult) - 1:
                out_ch = ch
                self.input_blocks.append(
                    TimestepEmbedSequential(
                        DownSample(ch)
                    )
                )
                ch = out_ch
                input_block_chans.append(ch)
                ds *= 2
                self._feature_size += ch

        self.middle_block = TimestepEmbedSequential(
            ResBlock(
                in_ch=ch,
                out_ch=ch,
                tdim=tdim,
                dropout=dropout,
            ),
            ResBlock(
                in_ch=ch,
                out_ch=ch,
                tdim=tdim,
                dropout=dropout,
            ),
        )
        self._feature_size += ch

        self.output_blocks = nn.ModuleList([])
        for level, mult in list(enumerate(ch_mult))[::-1]:
            for i in range(num_res_blocks + 1):
                ich = input_block_chans.pop()
                layers = [
                    ResBlock(
                        in_ch=ch + ich,
                        out_ch=model_channels * mult,
                        tdim=tdim,
                        dropout=dropout,
                    )
                ]
                ch = model_channels * mult
                
                if level and i == num_res_blocks:
                    out_ch = ch
                    layers.append(
                        UpSample(ch)
                    )
                    ds //= 2
                self.output_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch

        self.out = nn.Sequential(
            nn.GroupNorm(32, ch),
            nn.SiLU(),
            nn.Conv2d(ch, 3, 3, stride=1, padding=1),
        )
 

    def forward(self, x, t, size, atr, obj, force_drop_ids=None):
        # Timestep embedding
        temb = self.time_embedding(t)
        semb = self.size_embedding(size, force_drop_ids=force_drop_ids)
        cemb = self.atr_embedding(atr, force_drop_ids=force_drop_ids)
        cemb1 = self.obj_embedding(obj, force_drop_ids=force_drop_ids)
        temb = temb + semb + cemb + cemb1
        # Downsampling
        hs = []
        h = x
        for layer in self.input_blocks:
            h = layer(h, temb)
            hs.append(h)
        # Middle
        h = self.middle_block(h, temb)
        # Upsampling
        for layer in self.output_blocks:
            h = torch.cat([h, hs.pop()], dim=1)
            h = layer(h, temb)
        h = self.out(h)

        assert len(hs) == 0
        return h


class UNetICTriple(nn.Module):
    def __init__(self, T, num_size, num_atr, num_obj, ch, ch_mult, num_res_blocks, dropout, drop_prob=0.1):
        super().__init__()
        tdim = ch * 4
        self.time_embedding = TimeEmbedding(T, ch, tdim)
        self.size_embedding = ConditionalEmbedding(num_size, ch, tdim, drop_prob)
        self.atr_embedding = ConditionalEmbedding(num_atr, ch, tdim, drop_prob)
        self.obj_embedding = ConditionalEmbedding(num_obj, ch, tdim, drop_prob)
        self.head = nn.Conv2d(3, ch, kernel_size=3, stride=1, padding=1)
        self.downblocks = nn.ModuleList()
        chs = [ch]  # record output channel when dowmsample for upsample
        now_ch = ch
        for i, mult in enumerate(ch_mult):
            out_ch = ch * mult
            for _ in range(num_res_blocks):
                self.downblocks.append(ResBlockImageClassConcatTri(in_ch=now_ch, out_ch=out_ch, tdim=tdim, dropout=dropout))
                now_ch = out_ch
                chs.append(now_ch)
            if i != len(ch_mult) - 1:
                self.downblocks.append(DownSample(now_ch))
                chs.append(now_ch)

        self.middleblocks = nn.ModuleList([
            ResBlockImageClassConcatTri(now_ch, now_ch, tdim, dropout, attn=False),
            ResBlockImageClassConcatTri(now_ch, now_ch, tdim, dropout, attn=False),
        ])

        self.upblocks = nn.ModuleList()
        for i, mult in reversed(list(enumerate(ch_mult))):
            out_ch = ch * mult
            for _ in range(num_res_blocks + 1):
                self.upblocks.append(ResBlockImageClassConcatTri(in_ch=chs.pop() + now_ch, out_ch=out_ch, tdim=tdim, dropout=dropout, attn=False))
                now_ch = out_ch
            if i != 0:
                self.upblocks.append(UpSample(now_ch))
        assert len(chs) == 0

        self.tail = nn.Sequential(
            nn.GroupNorm(32, now_ch),
            Swish(),
            nn.Conv2d(now_ch, 3, 3, stride=1, padding=1)
        )
 

    def forward(self, x, t, size, atr, obj, force_drop_ids=None):
        # Timestep embedding
        temb = self.time_embedding(t)
        size_emb = self.size_embedding(size, force_drop_ids=force_drop_ids)
        atr_emb = self.atr_embedding(atr, force_drop_ids=force_drop_ids)
        obj_emb = self.obj_embedding(obj, force_drop_ids=force_drop_ids)
        
        # Downsampling
        h = self.head(x)
        hs = [h]
        for layer in self.downblocks:
            if isinstance(layer, DownSample):
                h = layer(h)
            else:
                h = layer(h, temb, size_emb, atr_emb, obj_emb)
            hs.append(h)
        # Middle
        for layer in self.middleblocks:
            h = layer(h, temb, size_emb, atr_emb, obj_emb)
        # Upsampling
        for layer in self.upblocks:
            if isinstance(layer, ResBlock):
                h = torch.cat([h, hs.pop()], dim=1)
            if isinstance(layer, UpSample):
                h = layer(h)
            else:
                h = layer(h, temb, size_emb, atr_emb, obj_emb)
        h = self.tail(h)

        assert len(hs) == 0
        return h
        
        
class UNetEncoderAttention(nn.Module):
    """
    The full UNet model with attention and timestep embedding.
    :param in_channels: channels in the input Tensor.
    :param model_channels: base channel count for the model.
    :param out_channels: channels in the output Tensor.
    :param num_res_blocks: number of residual blocks per downsample.
    :param attention_resolutions: a collection of downsample rates at which
        attention will take place. May be a set, list, or tuple.
        For example, if this contains 4, then at 4x downsampling, attention
        will be used.
    :param dropout: the dropout probability.
    :param channel_mult: channel multiplier for each level of the UNet.
    :param conv_resample: if True, use learned convolutions for upsampling and
        downsampling.
    :param dims: determines if the signal is 1D, 2D, or 3D.
    :param num_classes: if specified (as an int), then this model will be
        class-conditional with `num_classes` classes.
    :param use_checkpoint: use gradient checkpointing to reduce memory usage.
    :param num_heads: the number of attention heads in each attention layer.
    :param num_heads_channels: if specified, ignore num_heads and instead use
                               a fixed channel width per attention head.
    :param num_heads_upsample: works with num_heads to set a different number
                               of heads for upsampling. Deprecated.
    :param use_scale_shift_norm: use a FiLM-like conditioning mechanism.
    :param resblock_updown: use residual blocks for up/downsampling.
    :param use_new_attention_order: use a different attention pattern for potentially
                                    increased efficiency.
    """

    def __init__(
        self,
        T,
        image_size,
        in_channels,
        model_channels,
        out_channels,
        num_res_blocks,
        attention_resolutions,
        dropout=0,
        drop_prob=0.1,
        channel_mult=(1, 2, 4, 8),
        use_checkpoint=False,
        num_heads=-1,
        num_head_channels=-1,
        use_scale_shift_norm=False,
        resblock_updown=False,
        use_new_attention_order=False,
        use_spatial_transformer=False,    # custom transformer support
        transformer_depth=1,              # custom transformer support
        context_dim=None,                 # custom transformer support
        n_embed=None,                     # custom support for prediction of discrete ids into codebook of first stage vq model
        legacy=True,
    ):
        super().__init__()
        if num_heads == -1:
            assert num_head_channels != -1, 'Either num_heads or num_head_channels has to be set'

        if num_head_channels == -1:
            assert num_heads != -1, 'Either num_heads or num_head_channels has to be set'

        self.image_size = image_size
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.use_checkpoint = use_checkpoint
        self.dtype = torch.float32
        self.num_heads = num_heads
        self.num_head_channels = num_head_channels
        self.predict_codebook_ids = n_embed is not None

        time_embed_dim = model_channels * 4
        self.time_embed = TimeEmbedding(T, model_channels, time_embed_dim)

        self.input_blocks = nn.ModuleList(
            [
                TimestepEmbedSequential(
                    nn.Conv2d(in_channels, model_channels, 3, stride=1, padding=1)
                )
            ]
        )
        self._feature_size = model_channels
        input_block_chans = [model_channels]
        ch = model_channels
        ds = 1
        for level, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks):
                layers = [
                    ResBlock(
                        in_ch=ch,
                        out_ch=mult * model_channels,
                        tdim=time_embed_dim,
                        dropout=dropout,
                    )
                ]
                ch = mult * model_channels
                if ds in attention_resolutions:
                    if num_head_channels == -1:
                        dim_head = ch // num_heads
                    else:
                        num_heads = ch // num_head_channels
                        dim_head = num_head_channels
                    if legacy:
                        #num_heads = 1
                        dim_head = ch // num_heads if use_spatial_transformer else num_head_channels
                    layers.append(
                        AttentionBlock(
                            ch,
                            use_checkpoint=use_checkpoint,
                            num_heads=num_heads,
                            num_head_channels=dim_head,
                            use_new_attention_order=use_new_attention_order,
                        ) if not use_spatial_transformer else SpatialTransformer(
                            ch, num_heads, dim_head, depth=transformer_depth, context_dim=context_dim
                        )
                    )
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch
                input_block_chans.append(ch)
            if level != len(channel_mult) - 1:
                out_ch = ch
                self.input_blocks.append(
                    TimestepEmbedSequential(
                        ResBlock(
                            in_ch=ch,
                            out_ch=out_ch,
                            tdim=time_embed_dim,
                            dropout=dropout,
                        )
                        if resblock_updown
                        else DownSample(ch)
                    )
                )
                ch = out_ch
                input_block_chans.append(ch)
                ds *= 2
                self._feature_size += ch

        if num_head_channels == -1:
            dim_head = ch // num_heads
        else:
            num_heads = ch // num_head_channels
            dim_head = num_head_channels
        if legacy:
            #num_heads = 1
            dim_head = ch // num_heads if use_spatial_transformer else num_head_channels
        self.middle_block = TimestepEmbedSequential(
            ResBlock(
                in_ch=ch,
                out_ch=ch,
                tdim=time_embed_dim,
                dropout=dropout,
            ),
            AttentionBlock(
                ch,
                use_checkpoint=use_checkpoint,
                num_heads=num_heads,
                num_head_channels=dim_head,
                use_new_attention_order=use_new_attention_order,
            ) if not use_spatial_transformer else SpatialTransformer(
                ch, num_heads, dim_head, depth=transformer_depth, context_dim=context_dim
            ),
            ResBlock(
                in_ch=ch,
                out_ch=ch,
                tdim=time_embed_dim,
                dropout=dropout,
            ),
        )
        self._feature_size += ch

        self.output_blocks = nn.ModuleList([])
        for level, mult in list(enumerate(channel_mult))[::-1]:
            for i in range(num_res_blocks + 1):
                ich = input_block_chans.pop()
                layers = [
                    ResBlock(
                        in_ch=ch + ich,
                        out_ch=model_channels * mult,
                        tdim=time_embed_dim,
                        dropout=dropout,
                    )
                ]
                ch = model_channels * mult
                if ds in attention_resolutions:
                    if num_head_channels == -1:
                        dim_head = ch // num_heads
                    else:
                        num_heads = ch // num_head_channels
                        dim_head = num_head_channels
                    if legacy:
                        #num_heads = 1
                        dim_head = ch // num_heads if use_spatial_transformer else num_head_channels
                    layers.append(
                        AttentionBlock(
                            ch,
                            use_checkpoint=use_checkpoint,
                            num_heads=num_heads,
                            num_head_channels=dim_head,
                            use_new_attention_order=use_new_attention_order,
                        ) if not use_spatial_transformer else SpatialTransformer(
                            ch, num_heads, dim_head, depth=transformer_depth, context_dim=context_dim
                        ),
                    )
                if level and i == num_res_blocks:
                    out_ch = ch
                    layers.append(
                        ResBlock(
                            in_ch=ch,
                            out_ch=out_ch,
                            tdim=time_embed_dim,
                            dropout=dropout,
                        )
                        if resblock_updown
                        else UpSample(ch)
                    )
                    ds //= 2
                self.output_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch

        self.out = nn.Sequential(
            nn.GroupNorm(32, ch),
            nn.SiLU(),
            nn.Conv2d(ch, out_channels, 3, stride=1, padding=1),
        )
#         if self.predict_codebook_ids:
#             self.id_predictor = nn.Sequential(
#             normalization(ch),
#             conv_nd(dims, model_channels, n_embed, 1),
#             #nn.LogSoftmax(dim=1)  # change to cross_entropy and produce non-normalized logits
#         )


    def forward(self, x, timesteps=None, context=None, **kwargs):
        """
        Apply the model to an input batch.
        :param x: an [N x C x ...] Tensor of inputs.
        :param timesteps: a 1-D batch of timesteps.
        :param c1: an [N] Tensor of labels.
        :param c2: an [N] Tensor of labels.
        :return: an [N x C x ...] Tensor of outputs.
        """
        hs = []
        emb = self.time_embed(timesteps)
#         context = context[:, None, :]
        h = x.type(self.dtype)
        for module in self.input_blocks:
            h = module(h, emb, context=context)
            hs.append(h)
        h = self.middle_block(h, emb, context=context)
        for module in self.output_blocks:
            h = torch.cat([h, hs.pop()], dim=1)
            h = module(h, emb, context=context)
        h = h.type(x.dtype)
        if self.predict_codebook_ids:
            return self.id_predictor(h)
        else:
            return self.out(h)
        
        
class UNetAttentionTripleCond(nn.Module):
    """
    The full UNet model with attention and timestep embedding.
    :param in_channels: channels in the input Tensor.
    :param model_channels: base channel count for the model.
    :param out_channels: channels in the output Tensor.
    :param num_res_blocks: number of residual blocks per downsample.
    :param attention_resolutions: a collection of downsample rates at which
        attention will take place. May be a set, list, or tuple.
        For example, if this contains 4, then at 4x downsampling, attention
        will be used.
    :param dropout: the dropout probability.
    :param channel_mult: channel multiplier for each level of the UNet.
    :param conv_resample: if True, use learned convolutions for upsampling and
        downsampling.
    :param dims: determines if the signal is 1D, 2D, or 3D.
    :param num_classes: if specified (as an int), then this model will be
        class-conditional with `num_classes` classes.
    :param use_checkpoint: use gradient checkpointing to reduce memory usage.
    :param num_heads: the number of attention heads in each attention layer.
    :param num_heads_channels: if specified, ignore num_heads and instead use
                               a fixed channel width per attention head.
    :param num_heads_upsample: works with num_heads to set a different number
                               of heads for upsampling. Deprecated.
    :param use_scale_shift_norm: use a FiLM-like conditioning mechanism.
    :param resblock_updown: use residual blocks for up/downsampling.
    :param use_new_attention_order: use a different attention pattern for potentially
                                    increased efficiency.
    """

    def __init__(
        self,
        T,
        image_size,
        in_channels,
        model_channels,
        out_channels,
        num_res_blocks,
        attention_resolutions,
        dropout=0,
        drop_prob=0.1,
        channel_mult=(1, 2, 4, 8),
        num_size=None,
        num_obj=None,
        num_atr=None,
        use_checkpoint=False,
        num_heads=-1,
        num_head_channels=-1,
        use_scale_shift_norm=False,
        resblock_updown=False,
        use_new_attention_order=False,
        use_spatial_transformer=False,    # custom transformer support
        transformer_depth=1,              # custom transformer support
        context_dim=None,                 # custom transformer support
        n_embed=None,                     # custom support for prediction of discrete ids into codebook of first stage vq model
        legacy=True,
        concat=False
    ):
        super().__init__()
#         if use_spatial_transformer:
#             assert context_dim is not None, 'Fool!! You forgot to include the dimension of your cross-attention conditioning...'

#         if context_dim is not None:
#             assert use_spatial_transformer, 'Fool!! You forgot to use the spatial transformer for your cross-attention conditioning...'
#             from omegaconf.listconfig import ListConfig
#             if type(context_dim) == ListConfig:
#                 context_dim = list(context_dim)
        if num_heads == -1:
            assert num_head_channels != -1, 'Either num_heads or num_head_channels has to be set'

        if num_head_channels == -1:
            assert num_heads != -1, 'Either num_heads or num_head_channels has to be set'

        self.image_size = image_size
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.num_size = num_size
        self.num_obj = num_obj
        self.num_atr = num_atr
        self.use_checkpoint = use_checkpoint
        self.dtype = torch.float32
        self.num_heads = num_heads
        self.num_head_channels = num_head_channels
        self.predict_codebook_ids = n_embed is not None
        self.concat = concat

        time_embed_dim = model_channels * 4
        label_embed_dim = time_embed_dim // 2 if self.concat else time_embed_dim
        context_dim = model_channels * 4
        self.time_embed = TimeEmbedding(T, model_channels, time_embed_dim)

        if self.num_size is not None:
            self.size_emb = ConditionalEmbedding(num_size, model_channels, label_embed_dim, drop_prob)
        
        if self.num_obj is not None:
            self.obj_emb = ConditionalEmbedding(num_obj, model_channels, label_embed_dim, drop_prob)
            
        if self.num_atr is not None:
            self.atr_emb = ConditionalEmbedding(num_atr, model_channels, label_embed_dim, drop_prob)

        self.input_blocks = nn.ModuleList(
            [
                TimestepEmbedSequential(
                    nn.Conv2d(in_channels, model_channels, 3, stride=1, padding=1)
                )
            ]
        )
        self._feature_size = model_channels
        input_block_chans = [model_channels]
        ch = model_channels
        ds = 1
        for level, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks):
                layers = [
                    ResBlock(
                        in_ch=ch,
                        out_ch=mult * model_channels,
                        tdim=time_embed_dim,
                        dropout=dropout,
                    )
                ]
                ch = mult * model_channels
                if ds in attention_resolutions:
                    if num_head_channels == -1:
                        dim_head = ch // num_heads
                    else:
                        num_heads = ch // num_head_channels
                        dim_head = num_head_channels
                    if legacy:
                        #num_heads = 1
                        dim_head = ch // num_heads if use_spatial_transformer else num_head_channels
                    layers.append(
                        AttentionBlock(
                            ch,
                            use_checkpoint=use_checkpoint,
                            num_heads=num_heads,
                            num_head_channels=dim_head,
                            use_new_attention_order=use_new_attention_order,
                        ) if not use_spatial_transformer else SpatialTransformer(
                            ch, num_heads, dim_head, depth=transformer_depth, context_dim=context_dim
                        )
                    )
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch
                input_block_chans.append(ch)
            if level != len(channel_mult) - 1:
                out_ch = ch
                self.input_blocks.append(
                    TimestepEmbedSequential(
                        ResBlock(
                            in_ch=ch,
                            out_ch=out_ch,
                            tdim=time_embed_dim,
                            dropout=dropout,
                        )
                        if resblock_updown
                        else DownSample(ch)
                    )
                )
                ch = out_ch
                input_block_chans.append(ch)
                ds *= 2
                self._feature_size += ch

        if num_head_channels == -1:
            dim_head = ch // num_heads
        else:
            num_heads = ch // num_head_channels
            dim_head = num_head_channels
        if legacy:
            #num_heads = 1
            dim_head = ch // num_heads if use_spatial_transformer else num_head_channels
        self.middle_block = TimestepEmbedSequential(
            ResBlock(
                in_ch=ch,
                out_ch=ch,
                tdim=time_embed_dim,
                dropout=dropout,
            ),
            AttentionBlock(
                ch,
                use_checkpoint=use_checkpoint,
                num_heads=num_heads,
                num_head_channels=dim_head,
                use_new_attention_order=use_new_attention_order,
            ) if not use_spatial_transformer else SpatialTransformer(
                ch, num_heads, dim_head, depth=transformer_depth, context_dim=context_dim
            ),
            ResBlock(
                in_ch=ch,
                out_ch=ch,
                tdim=time_embed_dim,
                dropout=dropout,
            ),
        )
        self._feature_size += ch

        self.output_blocks = nn.ModuleList([])
        for level, mult in list(enumerate(channel_mult))[::-1]:
            for i in range(num_res_blocks + 1):
                ich = input_block_chans.pop()
                layers = [
                    ResBlock(
                        in_ch=ch + ich,
                        out_ch=model_channels * mult,
                        tdim=time_embed_dim,
                        dropout=dropout,
                    )
                ]
                ch = model_channels * mult
                if ds in attention_resolutions:
                    if num_head_channels == -1:
                        dim_head = ch // num_heads
                    else:
                        num_heads = ch // num_head_channels
                        dim_head = num_head_channels
                    if legacy:
                        #num_heads = 1
                        dim_head = ch // num_heads if use_spatial_transformer else num_head_channels
                    layers.append(
                        AttentionBlock(
                            ch,
                            use_checkpoint=use_checkpoint,
                            num_heads=num_heads,
                            num_head_channels=dim_head,
                            use_new_attention_order=use_new_attention_order,
                        ) if not use_spatial_transformer else SpatialTransformer(
                            ch, num_heads, dim_head, depth=transformer_depth, context_dim=context_dim
                        ),
                    )
                if level and i == num_res_blocks:
                    out_ch = ch
                    layers.append(
                        ResBlock(
                            in_ch=ch,
                            out_ch=out_ch,
                            tdim=time_embed_dim,
                            dropout=dropout,
                        )
                        if resblock_updown
                        else UpSample(ch)
                    )
                    ds //= 2
                self.output_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch

        self.out = nn.Sequential(
            nn.GroupNorm(32, ch),
            nn.SiLU(),
            nn.Conv2d(ch, out_channels, 3, stride=1, padding=1),
        )
#         if self.predict_codebook_ids:
#             self.id_predictor = nn.Sequential(
#             normalization(ch),
#             conv_nd(dims, model_channels, n_embed, 1),
#             #nn.LogSoftmax(dim=1)  # change to cross_entropy and produce non-normalized logits
#         )


    def forward(self, x, timesteps=None, c1=None, c2=None, c3=None, force_drop_ids=False, **kwargs):
        """
        Apply the model to an input batch.
        :param x: an [N x C x ...] Tensor of inputs.
        :param timesteps: a 1-D batch of timesteps.
        :param c1: an [N] Tensor of labels.
        :param c2: an [N] Tensor of labels.
        :return: an [N x C x ...] Tensor of outputs.
        """
        hs = []
        emb = self.time_embed(timesteps)
        context = None
        if self.num_obj is not None and self.num_atr is not None and self.num_size is not None:
            assert c1.shape == (x.shape[0],) and c2.shape == (x.shape[0],) and c3.shape == (x.shape[0],)
            c1 = self.size_emb(c1, force_drop_ids=force_drop_ids)[:, None, :]
            c2 = self.atr_emb(c2, force_drop_ids=force_drop_ids)[:, None, :]
            c3 = self.obj_emb(c3, force_drop_ids=force_drop_ids)[:, None, :]
            if self.concat:
                context = torch.cat([c1, c2, c3], dim=2)
            else:
                context = torch.cat([c1, c2, c3], dim=1)
        h = x.type(self.dtype)
        for module in self.input_blocks:
            h = module(h, emb, context=context)
            hs.append(h)
        h = self.middle_block(h, emb, context=context)
        for module in self.output_blocks:
            h = torch.cat([h, hs.pop()], dim=1)
            h = module(h, emb, context=context)
        h = h.type(x.dtype)
        if self.predict_codebook_ids:
            return self.id_predictor(h)
        else:
            return self.out(h)
        
        
        




    