import os
import random
import torch
import torch.nn as nn
import timm
from timm.models.layers import to_2tuple, trunc_normal_, DropPath
from timm.models.vision_transformer import Attention, Mlp, PatchEmbed, Block
from .pos_embed import get_2d_sincos_pos_embed

class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1] * img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x

class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.norm1_a = norm_layer(dim)
        self.norm1_v = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.norm2_a = norm_layer(dim)
        self.norm2_v = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, modality=None):
        if modality == None:
            x = x + self.drop_path(self.attn(self.norm1(x)))
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        elif modality == 'a':
            x = x + self.drop_path(self.attn(self.norm1_a(x)))
            x = x + self.drop_path(self.mlp(self.norm2_a(x)))
        elif modality == 'v':
            x = x + self.drop_path(self.attn(self.norm1_v(x)))
            x = x + self.drop_path(self.mlp(self.norm2_v(x)))
        return x

class CAVMAEST(nn.Module):
    def __init__(self, img_size=224, audio_length=208, patch_size=16, 
                 in_chans=3, embed_dim=768, modality_specific_depth=11, 
                 num_heads=12, contrastive_heads=False, cls_token=True,
                 global_local_losses=True, total_frame=20,
                 decoder_embed_dim=512, decoder_depth=8, 
                 decoder_num_heads=16, mlp_ratio=4., norm_layer=nn.LayerNorm, 
                 norm_pix_loss=False, tr_pos=False):
        super().__init__()
        timm.models.vision_transformer.PatchEmbed = PatchEmbed
        timm.models.vision_transformer.Block = Block
        self.patch_embed_a = PatchEmbed(img_size, patch_size, 1, embed_dim)
        self.patch_embed_v = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        self.audio_length = audio_length
        self.patch_embed_a.num_patches = int(audio_length * 128 / 256)
        self.modality_a = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.modality_v = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed_a = nn.Parameter(torch.zeros(1, self.patch_embed_a.num_patches, embed_dim), requires_grad=tr_pos)
        self.pos_embed_v = nn.Parameter(torch.zeros(1, self.patch_embed_v.num_patches, embed_dim), requires_grad=tr_pos)
        self.blocks_a = nn.ModuleList([Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer) for i in range(modality_specific_depth)])
        self.blocks_v = nn.ModuleList([Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer) for i in range(modality_specific_depth)])
        self.blocks_u = nn.ModuleList([Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer) for i in range(12-modality_specific_depth)])
        self.contrastive_heads = contrastive_heads
        if self.contrastive_heads:
            self.contrastive_head_audio = nn.ModuleList([Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, qk_scale=None, norm_layer=norm_layer) for i in range(2)])
            self.contrastive_head_visual = nn.ModuleList([Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, qk_scale=None, norm_layer=norm_layer) for i in range(2)])
        self.norm_a = norm_layer(embed_dim)
        self.norm_v = norm_layer(embed_dim)
        self.norm = norm_layer(embed_dim)
        # decoder
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        self.decoder_modality_a = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        self.decoder_modality_v = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        self.decoder_pos_embed_a = nn.Parameter(torch.zeros(1, self.patch_embed_a.num_patches, decoder_embed_dim), requires_grad=tr_pos)
        self.decoder_pos_embed_v = nn.Parameter(torch.zeros(1, self.patch_embed_v.num_patches, decoder_embed_dim), requires_grad=tr_pos)
        self.decoder_blocks = nn.ModuleList([Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer) for i in range(decoder_depth)])
        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_pred_a = nn.Linear(decoder_embed_dim, patch_size**2 * 1, bias=True) # decoder to patch
        self.decoder_pred_v = nn.Linear(decoder_embed_dim, patch_size**2 * in_chans, bias=True) # decoder to patch
        self.cls_token = cls_token
        if self.cls_token:
            self.cls_token_a = nn.Parameter(torch.zeros(1, 1, embed_dim))
            self.cls_token_v = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.global_local_losses = global_local_losses
        self.norm_pix_loss = norm_pix_loss
        self.total_frame = total_frame
        self.intermediate_outputs = {}
        self.initialize_weights()

    def register_hooks(self, blocks, block_type):
        def hook_fn(m, i, o):
            self.intermediate_outputs[block_type + str(m)] = o
        for idx, block in enumerate(blocks):
            block.register_forward_hook(hook_fn)

    def initialize_weights(self):
        pos_embed_a = get_2d_sincos_pos_embed(self.pos_embed_a.shape[-1], 8, int(self.patch_embed_a.num_patches/8), cls_token=False)
        self.pos_embed_a.data.copy_(torch.from_numpy(pos_embed_a).float().unsqueeze(0))
        pos_embed_v = get_2d_sincos_pos_embed(self.pos_embed_v.shape[-1], int(self.patch_embed_v.num_patches ** .5), int(self.patch_embed_v.num_patches ** .5), cls_token=False)
        self.pos_embed_v.data.copy_(torch.from_numpy(pos_embed_v).float().unsqueeze(0))
        decoder_pos_embed_a = get_2d_sincos_pos_embed(self.decoder_pos_embed_a.shape[-1], 8, int(self.patch_embed_a.num_patches/8), cls_token=False)
        self.decoder_pos_embed_a.data.copy_(torch.from_numpy(decoder_pos_embed_a).float().unsqueeze(0))
        decoder_pos_embed_v = get_2d_sincos_pos_embed(self.decoder_pos_embed_v.shape[-1], int(self.patch_embed_v.num_patches ** .5), int(self.patch_embed_v.num_patches ** .5), cls_token=False)
        self.decoder_pos_embed_v.data.copy_(torch.from_numpy(decoder_pos_embed_v).float().unsqueeze(0))
        w = self.patch_embed_a.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        w  = self.patch_embed_v.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        if self.cls_token:
            nn.init.normal_(self.cls_token_a, std=.02)
            nn.init.normal_(self.cls_token_v, std=.02)
        torch.nn.init.normal_(self.modality_a, std=.02)
        torch.nn.init.normal_(self.modality_v, std=.02)
        torch.nn.init.normal_(self.decoder_modality_a, std=.02)
        torch.nn.init.normal_(self.decoder_modality_v, std=.02)
        torch.nn.init.normal_(self.mask_token, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_feat(self,a ,v):
        a = a.unsqueeze(1)
        a = a.transpose(2, 3)
        a = self.patch_embed_a(a)
        a = a + self.pos_embed_a
        a = a + self.modality_a
        v = self.patch_embed_v(v)
        v = v + self.pos_embed_v
        v = v +  self.modality_v
        for blk in self.blocks_a:
            a = blk(a)
        for blk in self.blocks_v:
            v = blk(v)
        for blk in self.blocks_u:
            a = blk(a, 'a')
        a = self.norm_a(a)
        for blk in self.blocks_u:
            v = blk(v, 'v')
        v = self.norm_v(v)
        return a,v

    def forward_feat_a(self,a):
        a = a.unsqueeze(1)
        a = a.transpose(2, 3)
        a = self.patch_embed_a(a)
        a = a + self.pos_embed_a
        a = a + self.modality_a
        for blk in self.blocks_a:
            a = blk(a)
        for blk in self.blocks_u:
            a = blk(a, 'a')
        a = self.norm_a(a)
        return a

    def forward_feat_v(self,v):
        v = self.patch_embed_v(v)
        v = v + self.pos_embed_v
        v = v +  self.modality_v
        for blk in self.blocks_v:
            v = blk(v)
        for blk in self.blocks_u:
            v = blk(v, 'v')
        v = self.norm_v(v)
        return v





        


