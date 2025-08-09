import torch
import torch.nn as nn
import numpy as np
import math
from timm.models.vision_transformer import PatchEmbed, Mlp
from timm.models.layers import trunc_normal_
import math
from torch.cuda.amp import autocast
from einops import rearrange 


def prob_mask_like(shape, prob, device):
    if prob == 1:
        return torch.ones(shape, device = device, dtype = torch.bool)
    elif prob == 0:
        return torch.zeros(shape, device = device, dtype = torch.bool)
    else:
        return torch.zeros(shape, device = device).float().uniform_(0, 1) < prob

def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., num_patches=None):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.rel_pos_bias = RelativePositionBias(
            window_size=[int(num_patches**0.5), int(num_patches**0.5)], num_heads=num_heads)

    def get_masked_rel_bias(self, B, ids_keep):
        rel_pos_bias = self.rel_pos_bias()
        rel_pos_bias = rel_pos_bias.unsqueeze(dim=0).repeat(B, 1, 1, 1)

        rel_pos_bias_masked = torch.gather(
            rel_pos_bias, dim=2, index=ids_keep.unsqueeze(dim=1).unsqueeze(dim=-1).repeat(1, rel_pos_bias.shape[1], 1, rel_pos_bias.shape[-1]))
        rel_pos_bias_masked = torch.gather(
            rel_pos_bias_masked, dim=3, index=ids_keep.unsqueeze(dim=1).unsqueeze(dim=2).repeat(1, rel_pos_bias.shape[1], ids_keep.shape[1], 1))
        return rel_pos_bias_masked

    def forward(self, x, ids_keep=None):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C //
                                  self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        if ids_keep is not None:
            rp_bias = self.get_masked_rel_bias(B, ids_keep)
        else:
            rp_bias = self.rel_pos_bias()
        attn += rp_bias
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class RelativePositionBias(nn.Module):
    # https://github.com/microsoft/unilm/blob/master/beit/modeling_finetune.py
    def __init__(self, window_size, num_heads):
        super().__init__()
        self.window_size = window_size
        self.num_relative_distance = (
            2 * window_size[0] - 1) * (2 * window_size[1] - 1) + 3
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros(self.num_relative_distance, num_heads))

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(window_size[0])
        coords_w = torch.arange(window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - \
            coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(
            1, 2, 0).contiguous() 
        relative_coords[:, :, 0] += window_size[0] - 1
        relative_coords[:, :, 1] += window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * window_size[1] - 1
        relative_position_index = \
            torch.zeros(
                size=(window_size[0] * window_size[1],) * 2, dtype=relative_coords.dtype)
        relative_position_index = relative_coords.sum(-1)

        self.register_buffer("relative_position_index",
                             relative_position_index)

        trunc_normal_(self.relative_position_bias_table, std=.02)

    def forward(self):
        relative_position_bias = \
            self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
                self.window_size[0] * self.window_size[1],
                self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        # nH, Wh*Ww, Wh*Ww
        return relative_position_bias.permute(2, 0, 1).contiguous()
    

#################################################################################
#               Embedding Layers for Timesteps and Class Labels                 #
#################################################################################


class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """

    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0,
                                                 end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat(
                [embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class LabelEmbedder(nn.Module):
    """
    Embeds class labels into vector representations. Also handles label dropout for classifier-free guidance.
    """

    def __init__(self, num_classes, hidden_size, dropout_prob):
        super().__init__()
        use_cfg_embedding = dropout_prob > 0
        self.embedding_table = nn.Embedding(
            num_classes + use_cfg_embedding, hidden_size)
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob

    def token_drop(self, labels, force_drop_ids=None):
        """
        Drops labels to enable classifier-free guidance.
        """
        if force_drop_ids is None:
            drop_ids = torch.rand(labels.shape[0]) < self.dropout_prob
        else:
            drop_ids = force_drop_ids == 1

        labels = torch.where(drop_ids.to(labels.device),
                             self.num_classes, labels)
        return labels

    def forward(self, labels, train, force_drop_ids=None):
        use_dropout = self.dropout_prob > 0
        if (train and use_dropout) or (force_drop_ids is not None):
            labels = self.token_drop(labels, force_drop_ids)
        embeddings = self.embedding_table(labels)
        return embeddings


#################################################################################
#                                 Core MDT Model                                #
#################################################################################

class MDTBlock_CA(nn.Module):
    """
    A MDT block with Cross Attention conditioning.
    """

    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, num_patches=None, skip=False, **block_kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(
            hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(
            hidden_size, num_heads=num_heads, qkv_bias=True, num_patches=num_patches, **block_kwargs)
        self.norm2 = nn.LayerNorm(
            hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        def approx_gelu(): return nn.GELU(approximate="tanh")
        self.mlp = Mlp(in_features=hidden_size,
                       hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)
        self.skip_linear = nn.Linear(2 * hidden_size, hidden_size) if skip else None
        self.cross_norm = nn.LayerNorm(
            hidden_size, elementwise_affine=False, eps=1e-6)
        self.cross_attn = Attention_Cross(
            hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs)

    def forward(self, x, c, skip=None, ids_keep=None):
        if self.skip_linear is not None:
            x = self.skip_linear(torch.cat([x, skip], dim=-1))
        x = x + self.attn(self.norm1(x), ids_keep=ids_keep)
        x = x + self.cross_attn(self.cross_norm(x), context=c)
        x = x + self.mlp(self.norm2(x))
        return x

class Attention_Cross(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.head_dim = head_dim

        self.to_q = nn.Linear(dim, dim, bias=qkv_bias)
        self.to_kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, context=None):
        B, N, C = x.shape
        q, k, v = (self.to_q(x), *self.to_kv(context).chunk(2, dim = -1))
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.num_heads), (q, k, v))

        attn = (q @ k.transpose(-2, -1)) * self.scale

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class FinalLayer(nn.Module):
    """
    The final layer of MDT-IVTON.
    """

    def __init__(self, hidden_size, num_heads, patch_size, out_channels, **block_kwargs):
        super().__init__()
        self.norm_final = nn.LayerNorm(
            hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(
            hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.cross_attn = Attention_Cross(
            hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs)

    def forward(self, x, c):
        x = x + self.cross_attn(self.norm_final(x), context=c)
        x = self.linear(x)
        return x


class ITAFA(nn.Module):
    """
    Image-Time Adaptive Feature Aggregator (ITAFA).

    Args:
        vit_cloth (torch.Tensor): 
            ViT-encoded garment features from all hidden layers.
            Shape: (batch_size, num_hidden_layers, num_patches, embedding_dim)
            Example: (B, L, 257, 1280)
        t (torch.Tensor): 
            Diffusion timestep embeddings for each sample.
            Shape: (batch_size, timestep_dim)

    Returns:
        torch.Tensor:
            Aggregated garment feature tensor after applying
            timestep- and complexity-adaptive weights.
            Shape: (batch_size, num_patches, embedding_dim)
            Example: (B, 257, 1280)

    """
    def __init__(self, num_hidden_layers, timestep_dim):
        super(ITAFA, self).__init__()
        
        # Separate linear layers for timestep and complexity components
        self.timestep_projector = nn.Linear(timestep_dim, num_hidden_layers)
        self.complexity_projector = nn.Linear(3, num_hidden_layers)  # 3 components: sparsity, variance, gradient magnitude

        # Learnable weight to balance timestep and complexity contributions
        self.alpha = nn.Parameter(torch.tensor(0.5, device=self.timestep_projector.weight.device))  # Initialize alpha at 0.5, learnable weight

    def feature_sparsity(self, feature_embeddings, threshold=0.01):
        """Measure sparsity of the feature_embeddings per batch item."""
        sparsity = torch.mean((torch.abs(feature_embeddings) < threshold).float(), dim=[1, 2, 3])  # Shape: [batch_size]
        return sparsity

    def feature_variance(self, feature_embeddings):
        """Compute variance of activations per batch item."""
        return torch.var(feature_embeddings, dim=[1, 2, 3])  # Shape: [batch_size]

    def feature_gradient_magnitude(self, feature_embeddings):
        """Compute the gradient magnitude of feature_embeddings) as a measure of complexity.
        
        This function computes gradients along patches and embedding dimension.
        """
        feature_embeddings = feature_embeddings.to(self.timestep_projector.weight.device)
        # Gradient along the patches+position dimension (257)
        dx = feature_embeddings[:, :, 1:, :] - feature_embeddings[:, :, :-1, :].to(feature_embeddings.device)
        
        # Gradient along the embedding dimension (1280)
        dy = feature_embeddings[:, :, :, 1:] - feature_embeddings[:, :, :, :-1].to(feature_embeddings.device)

        # Compute magnitude of gradients, ensure matching dimensions
        gradient_magnitude = torch.sqrt(dx[:, :, :, :-1]**2 + dy[:, :, :-1, :]**2)  # Shape: [batch_size, num_hidden_layers, patches-1, embedding-1]

        # Average over patches and embedding to get a single score per batch item
        avg_gradient_magnitude = gradient_magnitude.mean(dim=[2, 3])  # Shape: [batch_size, num_hidden_layers]
        
        # Return the mean gradient magnitude across hidden layers
        return avg_gradient_magnitude.mean(dim=1)  # Shape: [batch_size]


    def compute_complexity(self, feature_embeddings):
        """Compute overall complexity based on sparsity, variance, and gradient magnitude."""
        feature_embeddings = feature_embeddings.to(self.timestep_projector.weight.device)
        
        sparsity_score = self.feature_sparsity(feature_embeddings)  # Shape: [batch_size]
        variance_score = self.feature_variance(feature_embeddings)  # Shape: [batch_size]
        gradient_score = self.feature_gradient_magnitude(feature_embeddings)  # Shape: [batch_size]

        # Concatenate the complexity components into a single tensor for each batch
        complexity_components = torch.stack([sparsity_score, variance_score, gradient_score], dim=1).to(self.timestep_projector.weight.device)  # Shape: [batch_size, 3]
        
        return complexity_components  # Shape: [batch_size, 3]

    def forward(self, vit_cloth, t):
        vit_cloth = vit_cloth.to(self.timestep_projector.weight.device)
        t = t.to(self.timestep_projector.weight.device)

        # Compute the garment complexity score for each image in the batch
        complexity_components = self.compute_complexity(vit_cloth)  # Shape: [batch_size, 3]

        # Project the timestep embedding to the hidden layers
        projected_timestep = self.timestep_projector(t)  # Shape: [batch_size, num_hidden_layers]

        # Project the complexity components to the hidden layers
        projected_complexity = self.complexity_projector(complexity_components)  # Shape: [batch_size, num_hidden_layers]
        
        # Combine the projected timestep and complexity components
        combined_weights = self.alpha * projected_timestep + (1 - self.alpha) * projected_complexity  # Shape: [batch_size, num_hidden_layers]
        
        # Normalize weights using softmax to ensure they sum to 1
        normalized_weights = torch.softmax(combined_weights, dim=-1)  # Shape: [batch_size, num_hidden_layers]

        # Apply the weights to the hidden layer outputs (vit_cloth)
        vit_cloth = torch.sum(normalized_weights.view(vit_cloth.shape[0], vit_cloth.shape[1], 1, 1) * vit_cloth, dim=1)  # Shape: [batch_size, 257, 1280]
        
        return vit_cloth



class MDT_IVTON(nn.Module):
    """
    Masked Diffusion Transformer for Image-based Virtual Try-on (MDT-IVTON).
    """

    def __init__(
        self,
        input_size=32,
        patch_size=2,
        in_channels=4,
        hidden_size=1152,
        depth=28,
        num_heads=16,
        mlp_ratio=4.0,
        learn_sigma=True,
        mask_ratio=None,
        decode_layer=4,
    ):
        super().__init__()
        self.learn_sigma = learn_sigma
        self.in_channels = in_channels
        self.out_channels = in_channels * 2 if learn_sigma else in_channels
        self.patch_size = patch_size
        self.num_heads = num_heads
        decode_layer = int(decode_layer)

        self.x_embedder = PatchEmbed(
            input_size, patch_size, in_channels * 4, hidden_size, bias=True) # in_channels *4 due to concatenating 4 latents. z_t, A, P, M_x
        self.t_embedder = TimestepEmbedder(hidden_size)
        num_patches = self.x_embedder.num_patches
        self.pos_embed = nn.Parameter(torch.zeros(
            1, num_patches, hidden_size), requires_grad=True)
        self.decoder_pos_embed = nn.Parameter(torch.zeros(
            1, num_patches, hidden_size), requires_grad=True)

        half_depth = (depth - decode_layer)//2
        self.half_depth=half_depth
        
        self.en_inblocks = nn.ModuleList([
            MDTBlock_CA(hidden_size, num_heads, mlp_ratio=mlp_ratio, num_patches=num_patches) for _ in range(half_depth)
        ])
        self.en_outblocks = nn.ModuleList([
            MDTBlock_CA(hidden_size, num_heads, mlp_ratio=mlp_ratio, num_patches=num_patches, skip=True) for _ in range(half_depth)
        ])
        self.de_blocks = nn.ModuleList([
            MDTBlock_CA(hidden_size, num_heads, mlp_ratio=mlp_ratio, num_patches=num_patches, skip=True) for i in range(decode_layer)
        ])
        self.sideblocks = nn.ModuleList([
            MDTBlock_CA(hidden_size, num_heads, mlp_ratio=mlp_ratio, num_patches=num_patches) for _ in range(1)
        ])
        self.final_layer = FinalLayer(
            hidden_size, num_heads, patch_size, self.out_channels)

        if mask_ratio is not None:
            self.mask_token = nn.Parameter(torch.zeros(1, 1, hidden_size))
            self.mask_ratio = float(mask_ratio)
            self.decode_layer = int(decode_layer)
        else:
            self.mask_token = nn.Parameter(torch.zeros(
                1, 1, hidden_size), requires_grad=False)
            self.mask_ratio = None
            self.decode_layer = int(decode_layer)
        print("mask ratio:", self.mask_ratio, "decode_layer:", self.decode_layer)
      
        self.mlp_image_encoder = nn.Sequential(
            nn.Linear(in_features=1536, out_features=hidden_size, bias=True),
            nn.GELU(),
            nn.LayerNorm(hidden_size, eps=1e-12)
        )
        
        self.itafa = ITAFA(num_hidden_layers=40, timestep_dim=hidden_size) #num_hidden_layers=40 for DINOv2g14

        self.initialize_weights()

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Initialize pos_embed by sin-cos embedding:
        pos_embed = get_2d_sincos_pos_embed(
            self.pos_embed.shape[-1], int(self.x_embedder.num_patches ** 0.5))
        self.pos_embed.data.copy_(
            torch.from_numpy(pos_embed).float().unsqueeze(0))

        decoder_pos_embed = get_2d_sincos_pos_embed(
            self.decoder_pos_embed.shape[-1], int(self.x_embedder.num_patches ** 0.5))
        self.decoder_pos_embed.data.copy_(
            torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))

        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.proj.bias, 0)

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

        if self.mask_ratio is not None:
            torch.nn.init.normal_(self.mask_token, std=.02)

        # Initialize dino embedding MLP:
        nn.init.normal_(self.mlp_image_encoder[0].weight, std=0.02)

        # Initialize weights for ITAFA
        nn.init.xavier_uniform_(self.itafa.timestep_projector.weight)
        nn.init.constant_(self.itafa.timestep_projector.bias, 0)
        nn.init.xavier_uniform_(self.itafa.complexity_projector.weight)
        nn.init.constant_(self.itafa.complexity_projector.bias, 0)
        nn.init.normal_(self.itafa.alpha, mean=0.5, std=0.1)  

    def unpatchify(self, x):
        """
        x: (N, T, patch_size**2 * C)
        imgs: (N, H, W, C)
        """
        c = self.out_channels
        p = self.x_embedder.patch_size[0]
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, h * p))
        return imgs

    def random_masking(self, x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))

        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]

        # sort noise for each sample
        # ascend: small is keep, large is remove
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(
            x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore, ids_keep

    def forward_side_interpolater(self, x, c, mask, ids_restore):
        # append mask tokens to sequence
        mask_tokens = self.mask_token.repeat(
            x.shape[0], ids_restore.shape[1] - x.shape[1], 1)
        x_ = torch.cat([x, mask_tokens], dim=1)
        x = torch.gather(
            x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle
        
        # add pos embed
        x = x + self.decoder_pos_embed

        # pass to the basic block
        x_before = x
        for sideblock in self.sideblocks:
            x = sideblock(x, c, ids_keep=None)
                
        # masked shortcut
        mask = mask.unsqueeze(dim=-1)
        x = x*mask + (1-mask)*x_before

        return x

    def forward(self, x, t, enable_mask=False, cond=None, prob=0.1): 
        """
        Forward pass of MDT-IVTON.
        x: (N, C, H, W) tensor of spatial inputs (images or latent representations of images)
        t: (N,) tensor of diffusion timesteps
        enable_mask: Use mask latent modeling
        cond: dictionary of tensors containing conditioning inputs (VAE/ViT encoded images)
        """
        with autocast(dtype=torch.bfloat16):
            x = torch.cat([x, cond['agn'], cond['image_densepose'], cond['agn_mask']], 1) # x is [b, 4, 64, 64]
            x = self.x_embedder(x) + self.pos_embed  # (N, T, D), where T = H * W / patch_size ** 2 , [b, 1024, 1152]
            t = self.t_embedder(t)                   # (N, 1152) timestep embedding

            vit_cloth = self.itafa(cond['vit_cloth'], t) 
            vit_cloth = vit_cloth[:, 1:, :]
            vit_cloth_sr = self.itafa(cond['vit_cloth_sr'], t)
            vit_cloth_sr = vit_cloth_sr[:, 1:, :]

            f_g = self.mlp_image_encoder(vit_cloth) 
            f_s = self.mlp_image_encoder(vit_cloth_sr) 
            
            y = torch.cat([f_g, f_s], 1) 
            
            cond_mask = prob_mask_like((y.shape[0],), prob = 1-prob, device = x.device) # classifier free guidance
            y = cond_mask[:, None, None]*y
            c = t.unsqueeze(1) + y 
            masked_stage = False

            input_skip = x

            masked_stage = False
            skips = []
            # masking op for training
            if self.mask_ratio is not None and enable_mask:
                # masking: length -> length * mask_ratio
                rand_mask_ratio = torch.rand(1, device=x.device)  # noise in [0, 1]
                rand_mask_ratio = rand_mask_ratio * 0.2 + self.mask_ratio 
                x, mask, ids_restore, ids_keep = self.random_masking(
                    x, rand_mask_ratio)
                masked_stage = True


            for block in self.en_inblocks:
                if masked_stage:
                    x = block(x, c, ids_keep=ids_keep)
                else:
                    x = block(x, c, ids_keep=None)
                skips.append(x)

            for block in self.en_outblocks:
                if masked_stage:
                    x = block(x, c, skip=skips.pop(), ids_keep=ids_keep)
                else:
                    x = block(x, c, skip=skips.pop(), ids_keep=None)

            if self.mask_ratio is not None and enable_mask:
                x = self.forward_side_interpolater(x, c, mask, ids_restore)
                masked_stage = False
            else:
                # add pos embed
                x = x + self.decoder_pos_embed

            for i in range(len(self.de_blocks)):
                block = self.de_blocks[i]
                this_skip = input_skip

                x = block(x, c, skip=this_skip, ids_keep=None)
            x = self.final_layer(x, c) # [b, 1024, 32]
            x = self.unpatchify(x)  # (N, out_channels, H, W), [b, 8, 64, 64]
            return x


    def forward_with_cfg(self, x, t, cfg_scale=None, diffusion_steps=1000, scale_pow=4.0, cond=None):
        """
        Forward pass of MDT, but also batches the unconditional forward pass for classifier-free guidance.
        """
        # https://github.com/openai/glide-text2im/blob/main/notebooks/text2im.ipynb
        ### MADPIT inference with cfg
        if cfg_scale is not None:
            uncond_eps = self.forward(x, t, False, cond, 1)
            cond_eps = self.forward(x, t, False, cond, 0)
            
            scale_step = (
                1-torch.cos(((1-t/diffusion_steps)**scale_pow)*math.pi))*1/2 # power-cos scaling 
            real_cfg_scale = (cfg_scale-1)*scale_step + 1
            real_cfg_scale = real_cfg_scale[: len(x)].view(-1, 1, 1, 1)
            
            eps = uncond_eps + real_cfg_scale * (cond_eps - uncond_eps)
            return eps
        else:
            pass

#################################################################################
#                   Sine/Cosine Positional Embedding Functions                  #
#################################################################################
# https://github.com/facebookresearch/mae/blob/main/util/pos_embed.py


def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False, extra_tokens=0):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate(
            [np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(
        embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(
        embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1)  # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


#################################################################################
#                               MDT-IVTON Configs                               #
#################################################################################

def MDT_IVTON_XL(**kwargs):
    return MDT_IVTON(depth=28, hidden_size=1152, patch_size=2, num_heads=16, **kwargs)
    
def MDT_IVTON_L(**kwargs):
    return MDT_IVTON(depth=24, hidden_size=1024, patch_size=2, num_heads=16, **kwargs)

def MDT_IVTON_B(**kwargs):
    return MDT_IVTON(depth=12, hidden_size=768, patch_size=2, num_heads=12, **kwargs)

def MDT_IVTON_S(**kwargs):
    return MDT_IVTON(depth=12, hidden_size=384, patch_size=2, num_heads=6, **kwargs)

def MDT_IVTON_Tiny(**kwargs):
    return MDT_IVTON(depth=12, hidden_size=192, patch_size=2, num_heads=3, **kwargs)

def MDT_IVTON_SuperTiny(**kwargs):
    return MDT_IVTON(depth=6, hidden_size=192, patch_size=2, num_heads=3, **kwargs)
