from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from einops.layers.torch import Rearrange
from lib.rotary_embeddings import apply_rope
from controlfoley.neural_blocks import MLP, ChannelLastConv1d, ConvMLP


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """
    Apply modulation to input tensor using shift and scale parameters

    Args:
        x: Input tensor
        shift: Shift parameter
        scale: Scale parameter

    Returns:
        Modulated tensor: x * (1 + scale) + shift
    """
    return x * (1 + scale) + shift


def attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """
    Compute scaled dot-product attention

    Args:
        q: Query tensor of shape (B, H, N, D)
        k: Key tensor of shape (B, H, N, D)
        v: Value tensor of shape (B, H, N, D)

    Returns:
        Output tensor of shape (B, N, H*D)
    """
    # Ensure tensors are contiguous for CUDNN compatibility
    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()

    # Compute scaled dot-product attention
    out = F.scaled_dot_product_attention(q, k, v)

    # Rearrange from (B, H, N, D) to (B, N, H*D)
    out = rearrange(out, 'b h n d -> b n (h d)').contiguous()

    return out


class SelfAttention(nn.Module):
    """Self-attention module with rotary position embeddings"""

    def __init__(self, dim: int, nheads: int):
        """
        Initialize self-attention module

        Args:
            dim: Dimension of input features
            nheads: Number of attention heads
        """
        super().__init__()

        # Validate inputs
        if dim <= 0:
            raise ValueError(f"dim must be positive, got {dim}")
        if nheads <= 0:
            raise ValueError(f"nheads must be positive, got {nheads}")
        if dim % nheads != 0:
            raise ValueError(f"dim ({dim}) must be divisible by nheads ({nheads})")

        self.dim = dim
        self.nheads = nheads
        self.head_dim = dim // nheads

        # Initialize components
        self._initialize_components()

    def _initialize_components(self):
        """Initialize all components of the self-attention module"""
        # QKV projection layer
        self.qkv = nn.Linear(self.dim, self.dim * 3, bias=True)

        # Normalization layers for query and key
        self.q_norm = nn.RMSNorm(self.head_dim)
        self.k_norm = nn.RMSNorm(self.head_dim)

        # Rearrange layer for splitting QKV into heads
        self.split_into_heads = Rearrange(
            'b n (h d j) -> b h n d j',
            h=self.nheads,
            d=self.head_dim,
            j=3
        )

    def _project_qkv(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Project input to query, key, and value

        Args:
            x: Input tensor of shape (B, N, D)

        Returns:
            Tuple of (query, key, value) tensors each of shape (B, H, N, D_head)
        """
        # Project to QKV
        qkv = self.qkv(x)

        # Split into Q, K, V and reshape for multi-head attention
        qkv_split = self.split_into_heads(qkv)

        # Separate Q, K, V
        q, k, v = qkv_split.chunk(3, dim=-1)

        # Remove last dimension
        q = q.squeeze(-1)
        k = k.squeeze(-1)
        v = v.squeeze(-1)

        return q, k, v

    def _normalize_qk(self, q: torch.Tensor, k: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Apply RMS normalization to query and key

        Args:
            q: Query tensor
            k: Key tensor

        Returns:
            Normalized query and key tensors
        """
        q_normalized = self.q_norm(q)
        k_normalized = self.k_norm(k)

        return q_normalized, k_normalized

    def _apply_rotary_embeddings(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        rot: Optional[torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Apply rotary position embeddings to query and key

        Args:
            q: Query tensor
            k: Key tensor
            rot: Rotary embedding tensor (optional)

        Returns:
            Query and key tensors with rotary embeddings applied
        """
        if rot is not None:
            q_rotated = apply_rope(q, rot)
            k_rotated = apply_rope(k, rot)
            return q_rotated, k_rotated

        return q, k

    def pre_attention(
        self,
        x: torch.Tensor,
        rot: Optional[torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Pre-attention processing: project, normalize, and apply rotary embeddings

        Args:
            x: Input tensor of shape (B, N, D)
            rot: Rotary embedding tensor (optional)

        Returns:
            Tuple of (query, key, value) tensors
        """
        # Project to QKV
        q, k, v = self._project_qkv(x)

        # Normalize Q and K
        q, k = self._normalize_qk(q, k)

        # Apply rotary embeddings
        q, k = self._apply_rotary_embeddings(q, k, rot)

        return q, k, v

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of self-attention

        Args:
            x: Input tensor of shape (B, N, D)

        Returns:
            Output tensor of shape (B, N, D)
        """
        # Pre-attention processing
        q, k, v = self.pre_attention(x, rot=None)

        # Compute attention
        out = attention(q, k, v)

        return out


class MMDitSingleBlock(nn.Module):
    """Single block of MMDiT with optional pre-only mode"""

    def __init__(
        self,
        dim: int,
        nhead: int,
        mlp_ratio: float = 4.0,
        pre_only: bool = False,
        kernel_size: int = 7,
        padding: int = 3
    ):
        """
        Initialize MMDit single block

        Args:
            dim: Dimension of input features
            nhead: Number of attention heads
            mlp_ratio: Ratio for MLP hidden dimension
            pre_only: If True, only pre-attention components are created
            kernel_size: Kernel size for convolution layers
            padding: Padding for convolution layers
        """
        super().__init__()

        # Validate inputs
        if dim <= 0:
            raise ValueError(f"dim must be positive, got {dim}")
        if nhead <= 0:
            raise ValueError(f"nhead must be positive, got {nhead}")
        if mlp_ratio <= 0:
            raise ValueError(f"mlp_ratio must be positive, got {mlp_ratio}")

        self.pre_only = pre_only
        self.dim = dim
        self.nhead = nhead
        self.mlp_ratio = mlp_ratio
        self.kernel_size = kernel_size
        self.padding = padding

        # Initialize components
        self._initialize_components()

    def _initialize_components(self):
        """Initialize all components based on pre_only flag"""
        # Normalization layer
        self.norm1 = nn.LayerNorm(self.dim, elementwise_affine=False)

        # Self-attention module
        self.attn = SelfAttention(self.dim, self.nhead)

        # Modulation layer
        self._initialize_modulation()

        # Additional components for full mode
        if not self.pre_only:
            self._initialize_full_mode_components()

    def _initialize_modulation(self):
        """Initialize modulation layer"""
        if self.pre_only:
            # Pre-only mode: shift and scale for attention only
            self.adaLN_modulation = nn.Sequential(
                nn.SiLU(),
                nn.Linear(self.dim, 2 * self.dim, bias=True)
            )
        else:
            # Full mode: shift, scale, and gate for attention and MLP
            self.adaLN_modulation = nn.Sequential(
                nn.SiLU(),
                nn.Linear(self.dim, 6 * self.dim, bias=True)
            )

    def _initialize_full_mode_components(self):
        """Initialize components for full mode (not pre-only)"""
        # Linear or convolutional layer for attention output
        if self.kernel_size == 1:
            self.linear1 = nn.Linear(self.dim, self.dim)
        else:
            self.linear1 = ChannelLastConv1d(
                self.dim,
                self.dim,
                kernel_size=self.kernel_size,
                padding=self.padding
            )

        # Second normalization layer
        self.norm2 = nn.LayerNorm(self.dim, elementwise_affine=False)

        # MLP layer
        if self.kernel_size == 1:
            self.ffn = MLP(self.dim, int(self.dim * self.mlp_ratio))
        else:
            self.ffn = ConvMLP(
                self.dim,
                int(self.dim * self.mlp_ratio),
                kernel_size=self.kernel_size,
                padding=self.padding
            )

    def _compute_modulation(self, c: torch.Tensor) -> tuple:
        """
        Compute modulation parameters from condition

        Args:
            c: Condition tensor of shape (B, D)

        Returns:
            Tuple of modulation parameters
        """
        modulation = self.adaLN_modulation(c)

        if self.pre_only:
            # Pre-only mode: shift and scale for attention
            shift_msa, scale_msa = modulation.chunk(2, dim=-1)
            gate_msa = shift_mlp = scale_mlp = gate_mlp = None
        else:
            # Full mode: all modulation parameters
            (shift_msa, scale_msa, gate_msa,
             shift_mlp, scale_mlp, gate_mlp) = modulation.chunk(6, dim=-1)

        return (shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp)

    def pre_attention(
        self,
        x: torch.Tensor,
        c: torch.Tensor,
        rot: Optional[torch.Tensor]
    ) -> tuple[tuple[torch.Tensor, torch.Tensor, torch.Tensor], tuple]:
        """
        Pre-attention processing

        Args:
            x: Input tensor of shape (B, N, D)
            c: Condition tensor of shape (B, D)
            rot: Rotary embedding tensor (optional)

        Returns:
            Tuple of (QKV tensors, modulation parameters)
        """
        # Compute modulation parameters
        mod_params = self._compute_modulation(c)

        if self.pre_only:
            shift_msa, scale_msa = mod_params[0], mod_params[1]
            gate_msa = shift_mlp = scale_mlp = gate_mlp = None
        else:
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = mod_params

        # Apply modulation to input
        x_modulated = modulate(self.norm1(x), shift_msa, scale_msa)

        # Pre-attention processing
        q, k, v = self.attn.pre_attention(x_modulated, rot)

        return (q, k, v), (gate_msa, shift_mlp, scale_mlp, gate_mlp)

    def post_attention(
        self,
        x: torch.Tensor,
        attn_out: torch.Tensor,
        c: tuple[torch.Tensor]
    ) -> torch.Tensor:
        """
        Post-attention processing

        Args:
            x: Input tensor
            attn_out: Attention output
            c: Modulation parameters

        Returns:
            Output tensor
        """
        # Return early if pre-only mode
        if self.pre_only:
            return x

        # Unpack modulation parameters
        gate_msa, shift_mlp, scale_mlp, gate_mlp = c

        # Apply attention output with gate
        x = x + self.linear1(attn_out) * gate_msa

        # Apply modulation and MLP
        r = modulate(self.norm2(x), shift_mlp, scale_mlp)
        x = x + self.ffn(r) * gate_mlp

        return x

    def forward(
        self,
        x: torch.Tensor,
        cond: torch.Tensor,
        rot: Optional[torch.Tensor]
    ) -> torch.Tensor:
        """
        Forward pass of MMDit single block

        Args:
            x: Input tensor of shape (B, N, D)
            cond: Condition tensor of shape (B, D)
            rot: Rotary embedding tensor (optional)

        Returns:
            Output tensor of shape (B, N, D)
        """
        # Pre-attention processing
        x_qkv, x_conditions = self.pre_attention(x, cond, rot)

        # Compute attention
        attn_out = attention(*x_qkv)

        # Post-attention processing
        x = self.post_attention(x, attn_out, x_conditions)

        return x


class JointBlock(nn.Module):
    """Joint block that processes multiple modalities together"""

    def __init__(
        self,
        dim: int,
        nhead: int,
        mlp_ratio: float = 4.0,
        pre_only: bool = False
    ):
        """
        Initialize joint block

        Args:
            dim: Dimension of input features
            nhead: Number of attention heads
            mlp_ratio: Ratio for MLP hidden dimension
            pre_only: If True, clip, text, and audio blocks are pre-only
        """
        super().__init__()

        self.pre_only = pre_only
        self.dim = dim
        self.nhead = nhead
        self.mlp_ratio = mlp_ratio

        # Initialize modality-specific blocks
        self._initialize_blocks()

    def _initialize_blocks(self):
        """Initialize modality-specific MMDit blocks"""
        # Latent block (always full mode)
        self.latent_block = MMDitSingleBlock(
            self.dim,
            self.nhead,
            self.mlp_ratio,
            pre_only=False,
            kernel_size=3,
            padding=1
        )

        # CLIP block
        self.clip_block = MMDitSingleBlock(
            self.dim,
            self.nhead,
            self.mlp_ratio,
            pre_only=self.pre_only,
            kernel_size=3,
            padding=1
        )

        # Text block (kernel_size=1)
        self.text_block = MMDitSingleBlock(
            self.dim,
            self.nhead,
            self.mlp_ratio,
            pre_only=self.pre_only,
            kernel_size=1
        )

        # Audio block (kernel_size=1)
        self.audio_block = MMDitSingleBlock(
            self.dim,
            self.nhead,
            self.mlp_ratio,
            pre_only=self.pre_only,
            kernel_size=1
        )

    def _compute_pre_attention_for_all_modalities(
        self,
        latent: torch.Tensor,
        clip_f: torch.Tensor,
        audio_f: torch.Tensor,
        text_f: torch.Tensor,
        global_c: torch.Tensor,
        extended_c: torch.Tensor,
        latent_rot: torch.Tensor,
        clip_rot: torch.Tensor
    ) -> dict:
        """
        Compute pre-attention outputs for all modalities

        Args:
            latent: Latent tensor
            clip_f: CLIP features
            audio_f: Audio features
            text_f: Text features
            global_c: Global condition
            extended_c: Extended condition
            latent_rot: Latent rotary embeddings
            clip_rot: CLIP rotary embeddings

        Returns:
            Dictionary containing QKV and modulation for each modality
        """
        # Pre-attention for latent
        x_qkv, x_mod = self.latent_block.pre_attention(latent, extended_c, latent_rot)

        # Pre-attention for CLIP
        c_qkv, c_mod = self.clip_block.pre_attention(clip_f, global_c, clip_rot)

        # Pre-attention for text (no rotary embeddings)
        t_qkv, t_mod = self.text_block.pre_attention(text_f, global_c, rot=None)

        # Pre-attention for audio (no rotary embeddings)
        a_qkv, a_mod = self.audio_block.pre_attention(audio_f, global_c, rot=None)

        return {
            'x_qkv': x_qkv,
            'c_qkv': c_qkv,
            't_qkv': t_qkv,
            'a_qkv': a_qkv,
            'x_mod': x_mod,
            'c_mod': c_mod,
            't_mod': t_mod,
            'a_mod': a_mod
        }

    def _concatenate_qkv(self, pre_attention_outputs: dict) -> list:
        """
        Concatenate QKV from all modalities

        Args:
            pre_attention_outputs: Dictionary of pre-attention outputs

        Returns:
            List of concatenated QKV tensors
        """
        x_qkv = pre_attention_outputs['x_qkv']
        c_qkv = pre_attention_outputs['c_qkv']
        t_qkv = pre_attention_outputs['t_qkv']
        a_qkv = pre_attention_outputs['a_qkv']

        # Concatenate Q, K, V from all modalities
        joint_qkv = [
            torch.cat([x_qkv[i], a_qkv[i], c_qkv[i], t_qkv[i]], dim=2)
            for i in range(3)
        ]

        return joint_qkv

    def _split_attention_output(
        self,
        attn_out: torch.Tensor,
        latent_len: int,
        audio_len: int,
        clip_len: int,
        text_len: int
    ) -> dict:
        """
        Split attention output by modality

        Args:
            attn_out: Joint attention output
            latent_len: Length of latent sequence
            audio_len: Length of audio sequence
            clip_len: Length of CLIP sequence
            text_len: Length of text sequence

        Returns:
            Dictionary of attention outputs for each modality
        """
        # Split attention output by modality
        x_attn_out = attn_out[:, :latent_len]
        a_attn_out = attn_out[:, latent_len:latent_len + audio_len]
        c_attn_out = attn_out[:, latent_len + audio_len:latent_len + audio_len + clip_len]
        t_attn_out = attn_out[:, latent_len + audio_len + clip_len:]

        return {
            'x_attn_out': x_attn_out,
            'a_attn_out': a_attn_out,
            'c_attn_out': c_attn_out,
            't_attn_out': t_attn_out
        }

    def _apply_post_attention(
        self,
        latent: torch.Tensor,
        clip_f: torch.Tensor,
        audio_f: torch.Tensor,
        text_f: torch.Tensor,
        attn_outputs: dict,
        pre_attention_outputs: dict
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Apply post-attention processing to all modalities

        Args:
            latent: Latent tensor
            clip_f: CLIP features
            audio_f: Audio features
            text_f: Text features
            attn_outputs: Dictionary of attention outputs
            pre_attention_outputs: Dictionary of pre-attention outputs

        Returns:
            Tuple of updated tensors for each modality
        """
        # Post-attention for latent
        latent = self.latent_block.post_attention(
            latent,
            attn_outputs['x_attn_out'],
            pre_attention_outputs['x_mod']
        )

        # Post-attention for other modalities (if not pre-only)
        if not self.pre_only:
            clip_f = self.clip_block.post_attention(
                clip_f,
                attn_outputs['c_attn_out'],
                pre_attention_outputs['c_mod']
            )
            text_f = self.text_block.post_attention(
                text_f,
                attn_outputs['t_attn_out'],
                pre_attention_outputs['t_mod']
            )
            audio_f = self.audio_block.post_attention(
                audio_f,
                attn_outputs['a_attn_out'],
                pre_attention_outputs['a_mod']
            )

        return latent, clip_f, text_f, audio_f

    def forward(
        self,
        latent: torch.Tensor,
        clip_f: torch.Tensor,
        audio_f: torch.Tensor,
        text_f: torch.Tensor,
        global_c: torch.Tensor,
        extended_c: torch.Tensor,
        latent_rot: torch.Tensor,
        clip_rot: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of joint block

        Args:
            latent: Latent tensor of shape (B, N1, D)
            clip_f: CLIP features of shape (B, N2, D)
            audio_f: Audio features of shape (B, N3, D)
            text_f: Text features of shape (B, N4, D)
            global_c: Global condition of shape (B, D)
            extended_c: Extended condition of shape (B, D)
            latent_rot: Latent rotary embeddings
            clip_rot: CLIP rotary embeddings

        Returns:
            Tuple of (latent, clip_f) tensors
        """
        # Get sequence lengths
        latent_len = latent.shape[1]
        clip_len = clip_f.shape[1]
        text_len = text_f.shape[1]
        audio_len = audio_f.shape[1]

        # Compute pre-attention for all modalities
        pre_attention_outputs = self._compute_pre_attention_for_all_modalities(
            latent, clip_f, audio_f, text_f,
            global_c, extended_c, latent_rot, clip_rot
        )

        # Concatenate QKV from all modalities
        joint_qkv = self._concatenate_qkv(pre_attention_outputs)

        # Compute joint attention
        attn_out = attention(*joint_qkv)

        # Split attention output by modality
        attn_outputs = self._split_attention_output(
            attn_out, latent_len, audio_len, clip_len, text_len
        )

        # Apply post-attention processing
        latent, clip_f, text_f, audio_f = self._apply_post_attention(
            latent, clip_f, audio_f, text_f,
            attn_outputs, pre_attention_outputs
        )

        return latent, clip_f, text_f, audio_f


class FinalBlock(nn.Module):
    """Final block for output generation"""

    def __init__(self, dim: int, out_dim: int):
        """
        Initialize final block

        Args:
            dim: Dimension of input features
            out_dim: Dimension of output features
        """
        super().__init__()

        # Validate inputs
        if dim <= 0:
            raise ValueError(f"dim must be positive, got {dim}")
        if out_dim <= 0:
            raise ValueError(f"out_dim must be positive, got {out_dim}")

        self.dim = dim
        self.out_dim = out_dim

        # Initialize components
        self._initialize_components()

    def _initialize_components(self):
        """Initialize all components of the final block"""
        # Modulation layer
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(self.dim, 2 * self.dim, bias=True)
        )

        # Normalization layer
        self.norm = nn.LayerNorm(self.dim, elementwise_affine=False)

        # Convolutional layer
        self.conv = ChannelLastConv1d(self.dim, self.out_dim, kernel_size=7, padding=3)

    def _compute_modulation(self, c: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute modulation parameters from condition

        Args:
            c: Condition tensor of shape (B, D)

        Returns:
            Tuple of (shift, scale) tensors
        """
        modulation = self.adaLN_modulation(c)
        shift, scale = modulation.chunk(2, dim=-1)
        return shift, scale

    def forward(self, latent: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of final block

        Args:
            latent: Input tensor of shape (B, N, D)
            c: Condition tensor of shape (B, D)

        Returns:
            Output tensor of shape (B, N, out_dim)
        """
        # Compute modulation parameters
        shift, scale = self._compute_modulation(c)

        # Apply modulation to latent
        latent_modulated = modulate(self.norm(latent), shift, scale)

        # Apply convolution
        output = self.conv(latent_modulated)

        return output
