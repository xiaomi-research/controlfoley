import logging
from dataclasses import dataclass
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from lib.rotary_embeddings import compute_rope_rotations
from lib.embeddings import TimestepEmbedder
from controlfoley.neural_blocks import MLP, REPA_MLP, REPA_MLP_large, ChannelLastConv1d, ConvMLP
from controlfoley.attention_layers import (FinalBlock, JointBlock, MMDitSingleBlock)

log = logging.getLogger()


@dataclass
class PreprocessedConditions:
    """Container for preprocessed condition features"""
    clip_f: torch.Tensor
    sync_f: torch.Tensor
    text_f: torch.Tensor
    audio_f: torch.Tensor
    timbre_f: torch.Tensor
    clip_f_c: torch.Tensor
    text_f_c: torch.Tensor


class AudioGenerationNetwork(nn.Module):
    """
    Neural network for audio generation with multimodal conditioning
    Partially adapted from https://github.com/facebookresearch/DiT
    """

    def __init__(
        self,
        *,
        mode: int,
        latent_dim: int,
        clip_dim: int,
        visual_dim: int,
        sync_dim: int,
        text_dim: int,
        audio_dim: int,
        timbre_dim: int,
        hidden_dim: int,
        depth: int,
        fused_depth: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        latent_seq_len: int,
        clip_seq_len: int,
        visual_seq_len: int,
        sync_seq_len: int,
        text_seq_len: int = 77,
        audio_seq_len: int = 1,
        timbre_seq_len: int = 1,
        latent_mean: Optional[torch.Tensor] = None,
        latent_std: Optional[torch.Tensor] = None,
        empty_string_feat: Optional[torch.Tensor] = None,
        v2: bool = False
    ) -> None:
        """
        Initialize AudioGenerationNetwork

        Args:
            mode: Mode for REPA MLP selection
            latent_dim: Dimension of latent space
            clip_dim: Dimension of CLIP features
            visual_dim: Dimension of visual features
            sync_dim: Dimension of sync features
            text_dim: Dimension of text features
            audio_dim: Dimension of audio features
            timbre_dim: Dimension of timbre features
            hidden_dim: Hidden dimension for the network
            depth: Total depth of the network
            fused_depth: Number of fused blocks
            num_heads: Number of attention heads
            mlp_ratio: MLP ratio
            latent_seq_len: Length of latent sequence
            clip_seq_len: Length of CLIP sequence
            visual_seq_len: Length of visual sequence
            sync_seq_len: Length of sync sequence
            text_seq_len: Length of text sequence
            audio_seq_len: Length of audio sequence
            timbre_seq_len: Length of timbre sequence
            latent_mean: Mean for latent normalization
            latent_std: Std for latent normalization
            empty_string_feat: Empty string features
            v2: Version 2 flag
        """
        super().__init__()

        # Validate inputs
        self._validate_init_parameters(
            mode, latent_dim, clip_dim, visual_dim, sync_dim,
            text_dim, audio_dim, timbre_dim, hidden_dim,
            depth, fused_depth, num_heads, mlp_ratio,
            latent_seq_len, clip_seq_len, visual_seq_len, sync_seq_len,
            text_seq_len, audio_seq_len, timbre_seq_len
        )

        # Store configuration
        self._store_configuration(
            v2, latent_dim, latent_seq_len, clip_seq_len, visual_seq_len, sync_seq_len,
            text_seq_len, audio_seq_len, timbre_seq_len,
            hidden_dim, num_heads
        )

        # Initialize all components
        self._initialize_all_components(
            mode, latent_dim, clip_dim, visual_dim, sync_dim,
            text_dim, audio_dim, timbre_dim, hidden_dim,
            depth, fused_depth, num_heads, mlp_ratio,
            latent_mean, latent_std, empty_string_feat,
            v2, text_seq_len, clip_seq_len
        )

        # Initialize weights and rotations
        self.initialize_weights()
        self.initialize_rotations()

    def _validate_init_parameters(
        self,
        mode: int,
        latent_dim: int,
        clip_dim: int,
        visual_dim: int,
        sync_dim: int,
        text_dim: int,
        audio_dim: int,
        timbre_dim: int,
        hidden_dim: int,
        depth: int,
        fused_depth: int,
        num_heads: int,
        mlp_ratio: float,
        latent_seq_len: int,
        clip_seq_len: int,
        visual_seq_len: int,
        sync_seq_len: int,
        text_seq_len: int,
        audio_seq_len: int,
        timbre_seq_len: int
    ) -> None:
        """Validate initialization parameters"""
        if mode not in [0, 1, 2]:
            raise ValueError(f"mode must be 0, 1, or 2, got {mode}")

        if depth <= fused_depth:
            raise ValueError(f"depth ({depth}) must be greater than fused_depth ({fused_depth})")

        if num_heads <= 0:
            raise ValueError(f"num_heads must be positive, got {num_heads}")

        if mlp_ratio <= 0:
            raise ValueError(f"mlp_ratio must be positive, got {mlp_ratio}")

        # Validate sequence lengths
        for name, value in [
            ("latent_seq_len", latent_seq_len),
            ("clip_seq_len", clip_seq_len),
            ("visual_seq_len", visual_seq_len),
            ("sync_seq_len", sync_seq_len),
            ("text_seq_len", text_seq_len),
            ("audio_seq_len", audio_seq_len),
            ("timbre_seq_len", timbre_seq_len)
        ]:
            if value <= 0:
                raise ValueError(f"{name} must be positive, got {value}")

        # Validate dimensions
        for name, value in [
            ("latent_dim", latent_dim),
            ("clip_dim", clip_dim),
            ("visual_dim", visual_dim),
            ("sync_dim", sync_dim),
            ("text_dim", text_dim),
            ("audio_dim", audio_dim),
            ("timbre_dim", timbre_dim),
            ("hidden_dim", hidden_dim)
        ]:
            if value <= 0:
                raise ValueError(f"{name} must be positive, got {value}")

    def _store_configuration(
        self,
        v2: bool,
        latent_dim: int,
        latent_seq_len: int,
        clip_seq_len: int,
        visual_seq_len: int,
        sync_seq_len: int,
        text_seq_len: int,
        audio_seq_len: int,
        timbre_seq_len: int,
        hidden_dim: int,
        num_heads: int
    ) -> None:
        """Store configuration parameters"""
        self.v2 = v2
        self.latent_dim = latent_dim
        self._latent_seq_len = latent_seq_len
        self._clip_seq_len = clip_seq_len
        self._visual_seq_len = visual_seq_len
        self._sync_seq_len = sync_seq_len
        self._text_seq_len = text_seq_len
        self._audio_seq_len = audio_seq_len
        self._timbre_seq_len = timbre_seq_len
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads

    def _initialize_all_components(
        self,
        mode: int,
        latent_dim: int,
        clip_dim: int,
        visual_dim: int,
        sync_dim: int,
        text_dim: int,
        audio_dim: int,
        timbre_dim: int,
        hidden_dim: int,
        depth: int,
        fused_depth: int,
        num_heads: int,
        mlp_ratio: float,
        latent_mean: Optional[torch.Tensor],
        latent_std: Optional[torch.Tensor],
        empty_string_feat: Optional[torch.Tensor],
        v2: bool,
        text_seq_len: int,
        clip_seq_len: int
    ) -> None:
        """Initialize all network components"""
        # Initialize input projection layers
        self._initialize_input_projections(
            v2, latent_dim, clip_dim, visual_dim, sync_dim,
            text_dim, audio_dim, timbre_dim, hidden_dim
        )

        # Initialize condition projection layers
        self._initialize_condition_projections(hidden_dim)

        # Initialize REPA MLP
        self._initialize_repa_mlp(mode)

        # Initialize position embeddings
        self._initialize_position_embeddings(sync_dim)

        # Initialize final layer
        self._initialize_final_layer(hidden_dim, latent_dim)

        # Initialize timestep embedder
        self._initialize_timestep_embedder(v2, hidden_dim)

        # Initialize transformer blocks
        self._initialize_transformer_blocks(
            depth, fused_depth, hidden_dim, num_heads, mlp_ratio
        )

        # Initialize normalization parameters
        self._initialize_normalization_parameters(
            latent_mean, latent_std, latent_dim
        )

        # Initialize empty features
        self._initialize_empty_features(
            empty_string_feat, text_seq_len, text_dim,
            clip_dim, visual_dim, sync_dim, audio_dim, timbre_dim
        )

    def _initialize_input_projections(
        self,
        v2: bool,
        latent_dim: int,
        clip_dim: int,
        visual_dim: int,
        sync_dim: int,
        text_dim: int,
        audio_dim: int,
        timbre_dim: int,
        hidden_dim: int
    ) -> None:
        """Initialize input projection layers for all modalities"""
        if v2:
            # V2 projections with SiLU activation
            self.audio_input_proj = self._create_v2_audio_projection(
                latent_dim, hidden_dim
            )
            self.clip_input_proj = self._create_v2_clip_projection(
                clip_dim, hidden_dim
            )
            self.visual_input_proj = self._create_v2_visual_projection(
                visual_dim, hidden_dim
            )
            self.sync_input_proj = self._create_v2_sync_projection(
                sync_dim, hidden_dim
            )
            self.text_input_proj = self._create_v2_text_projection(
                text_dim, hidden_dim
            )
            self.clap_input_proj = self._create_v2_audio_projection(
                audio_dim, hidden_dim
            )
            self.timbre_input_proj = self._create_v2_text_projection(
                timbre_dim, hidden_dim
            )
        else:
            # V1 projections with SELU activation
            self.audio_input_proj = self._create_v1_audio_projection(
                latent_dim, hidden_dim
            )
            self.clip_input_proj = self._create_v1_clip_projection(
                clip_dim, hidden_dim
            )
            self.visual_input_proj = self._create_v1_visual_projection(
                visual_dim, hidden_dim
            )
            self.sync_input_proj = self._create_v1_sync_projection(
                sync_dim, hidden_dim
            )
            self.text_input_proj = self._create_v1_text_projection(
                text_dim, hidden_dim
            )
            self.clap_input_proj = self._create_v1_text_projection(
                audio_dim, hidden_dim
            )
            self.timbre_input_proj = self._create_v1_text_projection(
                timbre_dim, hidden_dim
            )

    def _create_v2_audio_projection(
        self,
        input_dim: int,
        hidden_dim: int
    ) -> nn.Sequential:
        """Create V2 audio projection"""
        return nn.Sequential(
            ChannelLastConv1d(input_dim, hidden_dim, kernel_size=7, padding=3),
            nn.SiLU(),
            ConvMLP(hidden_dim, hidden_dim * 4, kernel_size=7, padding=3),
        )

    def _create_v2_clip_projection(
        self,
        input_dim: int,
        hidden_dim: int
    ) -> nn.Sequential:
        """Create V2 CLIP projection"""
        return nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            ConvMLP(hidden_dim, hidden_dim * 4, kernel_size=3, padding=1),
        )

    def _create_v2_visual_projection(
        self,
        input_dim: int,
        hidden_dim: int
    ) -> nn.Sequential:
        """Create V2 visual projection"""
        return nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            ConvMLP(hidden_dim, hidden_dim * 4, kernel_size=3, padding=1),
        )

    def _create_v2_sync_projection(
        self,
        input_dim: int,
        hidden_dim: int
    ) -> nn.Sequential:
        """Create V2 sync projection"""
        return nn.Sequential(
            ChannelLastConv1d(input_dim, hidden_dim, kernel_size=7, padding=3),
            nn.SiLU(),
            ConvMLP(hidden_dim, hidden_dim * 4, kernel_size=3, padding=1),
        )

    def _create_v2_text_projection(
        self,
        input_dim: int,
        hidden_dim: int
    ) -> nn.Sequential:
        """Create V2 text projection"""
        return nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            MLP(hidden_dim, hidden_dim * 4),
        )

    def _create_v1_audio_projection(
        self,
        input_dim: int,
        hidden_dim: int
    ) -> nn.Sequential:
        """Create V1 audio projection"""
        return nn.Sequential(
            ChannelLastConv1d(input_dim, hidden_dim, kernel_size=7, padding=3),
            nn.SELU(),
            ConvMLP(hidden_dim, hidden_dim * 4, kernel_size=7, padding=3),
        )

    def _create_v1_clip_projection(
        self,
        input_dim: int,
        hidden_dim: int
    ) -> nn.Sequential:
        """Create V1 CLIP projection"""
        return nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            ConvMLP(hidden_dim, hidden_dim * 4, kernel_size=3, padding=1),
        )

    def _create_v1_visual_projection(
        self,
        input_dim: int,
        hidden_dim: int
    ) -> nn.Sequential:
        """Create V1 visual projection"""
        return nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            ConvMLP(hidden_dim, hidden_dim * 4, kernel_size=3, padding=1),
        )

    def _create_v1_sync_projection(
        self,
        input_dim: int,
        hidden_dim: int
    ) -> nn.Sequential:
        """Create V1 sync projection"""
        return nn.Sequential(
            ChannelLastConv1d(input_dim, hidden_dim, kernel_size=7, padding=3),
            nn.SELU(),
            ConvMLP(hidden_dim, hidden_dim * 4, kernel_size=3, padding=1),
        )

    def _create_v1_text_projection(
        self,
        input_dim: int,
        hidden_dim: int
    ) -> nn.Sequential:
        """Create V1 text projection"""
        return nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            MLP(hidden_dim, hidden_dim * 4),
        )

    def _initialize_condition_projections(self, hidden_dim: int) -> None:
        """Initialize condition projection layers"""
        self.clip_cond_proj = nn.Linear(hidden_dim, hidden_dim)
        self.text_cond_proj = nn.Linear(hidden_dim, hidden_dim)
        self.timbre_cond_proj = nn.Linear(hidden_dim, hidden_dim)
        self.global_cond_mlp = MLP(hidden_dim, hidden_dim * 4)

    def _initialize_repa_mlp(self, mode: int) -> None:
        """Initialize REPA MLP based on mode"""
        if mode == 0:
            self.repa_mlp = REPA_MLP()
        else:
            self.repa_mlp = REPA_MLP_large()

    def _initialize_position_embeddings(self, sync_dim: int) -> None:
        """Initialize position embeddings for sync features"""
        # each synchformer output segment has 8 feature frames
        self.sync_pos_emb = nn.Parameter(torch.zeros((1, 1, 8, sync_dim)))

    def _initialize_final_layer(self, hidden_dim: int, latent_dim: int) -> None:
        """Initialize final output layer"""
        self.final_layer = FinalBlock(hidden_dim, latent_dim)

    def _initialize_timestep_embedder(self, v2: bool, hidden_dim: int) -> None:
        """Initialize timestep embedder"""
        if v2:
            self.t_embed = TimestepEmbedder(
                hidden_dim,
                frequency_embedding_size=hidden_dim,
                max_period=1
            )
        else:
            self.t_embed = TimestepEmbedder(
                hidden_dim,
                frequency_embedding_size=256,
                max_period=10000
            )

    def _initialize_transformer_blocks(
        self,
        depth: int,
        fused_depth: int,
        hidden_dim: int,
        num_heads: int,
        mlp_ratio: float
    ) -> None:
        """Initialize transformer blocks"""
        # Joint blocks
        self.joint_blocks = nn.ModuleList([
            JointBlock(
                hidden_dim,
                num_heads,
                mlp_ratio=mlp_ratio,
                pre_only=(i == depth - fused_depth - 1)
            )
            for i in range(depth - fused_depth)
        ])

        # Fused blocks
        self.fused_blocks = nn.ModuleList([
            MMDitSingleBlock(
                hidden_dim,
                num_heads,
                mlp_ratio=mlp_ratio,
                kernel_size=3,
                padding=1
            )
            for i in range(fused_depth)
        ])

    def _initialize_normalization_parameters(
        self,
        latent_mean: Optional[torch.Tensor],
        latent_std: Optional[torch.Tensor],
        latent_dim: int
    ) -> None:
        """Initialize normalization parameters"""
        if latent_mean is None:
            # these values are not meant to be used
            # if you don't provide mean/std here, we should load them later from a checkpoint
            assert latent_std is None
            latent_mean = torch.ones(latent_dim).view(1, 1, -1).fill_(float('nan'))
            latent_std = torch.ones(latent_dim).view(1, 1, -1).fill_(float('nan'))
        else:
            assert latent_std is not None
            assert latent_mean.numel() == latent_dim, f'{latent_mean.numel()=} != {latent_dim=}'

        self.latent_mean = nn.Parameter(latent_mean.view(1, 1, -1), requires_grad=False)
        self.latent_std = nn.Parameter(latent_std.view(1, 1, -1), requires_grad=False)

    def _initialize_empty_features(
        self,
        empty_string_feat: Optional[torch.Tensor],
        text_seq_len: int,
        text_dim: int,
        clip_dim: int,
        visual_dim: int,
        sync_dim: int,
        audio_dim: int,
        timbre_dim: int
    ) -> None:
        """Initialize empty feature parameters"""
        if empty_string_feat is None:
            empty_string_feat = torch.zeros((text_seq_len, text_dim))

        self.empty_string_feat = nn.Parameter(empty_string_feat, requires_grad=False)
        self.empty_clip_feat = nn.Parameter(torch.zeros(1, clip_dim), requires_grad=True)
        self.empty_visual_feat = nn.Parameter(torch.zeros(1, visual_dim), requires_grad=True)
        self.empty_sync_feat = nn.Parameter(torch.zeros(1, sync_dim), requires_grad=True)
        self.empty_audio_feat = nn.Parameter(torch.zeros(1, audio_dim), requires_grad=True)
        self.empty_timbre_feat = nn.Parameter(torch.zeros(1, timbre_dim), requires_grad=True)

    def initialize_rotations(self) -> None:
        """Initialize rotary position embeddings"""
        base_freq = 1.0

        # Compute latent rotations
        latent_rot = compute_rope_rotations(
            self._latent_seq_len,
            self.hidden_dim // self.num_heads,
            10000,
            freq_scaling=base_freq,
            device=self.device
        )

        # Compute CLIP rotations
        clip_rot = compute_rope_rotations(
            self._clip_seq_len,
            self.hidden_dim // self.num_heads,
            10000,
            freq_scaling=base_freq * self._latent_seq_len / self._clip_seq_len,
            device=self.device
        )

        self.latent_rot = nn.Buffer(latent_rot, persistent=False)
        self.clip_rot = nn.Buffer(clip_rot, persistent=False)

    def update_seq_lengths(
        self,
        latent_seq_len: int,
        clip_seq_len: int,
        visual_seq_len: int,
        sync_seq_len: int
    ) -> None:
        """Update sequence lengths and recompute rotations"""
        self._latent_seq_len = latent_seq_len
        self._clip_seq_len = clip_seq_len
        self._visual_seq_len = visual_seq_len
        self._sync_seq_len = sync_seq_len
        self.initialize_rotations()

    def initialize_weights(self) -> None:
        """Initialize network weights"""
        # Basic initialization
        self._apply_basic_init()

        # Initialize timestep embedding MLP
        self._init_timestep_embedding()

        # Zero-out adaLN modulation layers in DiT blocks
        self._init_adaln_modulation()

        # Zero-out output layers
        self._init_output_layers()

        # Initialize empty features
        self._init_empty_features()

    def _apply_basic_init(self) -> None:
        """Apply basic initialization to all modules"""
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

    def _init_timestep_embedding(self) -> None:
        """Initialize timestep embedding MLP"""
        nn.init.normal_(self.t_embed.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embed.mlp[2].weight, std=0.02)

    def _init_adaln_modulation(self) -> None:
        """Zero-out adaLN modulation layers in DiT blocks"""
        # Joint blocks
        for block in self.joint_blocks:
            nn.init.constant_(block.latent_block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.latent_block.adaLN_modulation[-1].bias, 0)
            nn.init.constant_(block.audio_block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.audio_block.adaLN_modulation[-1].bias, 0)
            nn.init.constant_(block.clip_block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.clip_block.adaLN_modulation[-1].bias, 0)
            nn.init.constant_(block.text_block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.text_block.adaLN_modulation[-1].bias, 0)

        # Fused blocks
        for block in self.fused_blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

    def _init_output_layers(self) -> None:
        """Zero-out output layers"""
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.conv.weight, 0)
        nn.init.constant_(self.final_layer.conv.bias, 0)

    def _init_empty_features(self) -> None:
        """Initialize empty features"""
        nn.init.constant_(self.sync_pos_emb, 0)
        nn.init.constant_(self.empty_clip_feat, 0)
        nn.init.constant_(self.empty_sync_feat, 0)

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        """
        Normalize latent tensor

        Args:
            x: Input tensor

        Returns:
            Normalized tensor
        """
        return x.sub_(self.latent_mean).div_(self.latent_std)

    def unnormalize(self, x: torch.Tensor) -> torch.Tensor:
        """
        Unnormalize latent tensor

        Args:
            x: Input tensor

        Returns:
            Unnormalized tensor
        """
        return x.mul_(self.latent_std).add_(self.latent_mean)

    def preprocess_conditions(
        self,
        clip_f: torch.Tensor,
        visual_f: torch.Tensor,
        sync_f: torch.Tensor,
        text_f: torch.Tensor,
        audio_f: torch.Tensor,
        timbre_f: torch.Tensor,
    ) -> PreprocessedConditions:
        """
        Preprocess conditions that do not depend on latent/time step
        These features are reused over steps during inference

        Args:
            clip_f: CLIP features
            visual_f: Visual features
            sync_f: Sync features
            text_f: Text features
            audio_f: Audio features
            timbre_f: Timbre features

        Returns:
            PreprocessedConditions object
        """
        # Validate input shapes
        self._validate_condition_shapes(
            clip_f, visual_f, sync_f, text_f, audio_f, timbre_f
        )

        # Get batch size
        bs = clip_f.shape[0]

        # Process sync features
        sync_f = self._process_sync_features(sync_f, bs)

        # Process clip and visual features
        clip_f = self._process_clip_visual_features(clip_f, visual_f)

        # Process other features
        sync_f = self.sync_input_proj(sync_f)
        text_f = self.text_input_proj(text_f)
        audio_f = self.clap_input_proj(audio_f)
        timbre_f = self.timbre_input_proj(timbre_f)

        # Upsample sync features to match audio
        sync_f = self._upsample_sync_features(sync_f)

        # Compute conditional features
        clip_f_c = self.clip_cond_proj(clip_f.mean(dim=1))
        text_f_c = self.text_cond_proj(text_f.mean(dim=1))
        timbre_f = self.timbre_cond_proj(timbre_f.mean(dim=1))

        return PreprocessedConditions(
            clip_f=clip_f,
            sync_f=sync_f,
            text_f=text_f,
            audio_f=audio_f,
            timbre_f=timbre_f,
            clip_f_c=clip_f_c,
            text_f_c=text_f_c
        )

    def _validate_condition_shapes(
        self,
        clip_f: torch.Tensor,
        visual_f: torch.Tensor,
        sync_f: torch.Tensor,
        text_f: torch.Tensor,
        audio_f: torch.Tensor,
        timbre_f: torch.Tensor
    ) -> None:
        """Validate condition tensor shapes"""
        assert clip_f.shape[1] == self._clip_seq_len, f'{clip_f.shape=} {self._clip_seq_len=}'
        assert visual_f.shape[1] == self._visual_seq_len, f'{visual_f.shape=} {self._visual_seq_len=}'
        assert sync_f.shape[1] == self._sync_seq_len, f'{sync_f.shape=} {self._sync_seq_len=}'
        assert text_f.shape[1] == self._text_seq_len, f'{text_f.shape=} {self._text_seq_len=}'
        assert audio_f.shape[1] == self._audio_seq_len, f'{audio_f.shape=} {self._audio_seq_len=}'
        assert timbre_f.shape[1] == self._timbre_seq_len, f'{timbre_f.shape=} {self._timbre_seq_len=}'

    def _process_sync_features(
        self,
        sync_f: torch.Tensor,
        bs: int
    ) -> torch.Tensor:
        """Process sync features with position embeddings"""
        # B * num_segments (24) * 8 * 768
        num_sync_segments = self._sync_seq_len // 8
        sync_f = sync_f.view(bs, num_sync_segments, 8, -1) + self.sync_pos_emb
        sync_f = sync_f.flatten(1, 2)  # (B, VN, D)
        return sync_f

    def _process_clip_visual_features(
        self,
        clip_f: torch.Tensor,
        visual_f: torch.Tensor
    ) -> torch.Tensor:
        """Process and combine CLIP and visual features"""
        # Process features
        processed_clip_f = self.clip_input_proj(clip_f)
        processed_visual_f = self.visual_input_proj(visual_f)

        # Extend visual features to match clip
        if processed_visual_f.shape[1] != processed_clip_f.shape[1]:
            processed_visual_f = processed_visual_f.transpose(1, 2)
            processed_visual_f = F.interpolate(
                processed_visual_f,
                size=processed_clip_f.shape[1],
                mode='linear',
                align_corners=False
            )
            processed_visual_f = processed_visual_f.transpose(1, 2)

        # Combine features
        clip_f = processed_clip_f + processed_visual_f
        return clip_f

    def _upsample_sync_features(self, sync_f: torch.Tensor) -> torch.Tensor:
        """Upsample sync features to match latent sequence length"""
        # Transpose to (B, D, VN)
        sync_f = sync_f.transpose(1, 2)

        # Interpolate to latent sequence length
        sync_f = F.interpolate(sync_f, size=self._latent_seq_len, mode='nearest-exact')

        # Transpose back to (B, N, D)
        sync_f = sync_f.transpose(1, 2)

        return sync_f

    def predict_flow(
        self,
        latent: torch.Tensor,
        t: torch.Tensor,
        conditions: PreprocessedConditions
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Predict flow for non-cacheable computations

        Args:
            latent: Latent tensor
            t: Timestep tensor
            conditions: Preprocessed conditions

        Returns:
            Tuple of (flow, multimodal, hidden)
        """
        # Validate latent shape
        assert latent.shape[1] == self._latent_seq_len, f'{latent.shape=} {self._latent_seq_len=}'

        # Extract conditions
        clip_f = conditions.clip_f
        sync_f = conditions.sync_f
        text_f = conditions.text_f
        audio_f = conditions.audio_f
        timbre_f = conditions.timbre_f
        clip_f_c = conditions.clip_f_c
        text_f_c = conditions.text_f_c

        # Process latent
        latent = self.audio_input_proj(latent)

        # Compute global condition
        global_c = self.global_cond_mlp(clip_f_c + text_f_c + timbre_f)
        multimodal = global_c

        # Compute timestep embedding
        global_c = self.t_embed(t).unsqueeze(1) + global_c.unsqueeze(1)

        # Compute extended condition
        extended_c = global_c + sync_f

        # Apply joint blocks
        for block in self.joint_blocks:
            latent, clip_f, text_f, audio_f = block(
                latent, clip_f, audio_f, text_f,
                global_c, extended_c,
                self.latent_rot, self.clip_rot
            )

        # Apply fused blocks and compute REPA features
        hidden = None
        for i, block in enumerate(self.fused_blocks):
            latent = block(latent, extended_c, self.latent_rot)
            if i == 7:
                hidden = self.repa_mlp(latent)

        # Compute flow
        flow = self.final_layer(latent, extended_c)

        return flow, multimodal, hidden

    def forward(
        self,
        latent: torch.Tensor,
        clip_f: torch.Tensor,
        visual_f: torch.Tensor,
        sync_f: torch.Tensor,
        text_f: torch.Tensor,
        audio_f: torch.Tensor,
        timbre_f: torch.Tensor,
        t: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass of the network

        Args:
            latent: Latent tensor of shape (B, N, C)
            clip_f: CLIP features of shape (B, T, C_V)
            visual_f: Visual features
            sync_f: Sync features
            text_f: Text features
            audio_f: Audio features
            timbre_f: Timbre features
            t: Timestep tensor of shape (B,)

        Returns:
            Tuple of (flow, multimodal, hidden)
        """
        # Preprocess conditions
        conditions = self.preprocess_conditions(
            clip_f, visual_f, sync_f, text_f, audio_f, timbre_f
        )

        # Predict flow
        flow, multimodal, hidden = self.predict_flow(latent, t, conditions)

        return flow, multimodal, hidden

    def get_empty_string_sequence(self, bs: int) -> torch.Tensor:
        """Get empty string sequence"""
        return self.empty_string_feat.unsqueeze(0).expand(bs, -1, -1)

    def get_empty_clip_sequence(self, bs: int) -> torch.Tensor:
        """Get empty CLIP sequence"""
        return self.empty_clip_feat.unsqueeze(0).expand(bs, self._clip_seq_len, -1)

    def get_empty_visual_sequence(self, bs: int) -> torch.Tensor:
        """Get empty visual sequence"""
        return self.empty_visual_feat.unsqueeze(0).expand(bs, self._visual_seq_len, -1)

    def get_empty_sync_sequence(self, bs: int) -> torch.Tensor:
        """Get empty sync sequence"""
        return self.empty_sync_feat.unsqueeze(0).expand(bs, self._sync_seq_len, -1)

    def get_empty_audio_sequence(self, bs: int) -> torch.Tensor:
        """Get empty audio sequence"""
        return self.empty_audio_feat.unsqueeze(0).expand(bs, 1, -1)

    def get_empty_timbre_sequence(self, bs: int) -> torch.Tensor:
        """Get empty timbre sequence"""
        return self.empty_timbre_feat.unsqueeze(0).expand(bs, 1, -1)

    def get_empty_conditions(
        self,
        bs: int,
        *,
        negative_text_features: Optional[torch.Tensor] = None
    ) -> PreprocessedConditions:
        """
        Get empty conditions for classifier-free guidance

        Args:
            bs: Batch size
            negative_text_features: Optional negative text features

        Returns:
            PreprocessedConditions object
        """
        # Get empty sequences
        if negative_text_features is not None:
            empty_text = negative_text_features
        else:
            empty_text = self.get_empty_string_sequence(1)

        empty_clip = self.get_empty_clip_sequence(1)
        empty_visual = self.get_empty_visual_sequence(1)
        empty_sync = self.get_empty_sync_sequence(1)
        empty_audio = self.get_empty_audio_sequence(1)
        empty_timbre = self.get_empty_timbre_sequence(1)

        # Preprocess empty conditions
        conditions = self.preprocess_conditions(
            empty_clip, empty_visual, empty_sync,
            empty_text, empty_audio, empty_timbre
        )

        # Expand to batch size
        conditions.clip_f = conditions.clip_f.expand(bs, -1, -1)
        conditions.sync_f = conditions.sync_f.expand(bs, -1, -1)
        conditions.audio_f = conditions.audio_f.expand(bs, -1, -1)
        conditions.timbre_f = conditions.timbre_f.expand(bs, -1,)
        conditions.clip_f_c = conditions.clip_f_c.expand(bs, -1)

        if negative_text_features is None:
            conditions.text_f = conditions.text_f.expand(bs, -1, -1)
            conditions.text_f_c = conditions.text_f_c.expand(bs, -1)

        return conditions

    def ode_wrapper(
        self,
        t: torch.Tensor,
        latent: torch.Tensor,
        conditions: PreprocessedConditions,
        empty_conditions: PreprocessedConditions,
        cfg_strength: float
    ) -> torch.Tensor:
        """
        ODE wrapper for classifier-free guidance

        Args:
            t: Timestep
            latent: Latent tensor
            conditions: Conditions
            empty_conditions: Empty conditions for CFG
            cfg_strength: CFG strength

        Returns:
            Flow tensor
        """
        t = t * torch.ones(len(latent), device=latent.device, dtype=latent.dtype)
        flow, multimodal, hidden = self.predict_flow(latent, t, conditions)

        if cfg_strength < 1.0:
            return flow
        else:
            flow_empty, multimodal_empty, hidden_empty = self.predict_flow(latent, t, empty_conditions)
            return (cfg_strength * flow) + (1 - cfg_strength) * flow_empty

    def load_weights(self, src_dict) -> None:
        """
        Load weights from state dictionary

        Args:
            src_dict: Source state dictionary
        """
        # Remove incompatible keys
        if 't_embed.freqs' in src_dict:
            del src_dict['t_embed.freqs']
        if 'latent_rot' in src_dict:
            del src_dict['latent_rot']
        if 'clip_rot' in src_dict:
            del src_dict['clip_rot']

        # Load state dict
        self.load_state_dict(src_dict, strict=True)

    @property
    def device(self) -> torch.device:
        """Get device of the model"""
        return self.latent_mean.device

    @property
    def latent_seq_len(self) -> int:
        """Get latent sequence length"""
        return self._latent_seq_len

    @property
    def clip_seq_len(self) -> int:
        """Get CLIP sequence length"""
        return self._clip_seq_len

    @property
    def visual_seq_len(self) -> int:
        """Get visual sequence length"""
        return self._visual_seq_len

    @property
    def sync_seq_len(self) -> int:
        """Get sync sequence length"""
        return self._sync_seq_len

    @property
    def audio_seq_len(self) -> int:
        """Get audio sequence length"""
        return self._audio_seq_len

    @property
    def timbre_seq_len(self) -> int:
        """Get timbre sequence length"""
        return self._timbre_seq_len


def create_large_44k_model(**kwargs) -> AudioGenerationNetwork:
    """
    Create a large 44kHz audio generation model

    Args:
        **kwargs: Additional arguments for AudioGenerationNetwork

    Returns:
        AudioGenerationNetwork instance
    """
    num_heads = 14
    return AudioGenerationNetwork(
        mode=2,
        latent_dim=40,
        clip_dim=1024,
        visual_dim=768,
        sync_dim=768,
        text_dim=1024,
        audio_dim=512,
        timbre_dim=1536,
        hidden_dim=64 * num_heads,
        depth=54,
        fused_depth=36,
        num_heads=num_heads,
        latent_seq_len=345,
        clip_seq_len=64,
        visual_seq_len=32,
        sync_seq_len=192,
        **kwargs
    )


def create_audio_generation_model(model_name: str, **kwargs) -> AudioGenerationNetwork:
    """
    Factory function to create audio generation models

    Args:
        model_name: Name of the model to create
        **kwargs: Additional arguments for model creation

    Returns:
        AudioGenerationNetwork instance

    Raises:
        ValueError: If model_name is not recognized
    """
    model_creators = {
        'large_44k': create_large_44k_model,
    }

    if model_name not in model_creators:
        raise ValueError(
            f'Unknown model name: {model_name}. '
            f'Available models: {list(model_creators.keys())}'
        )

    return model_creators[model_name](**kwargs)


if __name__ == '__main__':
    # Test model creation and parameter count
    model = create_audio_generation_model('large_44k')

    # Calculate parameter count in millions
    total_parameters = sum(param.numel() for param in model.parameters()) / 1e6
    print(f'Total parameters: {total_parameters:.2f}M')
