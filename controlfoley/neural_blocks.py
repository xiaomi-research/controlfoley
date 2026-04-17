from torch import nn
import torch
from torch.nn import functional as F


class ChannelLastConv1d(nn.Conv1d):
    """1D convolution with channel-last format for better memory efficiency"""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Validate input shape
        if x.dim() != 3:
            raise ValueError(f"Expected 3D tensor (B, T, C), got shape {x.shape}")

        # Permute from (B, T, C) to (B, C, T) for Conv1d
        x_transposed = x.permute(0, 2, 1)

        # Apply convolution
        x_conv = super().forward(x_transposed)

        # Permute back to (B, T, C)
        x_output = x_conv.permute(0, 2, 1)

        return x_output


class REPA_MLP(nn.Module):
    """Multi-layer perceptron for REPA feature processing with temporal downsampling"""

    def __init__(self):
        super().__init__()

        # Build feature transformation pipeline
        self._build_feature_transformer()

        # Build temporal downsampler
        self._build_temporal_downsampler()

    def _build_temporal_downsampler(self):
        """Create adaptive pooling layer for temporal dimension reduction"""
        self.temporal_downsample = nn.AdaptiveAvgPool1d(200)

    def _build_feature_transformer(self):
        """Create MLP for feature dimension transformation"""
        self.feature_mlp = nn.Sequential(
            nn.Linear(448, 768),
            nn.LayerNorm(768),
            nn.GELU()
        )

    def _validate_input(self, x: torch.Tensor) -> None:
        """Validate input tensor shape and dimensions"""
        if x.dim() != 3:
            raise ValueError(f"Expected 3D tensor (B, T, C), got {x.dim()}D tensor with shape {x.shape}")

        batch_size, time_steps, features = x.shape
        if features != 448:
            raise ValueError(f"Expected feature dimension 448, got {features}")

    def _apply_temporal_downsampling(self, x: torch.Tensor) -> torch.Tensor:
        """Apply temporal downsampling to input tensor"""
        # Transpose to (B, C, T) for pooling
        x_transposed = x.transpose(1, 2)

        # Apply adaptive pooling
        x_pooled = self.temporal_downsample(x_transposed)

        # Transpose back to (B, T, C)
        x_output = x_pooled.transpose(1, 2)

        return x_output

    def _apply_feature_transformation(self, x: torch.Tensor) -> torch.Tensor:
        """Apply MLP transformation to feature dimension"""
        return self.feature_mlp(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for REPA MLP

        Args:
            x: Input tensor of shape (B, T, 448)

        Returns:
            Output tensor of shape (B, 200, 768)
        """
        # Input validation
        self._validate_input(x)

        # Apply temporal downsampling
        x_downsampled = self._apply_temporal_downsampling(x)

        # Apply feature transformation
        x_output = self._apply_feature_transformation(x_downsampled)

        return x_output


class REPA_MLP_large(nn.Module):
    """Large version of REPA MLP for higher-dimensional features"""

    def __init__(self):
        super().__init__()

        # Initialize feature transformation network
        self._initialize_network()

    def _initialize_network(self):
        """Build the feature transformation network"""
        self.feature_mlp = nn.Sequential(
            nn.Linear(896, 896),
            nn.LayerNorm(896),
            nn.GELU()
        )

    def _validate_input(self, x: torch.Tensor) -> None:
        """Validate input tensor shape and dimensions"""
        if x.dim() != 3:
            raise ValueError(f"Expected 3D tensor (B, T, C), got {x.dim()}D tensor with shape {x.shape}")

        _, _, features = x.shape
        if features != 896:
            raise ValueError(f"Expected feature dimension 896, got {features}")

    def _apply_temporal_pooling(self, x: torch.Tensor) -> torch.Tensor:
        """Apply mean pooling across temporal dimension"""
        return torch.mean(x, dim=1)

    def _apply_feature_transformation(self, x: torch.Tensor) -> torch.Tensor:
        """Apply MLP transformation to features"""
        return self.feature_mlp(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for REPA MLP large

        Args:
            x: Input tensor of shape (B, T, 896)

        Returns:
            Output tensor of shape (B, 896)
        """
        # Input validation
        self._validate_input(x)

        # Apply temporal pooling
        x_pooled = self._apply_temporal_pooling(x)

        # Apply feature transformation
        x_output = self._apply_feature_transformation(x_pooled)

        return x_output


class MLP(nn.Module):
    """
    FeedForward module with SwiGLU activation
    Reference: https://github.com/Stability-AI/sd3-ref
    """

    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        multiple_of: int = 256,
    ):
        """
        Initialize the FeedForward module.

        Args:
            dim (int): Input dimension.
            hidden_dim (int): Hidden dimension of the feedforward layer.
            multiple_of (int): Value to ensure hidden dimension is a multiple of this value.
        """
        super().__init__()

        # Validate inputs
        if dim <= 0:
            raise ValueError(f"dim must be positive, got {dim}")
        if hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be positive, got {hidden_dim}")
        if multiple_of <= 0:
            raise ValueError(f"multiple_of must be positive, got {multiple_of}")

        # Calculate adjusted hidden dimension
        self.hidden_dim = self._calculate_hidden_dim(hidden_dim, multiple_of)

        # Initialize linear layers
        self._initialize_layers(dim)

    def _calculate_hidden_dim(self, hidden_dim: int, multiple_of: int) -> int:
        """Calculate hidden dimension adjusted to be a multiple of multiple_of"""
        adjusted_dim = int(2 * hidden_dim / 3)
        adjusted_dim = multiple_of * ((adjusted_dim + multiple_of - 1) // multiple_of)
        return adjusted_dim

    def _initialize_layers(self, dim: int):
        """Initialize the three linear layers"""
        self.w1 = nn.Linear(dim, self.hidden_dim, bias=False)
        self.w2 = nn.Linear(self.hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, self.hidden_dim, bias=False)

    def _apply_swiglu(self, x: torch.Tensor) -> torch.Tensor:
        """Apply SwiGLU activation: SiLU(w1(x)) * w3(x)"""
        return F.silu(self.w1(x)) * self.w3(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with SwiGLU activation

        Args:
            x: Input tensor

        Returns:
            Output tensor after SwiGLU transformation
        """
        # Apply SwiGLU activation
        activated = self._apply_swiglu(x)

        # Apply final linear transformation
        output = self.w2(activated)

        return output


class ConvMLP(nn.Module):
    """
    Convolutional FeedForward module with SwiGLU activation
    Uses ChannelLastConv1d for efficient computation
    """

    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        multiple_of: int = 256,
        kernel_size: int = 3,
        padding: int = 1,
    ):
        """
        Initialize the Convolutional FeedForward module.

        Args:
            dim (int): Input dimension.
            hidden_dim (int): Hidden dimension of the feedforward layer.
            multiple_of (int): Value to ensure hidden dimension is a multiple of this value.
            kernel_size (int): Kernel size for convolution layers.
            padding (int): Padding for convolution layers.
        """
        super().__init__()

        # Validate inputs
        if dim <= 0:
            raise ValueError(f"dim must be positive, got {dim}")
        if hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be positive, got {hidden_dim}")
        if multiple_of <= 0:
            raise ValueError(f"multiple_of must be positive, got {multiple_of}")
        if kernel_size <= 0:
            raise ValueError(f"kernel_size must be positive, got {kernel_size}")

        # Calculate adjusted hidden dimension
        self.hidden_dim = self._calculate_hidden_dim(hidden_dim, multiple_of)

        # Store convolution parameters
        self.kernel_size = kernel_size
        self.padding = padding

        # Initialize convolutional layers
        self._initialize_conv_layers(dim)

    def _calculate_hidden_dim(self, hidden_dim: int, multiple_of: int) -> int:
        """Calculate hidden dimension adjusted to be a multiple of multiple_of"""
        adjusted_dim = int(2 * hidden_dim / 3)
        adjusted_dim = multiple_of * ((adjusted_dim + multiple_of - 1) // multiple_of)
        return adjusted_dim

    def _initialize_conv_layers(self, dim: int):
        """Initialize the three convolutional layers"""
        self.w1 = ChannelLastConv1d(
            dim,
            self.hidden_dim,
            bias=False,
            kernel_size=self.kernel_size,
            padding=self.padding
        )
        self.w2 = ChannelLastConv1d(
            self.hidden_dim,
            dim,
            bias=False,
            kernel_size=self.kernel_size,
            padding=self.padding
        )
        self.w3 = ChannelLastConv1d(
            dim,
            self.hidden_dim,
            bias=False,
            kernel_size=self.kernel_size,
            padding=self.padding
        )

    def _apply_swiglu(self, x: torch.Tensor) -> torch.Tensor:
        """Apply SwiGLU activation: SiLU(w1(x)) * w3(x)"""
        return F.silu(self.w1(x)) * self.w3(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with convolutional SwiGLU activation

        Args:
            x: Input tensor

        Returns:
            Output tensor after convolutional SwiGLU transformation
        """
        # Apply SwiGLU activation with convolutions
        activated = self._apply_swiglu(x)

        # Apply final convolutional transformation
        output = self.w2(activated)

        return output
