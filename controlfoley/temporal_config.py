import math
import dataclasses


@dataclasses.dataclass
class TemporalConfiguration:
    """
    Configuration class for temporal sequence parameters in audio-visual processing.
    Defines frame rates, sampling rates, and sequence lengths for different modalities.
    """
    
    # Core temporal parameters
    total_time_seconds: float
    
    # Audio-related parameters
    audio_sample_rate: int
    spec_frame_frequency: int
    latent_reduction_factor: int = 2
    
    # Visual-related parameters
    clip_frame_frequency: int = 8
    visual_frame_frequency: int = 4
    sync_frame_frequency: int = 25
    sync_segment_frame_count: int = 16
    sync_stride_frames: int = 8
    sync_downsampling_factor: int = 2
    
    def __post_init__(self):
        """Validate configuration parameters after initialization."""
        if self.total_time_seconds <= 0:
            raise ValueError(f"Total time must be positive, got {self.total_time_seconds}")
        if self.audio_sample_rate <= 0:
            raise ValueError(f"Audio sample rate must be positive, got {self.audio_sample_rate}")
        if self.spec_frame_frequency <= 0:
            raise ValueError(f"Spectrogram frame rate must be positive, got {self.spec_frame_frequency}")
        if self.latent_reduction_factor <= 0:
            raise ValueError(f"Latent reduction factor must be positive, got {self.latent_reduction_factor}")
        if self.clip_frame_frequency <= 0:
            raise ValueError(f"Clip frame frequency must be positive, got {self.clip_frame_frequency}")
        if self.visual_frame_frequency <= 0:
            raise ValueError(f"Visual frame frequency must be positive, got {self.visual_frame_frequency}")
        if self.sync_frame_frequency <= 0:
            raise ValueError(f"Sync frame frequency must be positive, got {self.sync_frame_frequency}")
        if self.sync_segment_frame_count <= 0:
            raise ValueError(f"Sync segment frame count must be positive, got {self.sync_segment_frame_count}")
        if self.sync_stride_frames <= 0:
            raise ValueError(f"Sync stride frames must be positive, got {self.sync_stride_frames}")
        if self.sync_downsampling_factor <= 0:
            raise ValueError(f"Sync downsampling factor must be positive, got {self.sync_downsampling_factor}")
    
    @property
    def total_audio_sample_count(self) -> int:
        """
        Calculate the total number of audio samples.
        Ensures an integer number of latent representations.
        """
        try:
            return self.latent_sequence_length * self.spec_frame_frequency * self.latent_reduction_factor
        except Exception as e:
            raise RuntimeError(f"Failed to calculate total audio sample count: {e}")
    
    @property
    def latent_sequence_length(self) -> int:
        """
        Calculate the length of the latent sequence.
        Based on duration, sampling rate, spectrogram frame rate, and latent downsampling.
        """
        try:
            numerator = self.total_time_seconds * self.audio_sample_rate
            denominator = self.spec_frame_frequency * self.latent_reduction_factor
            if denominator == 0:
                raise ZeroDivisionError("Denominator in latent sequence calculation is zero")
            return int(math.ceil(numerator / denominator))
        except Exception as e:
            raise RuntimeError(f"Failed to calculate latent sequence length: {e}")
    
    @property
    def visual_sequence_length(self) -> int:
        """
        Calculate the length of the visual sequence.
        Based on duration and visual frame rate.
        """
        try:
            return int(self.total_time_seconds * self.visual_frame_frequency)
        except Exception as e:
            raise RuntimeError(f"Failed to calculate visual sequence length: {e}")
    
    @property
    def clip_sequence_length(self) -> int:
        """
        Calculate the length of the CLIP sequence.
        Based on duration and CLIP frame rate.
        """
        try:
            return int(self.total_time_seconds * self.clip_frame_frequency)
        except Exception as e:
            raise RuntimeError(f"Failed to calculate clip sequence length: {e}")
    
    @property
    def sync_sequence_length(self) -> int:
        """
        Calculate the length of the sync sequence.
        Accounts for overlapping segments and downsampling.
        """
        try:
            total_frames = self.total_time_seconds * self.sync_frame_frequency
            segment_count = (total_frames - self.sync_segment_frame_count) // self.sync_stride_frames + 1
            if segment_count < 0:
                raise ValueError(f"Invalid segment count calculation: {segment_count}")
            result = segment_count * self.sync_segment_frame_count / self.sync_downsampling_factor
            return int(result)
        except Exception as e:
            raise RuntimeError(f"Failed to calculate sync sequence length: {e}")


# Pre-configured instance for 44kHz audio processing
DEFAULT_44K_CONFIG = TemporalConfiguration(
    total_time_seconds=8.0,
    audio_sample_rate=44100,
    spec_frame_frequency=512
)


# Backward compatibility aliases
SequenceConfig = TemporalConfiguration
CONFIG_44K = DEFAULT_44K_CONFIG


def _run_validation_tests():
    """
    Internal validation function to ensure configuration correctness.
    Tests all computed properties against expected values.
    """
    test_config = DEFAULT_44K_CONFIG
    
    # Test latent sequence length
    expected_latent = 345
    actual_latent = test_config.latent_sequence_length
    assert actual_latent == expected_latent, f"Latent sequence length mismatch: expected {expected_latent}, got {actual_latent}"
    
    # Test clip sequence length
    expected_clip = 64
    actual_clip = test_config.clip_sequence_length
    assert actual_clip == expected_clip, f"Clip sequence length mismatch: expected {expected_clip}, got {actual_clip}"
    
    # Test visual sequence length
    expected_visual = 32
    actual_visual = test_config.visual_sequence_length
    assert actual_visual == expected_visual, f"Visual sequence length mismatch: expected {expected_visual}, got {actual_visual}"
    
    # Test sync sequence length
    expected_sync = 192
    actual_sync = test_config.sync_sequence_length
    assert actual_sync == expected_sync, f"Sync sequence length mismatch: expected {expected_sync}, got {actual_sync}"
    
    # Test total audio sample count
    expected_audio = 353280
    actual_audio = test_config.total_audio_sample_count
    assert actual_audio == expected_audio, f"Total audio sample count mismatch: expected {expected_audio}, got {actual_audio}"
    
    print("All validation tests passed successfully")


if __name__ == '__main__':
    _run_validation_tests()
