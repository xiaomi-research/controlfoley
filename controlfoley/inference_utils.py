import dataclasses
import logging
from pathlib import Path
from typing import Optional
import numpy as np
import torch
from colorlog import ColoredFormatter
from PIL import Image
from torchvision.transforms import v2
from controlfoley.media_utils import ImageData, MediaClipData, extract_video_segments, encode_media_with_sound
from lib.flow_matching import FlowMatching
from controlfoley.audio_model import AudioGenerationNetwork
from controlfoley.temporal_config import DEFAULT_44K_CONFIG, TemporalConfiguration
from controlfoley.feature_extractor import FeaturesUtils


log = logging.getLogger()

@dataclasses.dataclass
class ModelConfig:
    model_name: str
    model_path: Path
    mode: str

    @property
    def seq_cfg(self) -> TemporalConfiguration:
        if self.mode == '44k':
            return DEFAULT_44K_CONFIG


large_44k = ModelConfig(model_name='large_44k', model_path=Path('./model_weights/weights/controlfoley.pth'), mode='44k')

all_model_cfg: dict[str, ModelConfig] = {
    'large_44k': large_44k,
}


def generate(
    clip_video: Optional[torch.Tensor],
    visual_video: Optional[torch.Tensor],
    sync_video: Optional[torch.Tensor],
    audio_frames: Optional[torch.Tensor],
    timbre_frames: Optional[torch.Tensor],
    timbre_dutation: float,
    text: Optional[list[str]],
    *,
    negative_text: Optional[list[str]] = None,
    feature_utils: FeaturesUtils,
    net: AudioGenerationNetwork,
    fm: FlowMatching,
    rng: torch.Generator,
    cfg_strength: float,
    clip_batch_size_multiplier: int = 40,
    sync_batch_size_multiplier: int = 40,
    image_input: bool = False,
) -> torch.Tensor:
    device = feature_utils.device
    dtype = feature_utils.dtype

    bs = len(text)
    
    if clip_video is not None:
        clip_video = clip_video.to(device, dtype, non_blocking=True)
        clip_features = feature_utils.encode_video_with_clip(clip_video,
                                                             batch_size=bs *
                                                             clip_batch_size_multiplier)
        if image_input:
            clip_features = clip_features.expand(-1, net.clip_seq_len, -1)
    else:
        clip_features = net.get_empty_clip_sequence(bs)


    if visual_video is not None:
        visual_video = visual_video.to(device, dtype, non_blocking=True)
        for i in range(visual_video.size(0)):
            tmp_v = visual_video[i]
            temp_features_v = feature_utils.encode_video_with_cav_mae(tmp_v)
        if i == 0:
            visual_features = temp_features_v
        else:
            visual_features = torch.cat([visual_features, temp_features_v], dim=0)    
        visual_features = torch.mean(visual_features, dim=2)
    else:
        visual_features = net.get_empty_visual_sequence(bs)
    visual_features = visual_features.cuda(non_blocking=True)

    if audio_frames is not None:
        audio_frames = audio_frames.squeeze(1)
        audio_frames = audio_frames.to(device, dtype, non_blocking=True)
        audio_features = feature_utils.encode_audio_with_clap(audio_frames)
        audio_features = audio_features.unsqueeze(1)
    else:
        audio_features = net.get_empty_audio_sequence(bs)  


    if timbre_frames is not None:
        for i in range(timbre_frames.size(0)):
            temp_a = timbre_frames[i]
            timbre_feature_tmp = feature_utils.encode_audio_with_music_model(temp_a,timbre_dutation)
            timbre_feature_tmp = torch.mean(timbre_feature_tmp, dim=1).unsqueeze(1)
            if i == 0:
                timbre_feature = timbre_feature_tmp
            else:
                timbre_feature = torch.cat([timbre_feature, timbre_feature_tmp], dim=0)
        timbre_feature = timbre_feature.float()
    else:
        timbre_feature = net.get_empty_timbre_sequence(bs)   

    if sync_video is not None and not image_input:
        sync_video = sync_video.to(device, dtype, non_blocking=True)
        sync_features = feature_utils.encode_video_with_sync(sync_video,
                                                             batch_size=bs *
                                                             sync_batch_size_multiplier)
    else:
        sync_features = net.get_empty_sync_sequence(bs)

    if text is not None:
        text_features = feature_utils.encode_text(text)
    else:
        text_features = net.get_empty_string_sequence(bs)

    if negative_text is not None:
        assert len(negative_text) == bs
        negative_text_features = feature_utils.encode_text(negative_text)
    else:
        negative_text_features = net.get_empty_string_sequence(bs)
    
    x0 = torch.randn(bs,
                     net.latent_seq_len,
                     net.latent_dim,
                     device=device,
                     dtype=dtype,
                     generator=rng)
    preprocessed_conditions = net.preprocess_conditions(clip_features, visual_features, sync_features, text_features, audio_features, timbre_feature)

    empty_conditions = net.get_empty_conditions(
        bs, negative_text_features=negative_text_features if negative_text is not None else None)

    cfg_ode_wrapper = lambda t, x: net.ode_wrapper(t, x, preprocessed_conditions, empty_conditions,
                                                   cfg_strength)
    x1 = fm.to_data(cfg_ode_wrapper, x0)
    x1 = net.unnormalize(x1)
    spec = feature_utils.decode(x1)
    audio = feature_utils.vocode(spec)
    return audio


LOGFORMAT = "[%(log_color)s%(levelname)-8s%(reset)s]: %(log_color)s%(message)s%(reset)s"


def setup_eval_logging(log_level: int = logging.INFO):
    logging.root.setLevel(log_level)
    formatter = ColoredFormatter(LOGFORMAT)
    stream = logging.StreamHandler()
    stream.setLevel(log_level)
    stream.setFormatter(formatter)
    log = logging.getLogger()
    log.setLevel(log_level)
    log.addHandler(stream)


_CLIP_SIZE = 384
_CLIP_FPS = 8.0

_VISUAL_SIZE = 224
_VISUAL_FPS = 4.0

_SYNC_SIZE = 224
_SYNC_FPS = 25.0


def load_video(video_path: Path, duration_sec: float, load_all_frames: bool = True) -> MediaClipData:

    clip_transform = v2.Compose([
        v2.Resize((_CLIP_SIZE, _CLIP_SIZE), interpolation=v2.InterpolationMode.BICUBIC),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
    ])

    image_transform = v2.Compose([
        v2.Resize(_VISUAL_SIZE, interpolation=v2.InterpolationMode.BICUBIC),
        v2.CenterCrop(_VISUAL_SIZE),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.4850, 0.4560, 0.4060],std=[0.2290, 0.2240, 0.2250]),    
    ])

    sync_transform = v2.Compose([
        v2.Resize(_SYNC_SIZE, interpolation=v2.InterpolationMode.BICUBIC),
        v2.CenterCrop(_SYNC_SIZE),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    output_frames, all_frames, orig_fps = extract_video_segments(video_path,
                                                      target_fps_list=[_CLIP_FPS, _VISUAL_FPS, _SYNC_FPS],
                                                      start_time=0,
                                                      end_time=duration_sec,
                                                      extract_all_frames=load_all_frames)

    clip_chunk, visual_chunk, sync_chunk = output_frames
    clip_chunk = torch.from_numpy(clip_chunk).permute(0, 3, 1, 2)
    visual_chunk = torch.from_numpy(visual_chunk).permute(0, 3, 1, 2)
    sync_chunk = torch.from_numpy(sync_chunk).permute(0, 3, 1, 2)

    clip_frames = clip_transform(clip_chunk)
    visual_frames = image_transform(visual_chunk)
    sync_frames = sync_transform(sync_chunk)

    clip_length_sec = clip_frames.shape[0] / _CLIP_FPS
    visual_length_sec = visual_frames.shape[0] / _VISUAL_FPS
    sync_length_sec = sync_frames.shape[0] / _SYNC_FPS

    if clip_length_sec < duration_sec:
        log.warning(f'Clip video is too short: {clip_length_sec:.2f} < {duration_sec:.2f}')
        log.warning(f'Truncating to {clip_length_sec:.2f} sec')
        duration_sec = clip_length_sec

    if visual_length_sec < duration_sec:
        log.warning(f'visual video is too short: {visual_length_sec:.2f} < {duration_sec:.2f}')
        log.warning(f'Truncating to {visual_length_sec:.2f} sec')
        duration_sec = visual_length_sec

    if sync_length_sec < duration_sec:
        log.warning(f'Sync video is too short: {sync_length_sec:.2f} < {duration_sec:.2f}')
        log.warning(f'Truncating to {sync_length_sec:.2f} sec')
        duration_sec = sync_length_sec

    clip_frames = clip_frames[:int(_CLIP_FPS * duration_sec)]
    visual_frames = visual_frames[:int(_VISUAL_FPS * duration_sec)]
    sync_frames = sync_frames[:int(_SYNC_FPS * duration_sec)]

    video_info = MediaClipData(
        total_duration=duration_sec,
        frame_rate=orig_fps,
        clip_embeddings=clip_frames,
        visual_features=visual_frames,
        sync_embeddings=sync_frames,
        frame_sequence=all_frames if load_all_frames else None,
    )
    return video_info


def load_image(image_path: Path) -> MediaClipData:
    clip_transform = v2.Compose([
        v2.Resize((_CLIP_SIZE, _CLIP_SIZE), interpolation=v2.InterpolationMode.BICUBIC),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
    ])

    sync_transform = v2.Compose([
        v2.Resize(_SYNC_SIZE, interpolation=v2.InterpolationMode.BICUBIC),
        v2.CenterCrop(_SYNC_SIZE),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    frame = np.array(Image.open(image_path))

    clip_chunk = torch.from_numpy(frame).unsqueeze(0).permute(0, 3, 1, 2)
    sync_chunk = torch.from_numpy(frame).unsqueeze(0).permute(0, 3, 1, 2)

    clip_frames = clip_transform(clip_chunk)
    sync_frames = sync_transform(sync_chunk)

    image_data = ImageData(
        clip_embeddings=clip_frames,
        sync_embeddings=sync_frames,
        original_frame=frame,
    )
    # Create a default MediaClipData using ImageData
    return MediaClipData.create_from_image_data(image_data, duration_seconds=1.0, frame_rate=Fraction(1, 1))


def make_video(video_info: MediaClipData, output_path: Path, audio: torch.Tensor, sampling_rate: int):
    encode_media_with_sound(video_info, output_path, audio, sampling_rate)

