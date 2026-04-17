import torch
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
lib_dir = os.path.join(current_dir, 'lib')
if lib_dir not in sys.path:
    sys.path.insert(0, lib_dir)
from pathlib import Path
import time
import logging
import torchaudio
from argparse import ArgumentParser
from controlfoley.inference_utils import (ModelConfig, all_model_cfg, generate, load_video, make_video,
                                setup_eval_logging)
from lib.flow_matching import FlowMatching
from controlfoley.audio_model import AudioGenerationNetwork, create_audio_generation_model
from controlfoley.feature_extractor import FeaturesUtils
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
log = logging.getLogger()

@torch.inference_mode()
def main():
    setup_eval_logging()
    parser = ArgumentParser()
    parser.add_argument('--variant',
                        type=str,
                        default='large_44k',
                        help='large_44k')
    parser.add_argument('--video', type=Path, help='Path to the video file')
    parser.add_argument('--audio', type=Path, help='Path to the audio file (2s-4s)')
    parser.add_argument('--prompt', type=str, help='Input prompt', default='')
    parser.add_argument('--negative_prompt', type=str, help='Negative prompt', default='')
    parser.add_argument('--duration', type=float, default=8.0)
    parser.add_argument('--cfg_strength', type=float, default=4.5)
    parser.add_argument('--num_steps', type=int, default=25)
    parser.add_argument('--output', type=Path, help='Output directory', default='./output')
    parser.add_argument('--seed', type=int, help='Random seed', default=42)
    parser.add_argument('--skip_video_composite', action='store_true')
    parser.add_argument('--mask_away_clip', action='store_true')
    args = parser.parse_args()
    if args.variant not in all_model_cfg:
        raise ValueError(f'Unknown model variant: {args.variant}')
    model: ModelConfig = all_model_cfg[args.variant]
    seq_cfg = model.seq_cfg

    if args.video:
        video_path: Path = Path(args.video).expanduser()
    else:
        video_path = None
    if args.audio:
        audio_path: Path = Path(args.audio).expanduser()
    else:
        audio_path = None

    prompt: str = args.prompt
    negative_prompt: str = args.negative_prompt
    output_dir: str = args.output.expanduser()
    seed: int = args.seed
    num_steps: int = args.num_steps
    duration: float = args.duration
    cfg_strength: float = args.cfg_strength
    skip_video_composite: bool = args.skip_video_composite
    mask_away_clip: bool = args.mask_away_clip

    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda'
    elif torch.backends.mps.is_available():
        device = 'mps'
    else:
        log.warning('CUDA/MPS are not available, running on CPU')
    dtype = torch.float32 
    output_dir.mkdir(parents=True, exist_ok=True)

    # load a pretrained model
    net: AudioGenerationNetwork = create_audio_generation_model(model.model_name).to(device, dtype).eval()
    net.load_weights(torch.load(model.model_path, map_location=device, weights_only=True))
    log.info(f'Loaded weights from {model.model_path}')

    # misc setup
    rng = torch.Generator(device=device)
    rng.manual_seed(seed)
    fm = FlowMatching(min_sigma=0, inference_mode='euler', num_steps=num_steps)


    feature_utils = FeaturesUtils(tod_vae_ckpt="./model_weights/ext_weights/v1-44.pth",
                                  synchformer_ckpt="./model_weights/ext_weights/synchformer_state_dict.pth",
                                  cav_mae_ckpt="./model_weights/ext_weights/cav_mae_st.pth",
                                  clap_ckpt="./model_weights/ext_weights/music_speech_audioset_epoch_15_esc_89.98.pt",
                                  mode=model.mode,
                                  enable_conditions=True,
                                  need_vae_encoder=False)
    feature_utils = feature_utils.to(device, dtype).eval()

    if video_path is not None:
        log.info(f'Using video {video_path}')
        video_info = load_video(video_path, duration)
        clip_frames = video_info.clip_embeddings
        visual_frames = video_info.visual_features
        sync_frames = video_info.sync_embeddings
        if video_info.total_duration < duration:
            log.info(f'Video duration {video_info.total_duration} is shorter than {duration}, using the whole video')
            duration = video_info.total_duration
        if mask_away_clip:
            clip_frames = None
        else:
            clip_frames = clip_frames.unsqueeze(0)
        sync_frames = sync_frames.unsqueeze(0)
        visual_frames = visual_frames.unsqueeze(0)
    else:
        log.info('No video provided -- text-to-audio mode')
        clip_frames = visual_frames = sync_frames = None

    if audio_path is not None:
        log.info(f'Using audio {audio_path}')
        audio_frames, sampling_rate = torchaudio.load(audio_path)
        audio_frames = audio_frames.to(device, dtype)
        timbre_frames = audio_frames
        if sampling_rate != 16000:
            log.warning(f'Audio sampling rate is not {16000} Hz, resampling')
            audio_frames = torchaudio.functional.resample(audio_frames, sampling_rate, 16000)
        audio_frames = audio_frames.mean(dim=0, keepdim=True)
        audio_frames = audio_frames.reshape(1, -1)
        audio_frames = audio_frames.unsqueeze(0)

        if sampling_rate != 32000:
            log.warning(f'Audio sampling rate is not {32000} Hz, resampling')
            timbre_frames = torchaudio.functional.resample(timbre_frames, sampling_rate, 32000)
        # Limit duration to 2-4 seconds
        target_sr = 32000  # Target sampling rate
        min_length = 2 * target_sr  # Number of samples for 2 seconds
        max_length = 4 * target_sr  # Number of samples for 4 seconds
        if timbre_frames.dim() == 2:
            # [channels, samples]
            num_samples = timbre_frames.shape[-1]
            channels = timbre_frames.shape[0]
        elif timbre_frames.dim() == 3:
            # [batch, channels, samples]
            num_samples = timbre_frames.shape[-1]
            channels = timbre_frames.shape[1]
            batch_size = timbre_frames.shape[0]
        else:
            raise ValueError(f"Unexpected audio tensor shape: {timbre_frames.shape}")

        # Process audio length
        if num_samples < min_length:
            # Less than 2 seconds, pad with zeros
            padding_length = min_length - num_samples
            if timbre_frames.dim() == 2:
                # [channels, samples] -> pad on the last dimension
                timbre_frames = torch.nn.functional.pad(timbre_frames, (0, padding_length), mode='constant', value=0)
            else:
                # [batch, channels, samples] -> pad on the last dimension
                timbre_frames = torch.nn.functional.pad(timbre_frames, (0, padding_length), mode='constant', value=0)
            log.info(f"Audio too short ({num_samples/target_sr:.2f}s), padded to {min_length/target_sr:.2f}s")

        elif num_samples > max_length:
            # Greater than 4 seconds, truncate to 4 seconds
            timbre_frames = timbre_frames[..., :max_length]
            log.info(f"Audio too long ({num_samples/target_sr:.2f}s), truncated to {max_length/target_sr:.2f}s")
        else:
            log.info(f"Audio length is within range: {num_samples/target_sr:.2f}s")

        # Ensure audio length is 2-4 seconds
        num_samples = timbre_frames.shape[-1]
        timbre_dutation = num_samples/target_sr
        assert min_length <= num_samples <= max_length, f"Audio length should be between 2-4s, got {num_samples/target_sr:.2f}s"
        timbre_frames = timbre_frames.mean(dim=0, keepdim=True)
        timbre_frames = timbre_frames.reshape(1, -1)
        timbre_frames = timbre_frames.unsqueeze(0)
    else:
        log.info('No audio provided -- video/text-to-video mode')
        audio_frames = None
        timbre_frames = None
        timbre_dutation = 0.0

    seq_cfg.total_time_seconds = duration
    net.update_seq_lengths(seq_cfg.latent_sequence_length, seq_cfg.clip_sequence_length, seq_cfg.visual_sequence_length, seq_cfg.sync_sequence_length)

    log.info(f'Prompt: {prompt}')
    log.info(f'Negative prompt: {negative_prompt}')

    start_time = time.time()
    audios = generate(clip_frames,
                      visual_frames,
                      sync_frames, 
                      audio_frames,
                      timbre_frames,
                      timbre_dutation,
                      [prompt],
                      negative_text=[negative_prompt],
                      feature_utils=feature_utils,
                      net=net,
                      fm=fm,
                      rng=rng,
                      cfg_strength=cfg_strength)
    audio = audios.float().cpu()[0]
    end_time = time.time()
    print("time: {:.2f}s".format(end_time - start_time))
    if video_path is not None:
        save_path = output_dir / f'{video_path.stem}.flac'
    else:
        safe_filename = prompt.replace(' ', '_').replace('/', '_').replace('.', '')
        save_path = output_dir / f'{safe_filename}.flac'
    torchaudio.save(save_path, audio, seq_cfg.audio_sample_rate)

    log.info(f'Audio saved to {save_path}')
    if video_path is not None and not skip_video_composite:
        video_save_path = output_dir / f'{video_path.stem}.mp4'
        make_video(video_info, video_save_path, audio, sampling_rate=seq_cfg.audio_sample_rate)
        log.info(f'Video saved to {output_dir / video_save_path}')

    log.info('Memory usage: %.2f GB', torch.cuda.max_memory_allocated() / (2**30))

if __name__ == '__main__':
    main()

