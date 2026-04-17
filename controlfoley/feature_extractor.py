import os
import sys
import json
from typing import Literal, Optional
import open_clip
import laion_clap
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from open_clip import create_model_from_pretrained
from torchvision.transforms import Normalize
from lib.autoencoder import AutoEncoderModule
from lib.mel_converter import get_mel_converter
from lib.synchformer import Synchformer
from lib.distributions import DiagonalGaussianDistribution
import lib.cav_mae_st.core.models as models
import julius
import typing as tp
import sys
from pathlib import Path

current_file = Path(__file__).resolve()
audiocraft_path = current_file.parent.parent / 'lib' / 'audiocraft'
sys.path.insert(0, str(audiocraft_path))

from lib.audiocraft.audiocraft.models import MusicGen

def convert_audio_channels(wav: torch.Tensor, channels: int = 2) -> torch.Tensor:
    """Convert audio to the given number of channels.

    Args:
        wav (torch.Tensor): Audio wave of shape [B, C, T].
        channels (int): Expected number of channels as output.
    Returns:
        torch.Tensor: Downmixed or unchanged audio wave [B, C, T].
    """
    *shape, src_channels, length = wav.shape
    if src_channels == channels:
        pass
    elif channels == 1:
        # Case 1:
        # The caller asked 1-channel audio, and the stream has multiple
        # channels, downmix all channels.
        wav = wav.mean(dim=-2, keepdim=True)
    elif src_channels == 1:
        # Case 2:
        # The caller asked for multiple channels, but the input file has
        # a single channel, replicate the audio over all channels.
        wav = wav.expand(*shape, channels, length)
    elif src_channels >= channels:
        # Case 3:
        # The caller asked for multiple channels, and the input file has
        # more channels than requested. In that case return the first channels.
        wav = wav[..., :channels, :]
    else:
        # Case 4: What is a reasonable choice here?
        raise ValueError('The audio file has less channels than requested but is not mono.')
    return wav
    
def convert_audio(wav: torch.Tensor, from_rate: float,
                  to_rate: float, to_channels: int) -> torch.Tensor:
    """Convert audio to new sample rate and number of audio channels."""
    wav = julius.resample_frac(wav, int(from_rate), int(to_rate))
    wav = convert_audio_channels(wav, to_channels)
    return wav

def patch_clip(clip_model):
    # a hack to make it output last hidden states
    # https://github.com/mlfoundations/open_clip/blob/fc5a37b72d705f760ebbc7915b84729816ed471f/src/open_clip/model.py#L269
    def new_encode_text(self, text, normalize: bool = False):
        cast_dtype = self.transformer.get_cast_dtype()

        x = self.token_embedding(text).to(cast_dtype)  # [batch_size, n_ctx, d_model]

        x = x + self.positional_embedding.to(cast_dtype)
        x = self.transformer(x, attn_mask=self.attn_mask)
        x = self.ln_final(x)  # [batch_size, n_ctx, transformer.width]
        return F.normalize(x, dim=-1) if normalize else x

    clip_model.encode_text = new_encode_text.__get__(clip_model)
    return clip_model

class FeaturesUtils(nn.Module):

    def __init__(
        self,
        *,
        tod_vae_ckpt: Optional[str] = None,
        bigvgan_vocoder_ckpt: Optional[str] = None,
        synchformer_ckpt: Optional[str] = None,
        cav_mae_ckpt: Optional[str] = None,
        clap_ckpt: Optional[str] = None,
        mode=Literal['16k', '44k'],
        need_vae_encoder: bool = True,
        enable_conditions: bool = True,
    ):
        super().__init__()

        if enable_conditions:
            self.clip_model = create_model_from_pretrained('hf-hub:apple/DFN5B-CLIP-ViT-H-14-384',
                                                           return_transform=False)
            self.clip_preprocess = Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                             std=[0.26862954, 0.26130258, 0.27577711])
            self.clip_model = patch_clip(self.clip_model)

            self.synchformer = Synchformer()
            self.synchformer.load_state_dict(
                torch.load(synchformer_ckpt, weights_only=True, map_location='cpu'))
            mdl_weight = torch.load(cav_mae_ckpt, map_location=torch.device('cpu'))
            self.cav_mae = models.CAVMAEST(audio_length=208, norm_pix_loss=False, modality_specific_depth=11, tr_pos=False)
            miss, unexpected = self.cav_mae.load_state_dict(mdl_weight, strict=False)
            self.tokenizer = open_clip.get_tokenizer('ViT-H-14-378-quickgelu')  # same as 'ViT-H-14'
            if clap_ckpt is not None:
                self.clap_model = laion_clap.CLAP_Module(enable_fusion=False, amodel= 'HTSAT-base')
                self.clap_model.load_ckpt(clap_ckpt)
            else:
                self.clap_model = None
            self.music_model = MusicGen.get_pretrained('facebook/musicgen-style')
        else:
            self.clip_model = None
            self.clap_model = None
            self.synchformer = None
            self.tokenizer = None
            self.cav_mae = None
            self.music_model = None
        if tod_vae_ckpt is not None:
            self.mel_converter = get_mel_converter(mode)
            self.tod = AutoEncoderModule(vae_ckpt_path=tod_vae_ckpt,
                                         vocoder_ckpt_path=bigvgan_vocoder_ckpt,
                                         mode=mode,
                                         need_vae_encoder=need_vae_encoder)
        else:
            self.tod = None

    def compile(self):
        if self.clip_model is not None:
            self.clip_model.encode_image = torch.compile(self.clip_model.encode_image)
            self.clip_model.encode_text = torch.compile(self.clip_model.encode_text)
        if self.clap_model is not None:
            self.clap_model = torch.compile(self.clap_model)
        if self.cav_mae is not None:
            self.cav_mae = torch.compile(self.cav_mae)
        if self.synchformer is not None:
            self.synchformer = torch.compile(self.synchformer)


    def train(self, mode: bool) -> None:
        return super().train(False)

    @torch.inference_mode()
    def encode_audio_with_clap(self, clap_audio: torch.Tensor, batch_size: int = -1) -> torch.Tensor:
        assert self.clap_model is not None, 'CLAP is not loaded'
        x = self.clap_model.get_audio_embedding_from_data(x = clap_audio, use_tensor = True)
        return x

    @torch.inference_mode()
    def encode_audio_with_music_model(self, timbre_audio: torch.Tensor, duration: float = 3.0, batch_size: int = -1) -> torch.Tensor:
        assert self.music_model is not None, 'music_model is not loaded'
        self.music_model.set_style_conditioner_params(eval_q=1,excerpt_length=duration,)
        # timbre_audio [1,len] sr=32k
        timbre_audio_list = []
        timbre_audio_list.append(timbre_audio)
        attributes, prompt_tokens = self.music_model._prepare_tokens_and_attributes(descriptions=[None], prompt=None, melody_wavs=timbre_audio_list)
        tokens = self.music_model._generate_tokens(attributes, prompt_tokens, False) #[1,5*duration,1536]
        return tokens

    @torch.inference_mode()
    def encode_video_with_clip(self, x: torch.Tensor, batch_size: int = -1) -> torch.Tensor:
        assert self.clip_model is not None, 'CLIP is not loaded'
        # x: (B, T, C, H, W) H/W: 384
        b, t, c, h, w = x.shape
        assert c == 3 and h == 384 and w == 384
        x = self.clip_preprocess(x)
        x = rearrange(x, 'b t c h w -> (b t) c h w')
        outputs = []
        if batch_size < 0:
            batch_size = b * t
        for i in range(0, b * t, batch_size):
            outputs.append(self.clip_model.encode_image(x[i:i + batch_size], normalize=True))
        x = torch.cat(outputs, dim=0)
        # x = self.clip_model.encode_image(x, normalize=True)
        x = rearrange(x, '(b t) d -> b t d', b=b)
        return x

    @torch.inference_mode()
    def encode_video_with_cav_mae(self, v_in: torch.Tensor, batch_size: int = -1) -> torch.Tensor:
        for i in range(v_in.size(0)):
            v_input = v_in[i]
            v_input = v_input.unsqueeze(0).cuda()
            v =  self.cav_mae.forward_feat_v(v_input)
            v = v.unsqueeze(1)
            if i == 0:
                v_features = v
            else:
                v_features = torch.cat((v_features, v), dim=1)
        return v_features

    @torch.inference_mode()
    def encode_video_with_sync(self, x: torch.Tensor, batch_size: int = -1) -> torch.Tensor:
        assert self.synchformer is not None, 'Synchformer is not loaded'
        # x: (B, T, C, H, W) H/W: 384

        b, t, c, h, w = x.shape
        assert c == 3 and h == 224 and w == 224
        # partition the video
        segment_size = 16
        step_size = 8
        num_segments = (t - segment_size) // step_size + 1
        segments = []
        for i in range(num_segments):
            segments.append(x[:, i * step_size:i * step_size + segment_size])
        x = torch.stack(segments, dim=1)  # (B, S, T, C, H, W)
        outputs = []
        if batch_size < 0:
            batch_size = b
        x = rearrange(x, 'b s t c h w -> (b s) 1 t c h w')
        for i in range(0, b * num_segments, batch_size):
            outputs.append(self.synchformer(x[i:i + batch_size]))
        x = torch.cat(outputs, dim=0)
        x = rearrange(x, '(b s) 1 t d -> b (s t) d', b=b)
        return x

    @torch.inference_mode()
    def encode_text(self, text: list[str]) -> torch.Tensor:
        assert self.clip_model is not None, 'CLIP is not loaded'
        assert self.tokenizer is not None, 'Tokenizer is not loaded'
        # x: (B, L)
        tokens = self.tokenizer(text).to(self.device)
        return self.clip_model.encode_text(tokens, normalize=True)

    @torch.inference_mode()
    def encode_audio(self, x) -> DiagonalGaussianDistribution:
        assert self.tod is not None, 'VAE is not loaded'
        # x: (B * L)
        mel = self.mel_converter(x)
        dist = self.tod.encode(mel)
        return dist

    @torch.inference_mode()
    def vocode(self, mel: torch.Tensor) -> torch.Tensor:
        assert self.tod is not None, 'VAE is not loaded'
        return self.tod.vocode(mel)

    @torch.inference_mode()
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        assert self.tod is not None, 'VAE is not loaded'
        return self.tod.decode(z.transpose(1, 2))

    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def dtype(self):
        return next(self.parameters()).dtype
