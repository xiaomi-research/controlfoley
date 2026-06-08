import av
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Optional, List
import numpy as np
import torch
from av import AudioFrame


@dataclass
class MediaClipData:
    total_duration: float
    frame_rate: Fraction
    clip_embeddings: torch.Tensor
    visual_features: torch.Tensor
    sync_embeddings: torch.Tensor
    frame_sequence: Optional[List[np.ndarray]]

    @property
    def frame_height(self):
        if not self.frame_sequence or len(self.frame_sequence) == 0:
            raise ValueError("No frames available in frame_sequence")
        return self.frame_sequence[0].shape[0]

    @property
    def frame_width(self):
        if not self.frame_sequence or len(self.frame_sequence) == 0:
            raise ValueError("No frames available in frame_sequence")
        return self.frame_sequence[0].shape[1]

    @classmethod
    def create_from_image_data(cls, image_data: 'ImageData', duration_seconds: float,
                              frame_rate: Fraction) -> 'MediaClipData':
        if not image_data or not hasattr(image_data, 'original_frame'):
            raise ValueError("Invalid image_data provided")
            
        frame_count = int(duration_seconds * frame_rate)
        frame_list = [image_data.original_frame] * frame_count
        
        return cls(total_duration=duration_seconds,
                   frame_rate=frame_rate,
                   clip_embeddings=image_data.clip_embeddings,
                   visual_features=image_data.visual_features,
                   sync_embeddings=image_data.sync_embeddings,
                   frame_sequence=frame_list)


@dataclass
class ImageData:
    clip_embeddings: torch.Tensor
    visual_features: torch.Tensor
    sync_embeddings: torch.Tensor
    original_frame: Optional[np.ndarray]

    @property
    def frame_height(self):
        if self.original_frame is None:
            raise ValueError("Original frame is not available")
        return self.original_frame.shape[0]

    @property
    def frame_width(self):
        if self.original_frame is None:
            raise ValueError("Original frame is not available")
        return self.original_frame.shape[1]


def extract_video_segments(video_file: Path, target_fps_list: List[float], 
                          start_time: float, end_time: float,
                          extract_all_frames: bool) -> tuple[List[np.ndarray], List[np.ndarray], Fraction]:
    """
    Extract video frames at specified FPS rates with safety checks
    """
    if not video_file.exists():
        raise FileNotFoundError(f"Video file not found: {video_file}")
    
    if start_time < 0 or end_time <= start_time:
        raise ValueError("Invalid time range: start_time must be >= 0 and end_time > start_time")
    
    if not target_fps_list:
        raise ValueError("Target FPS list cannot be empty")
    
    extracted_frames_per_fps = [[] for _ in target_fps_list]
    next_frame_timestamps = [0.0 for _ in target_fps_list]
    frame_intervals = [1.0 / fps for fps in target_fps_list]
    complete_frame_list = []

    try:
        with av.open(str(video_file)) as video_container:
            video_stream = video_container.streams.video[0]
            if not video_stream:
                raise ValueError("No video stream found in the file")
                
            # detected_fps = video_stream.guessed_rate
            detected_fps = video_stream.average_rate or video_stream.guessed_rate
            video_stream.thread_type = 'AUTO'
            
            for data_packet in video_container.demux(video_stream):
                for video_frame in data_packet.decode():
                    current_time = video_frame.time
                    
                    if current_time < start_time:
                        continue
                    if current_time > end_time:
                        break

                    frame_array = None
                    if extract_all_frames:
                        try:
                            frame_array = video_frame.to_ndarray(format='rgb24')
                            complete_frame_list.append(frame_array)
                        except Exception as e:
                            raise RuntimeError(f"Failed to convert frame to numpy array: {e}")

                    for idx, _ in enumerate(target_fps_list):
                        current_timestamp = current_time
                        while current_timestamp >= next_frame_timestamps[idx]:
                            if frame_array is None:
                                try:
                                    frame_array = video_frame.to_ndarray(format='rgb24')
                                except Exception as e:
                                    raise RuntimeError(f"Failed to convert frame to numpy array: {e}")

                            extracted_frames_per_fps[idx].append(frame_array)
                            next_frame_timestamps[idx] += frame_intervals[idx]
    
    except Exception as e:
        raise RuntimeError(f"Error processing video file {video_file}: {e}")

    processed_frames = [np.stack(frame_group) for frame_group in extracted_frames_per_fps]
    return processed_frames, complete_frame_list, detected_fps


def encode_media_with_sound(media_data: MediaClipData, output_file: Path,
                            audio_data: torch.Tensor, sample_rate: int):
    """Encode video frames + audio tensor into mp4. fps/pts 全部显式钉死。"""
    if not media_data or not media_data.frame_sequence:
        raise ValueError("Invalid media data or empty frame sequence")
    if audio_data is None or audio_data.numel() == 0:
        raise ValueError("Invalid audio data provided")
    if sample_rate <= 0:
        raise ValueError("Sample rate must be positive")

    # 把 frame_rate 规范化成 Fraction
    fps = Fraction(media_data.frame_rate).limit_denominator(60000)
    if fps <= 0:
        raise ValueError(f"Invalid fps: {fps}")

    # 用 90000 作 mp4 video timescale —— mp4 muxer 不会再改写
    VIDEO_TIMESCALE = 90000
    # 每帧在 timescale 下的 tick 数；为了避免浮点误差，用 Fraction 精算
    ticks_per_frame_frac = Fraction(VIDEO_TIMESCALE) * Fraction(fps.denominator, fps.numerator)
    # 必须是整数，否则用更大的 timescale。常见 fps 都能整除：
    # 24→3750, 25→3600, 30→3000, 50→1800, 60→1500, 23.976(24000/1001)→3753.75 ✗
    if ticks_per_frame_frac.denominator != 1:
        # 23.976 这类非整数：直接用 1/fps 作 stream time_base，pts=帧号
        VIDEO_TIMESCALE = fps.numerator
        stream_tb = Fraction(1, fps.numerator)
        ticks_per_frame = fps.denominator
    else:
        stream_tb = Fraction(1, VIDEO_TIMESCALE)
        ticks_per_frame = int(ticks_per_frame_frac)

    output_container = av.open(str(output_file), 'w')
    try:
        # 视频流：跳过 add_stream(codec, rate) 糖，自己设
        video_stream = output_container.add_stream('h264')
        video_stream.width = media_data.frame_width
        video_stream.height = media_data.frame_height
        video_stream.pix_fmt = 'yuv420p'
        video_stream.time_base = stream_tb
        # codec_context 上同时钉 framerate + time_base
        video_stream.codec_context.framerate = fps
        video_stream.codec_context.time_base = stream_tb
        video_stream.codec_context.bit_rate = 10 * 1_000_000

        audio_stream = output_container.add_stream('aac', sample_rate)

        # 每帧 pts 用 stream timescale 的整数 tick
        for i, image_frame in enumerate(media_data.frame_sequence):
            av_frame = av.VideoFrame.from_ndarray(image_frame, format='rgb24')
            av_frame.pts = i * ticks_per_frame
            av_frame.time_base = stream_tb
            for pkt in video_stream.encode(av_frame):
                output_container.mux(pkt)
        for pkt in video_stream.encode():
            output_container.mux(pkt)

        audio_numpy = audio_data.numpy().astype(np.float32)
        if audio_numpy.ndim == 1:
            audio_numpy = audio_numpy[np.newaxis, :]
        audio_frame = AudioFrame.from_ndarray(audio_numpy, format='flt', layout='mono')
        audio_frame.sample_rate = sample_rate
        for pkt in audio_stream.encode(audio_frame):
            output_container.mux(pkt)
        for pkt in audio_stream.encode():
            output_container.mux(pkt)
    finally:
        output_container.close()
    # try:
    #     output_container = av.open(str(output_file), 'w')
        
    #     # Configure video stream
    #     video_stream = output_container.add_stream('h264', media_data.frame_rate)
    #     video_stream.codec_context.bit_rate = 10 * 1000000  # 10 Mbps
    #     video_stream.width = media_data.frame_width
    #     video_stream.height = media_data.frame_height
    #     video_stream.pix_fmt = 'yuv420p'

    #     # Configure audio stream
    #     audio_stream = output_container.add_stream('aac', sample_rate)

    #     # Encode video frames
    #     for image_frame in media_data.frame_sequence:
    #         try:
    #             av_frame = av.VideoFrame.from_ndarray(image_frame)
    #             encoded_packet = video_stream.encode(av_frame)
    #             output_container.mux(encoded_packet)
    #         except Exception as e:
    #             raise RuntimeError(f"Failed to encode video frame: {e}")

    #     # Finalize video encoding
    #     for final_packet in video_stream.encode():
    #         output_container.mux(final_packet)

    #     # Convert and encode audio
    #     try:
    #         audio_numpy = audio_data.numpy().astype(np.float32)
    #         audio_frame = AudioFrame.from_ndarray(audio_numpy, format='flt', layout='mono')
    #         audio_frame.sample_rate = sample_rate

    #         for audio_packet in audio_stream.encode(audio_frame):
    #             output_container.mux(audio_packet)

    #         for final_audio_packet in audio_stream.encode():
    #             output_container.mux(final_audio_packet)
    #     except Exception as e:
    #         raise RuntimeError(f"Failed to encode audio: {e}")

    #     output_container.close()
        
    # except Exception as e:
    #     if 'output_container' in locals():
    #         output_container.close()
    #     raise RuntimeError(f"Error during media encoding: {e}")


def remux_video_with_audio(input_video: Path, audio_tensor: torch.Tensor, 
                          output_path: Path, audio_sample_rate: int):
    """
    NOTE: This function is kept for reference but not used due to duration accuracy issues
    without re-encoding. It remuxes video with new audio.
    """
    if not input_video.exists():
        raise FileNotFoundError(f"Input video not found: {input_video}")
    
    if audio_tensor is None or audio_tensor.numel() == 0:
        raise ValueError("Invalid audio tensor provided")
    
    try:
        source_video = av.open(str(input_video))
        output_container = av.open(str(output_path), 'w')
        
        input_video_stream = source_video.streams.video[0]
        output_video_stream = output_container.add_stream(template=input_video_stream)
        output_audio_stream = output_container.add_stream('aac', audio_sample_rate)

        audio_duration = audio_tensor.shape[-1] / audio_sample_rate

        for data_packet in source_video.demux(input_video_stream):
            if data_packet.dts is None:
                continue
            data_packet.stream = output_video_stream
            output_container.mux(data_packet)

        # Process audio
        try:
            audio_array = audio_tensor.numpy().astype(np.float32)
            audio_frame = AudioFrame.from_ndarray(audio_array, format='flt', layout='mono')
            audio_frame.sample_rate = audio_sample_rate

            for audio_packet in output_audio_stream.encode(audio_frame):
                output_container.mux(audio_packet)

            for final_audio_packet in output_audio_stream.encode():
                output_container.mux(final_audio_packet)
        except Exception as e:
            raise RuntimeError(f"Audio processing failed: {e}")

        source_video.close()
        output_container.close()
        
    except Exception as e:
        if 'source_video' in locals():
            source_video.close()
        if 'output_container' in locals():
            output_container.close()
        raise RuntimeError(f"Remuxing operation failed: {e}")