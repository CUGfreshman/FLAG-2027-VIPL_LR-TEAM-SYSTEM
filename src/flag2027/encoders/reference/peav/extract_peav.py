"""Frozen PE-AV Base audio and video projection helpers."""
import os

import numpy as np

SCHEMA = "flag2027_peav_base"
DIMENSION = 1024


def pe_video_embeddings(model, pixel_values_videos, padding_mask_videos=None):
    """Original encoder_helpers.py adapter for Transformers 5.18.0.

    PeAudioVideoModel.get_video_embeds calls a missing method in that
    release. These are its actual unchanged pooling/projection modules.
    """
    video = model.video_model
    output = video.video_encoder(pixel_values_videos=pixel_values_videos,
        padding_mask_videos=padding_mask_videos, return_dict=True)
    return video.video_head(output.pooler_output)


def check_array(value, rows):
    assert value.dtype == np.float32 and value.shape == (rows, DIMENSION)
    assert np.isfinite(value).all() and np.all(np.linalg.norm(value.astype(np.float64), axis=1) > 0)


class Encoders:
    def __init__(self, args):
        os.environ['HF_HUB_OFFLINE'] = '1'; os.environ['TRANSFORMERS_OFFLINE'] = '1'
        import torch
        import torchaudio
        import transformers
        from transformers import PeAudioVideoModel, PeAudioVideoProcessor
        assert transformers.__version__ == '5.18.0', transformers.__version__
        self.torch = torch; self.torchaudio = torchaudio
        torch.set_num_threads(args.threads); torch.manual_seed(0)
        np.random.seed(0)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False; torch.backends.cudnn.benchmark = False
        self.device = torch.device(args.device)
        self.model, loading = PeAudioVideoModel.from_pretrained(args.weights,
            local_files_only=True, dtype=torch.bfloat16, output_loading_info=True)
        assert not any(loading.values()), loading
        self.model.eval().requires_grad_(False).to(self.device)
        self.processor = PeAudioVideoProcessor.from_pretrained(args.weights, local_files_only=True)
        assert self.processor.feature_extractor.sampling_rate == 48000
        video_config = self.processor.video_processor.to_dict()
        assert video_config['size'] == {'height':336,'width':336}
        assert not video_config['do_sample_frames']
        self.resamplers = {}; self.image_batch = args.image_batch
        self.contract = dict(schema=SCHEMA, encoder='PE-AV Base', encoder_frozen=True,
            weights=str(args.weights), weights_bytes=(args.weights/'model.safetensors').stat().st_size,
            transformers=transformers.__version__, torch=torch.__version__, torchaudio=torchaudio.__version__,
            input_dtype='bfloat16', saved_dtype='float32', dimensions={'face':1024,'voice':1024},
            normalized_cache=False, image_batch_size=args.image_batch, audio_batch_size=1,
            visual='single RGB source center frame; no ArcFace alignment; official resize336 rescale1/255 mean/std0.5',
            audio='native-rate AAC decoded float32; arithmetic channel mean; same 16k-timebase window bounds; torchaudio Resample to48000; whole window; official padding; batch1',
            projection='independent original pretrained audio/video projection heads; no extra L2 normalization',
            video_adapter='unchanged original encoder_helpers.pe_video_embeddings',
            augmentation='none', tf32=False)
        self.processor_config = dict(audio=self.processor.feature_extractor.to_dict(), video=video_config)

    def move(self, inputs):
        return {k:v.to(device=self.device, dtype=self.torch.bfloat16) if v.is_floating_point()
                else v.to(self.device) for k,v in inputs.items()}

    def visual(self, frames):
        torch = self.torch; result = []
        with torch.inference_mode():
            for start in range(0, len(frames), self.image_batch):
                batch = frames[start:start+self.image_batch]
                inp = self.processor.video_processor(videos=[frame[None] for frame in batch],
                    do_sample_frames=False, return_tensors='pt')
                output = pe_video_embeddings(self.model, **self.move(inp)).float().cpu().numpy()
                check_array(output, len(batch)); result.append(output)
        return np.concatenate(result)

    def audio(self, segments):
        torch = self.torch; result = []; lengths = []
        with torch.inference_mode():
            for waveform, rate in segments:
                tensor = torch.from_numpy(waveform)
                if rate != 48000:
                    if rate not in self.resamplers:
                        self.resamplers[rate] = self.torchaudio.transforms.Resample(rate, 48000)
                    tensor = self.resamplers[rate](tensor)
                assert torch.isfinite(tensor).all()
                inp = self.processor.feature_extractor([tensor.numpy()], sampling_rate=48000,
                    padding=True, truncation=False, return_tensors='pt')
                output = self.model.get_audio_embeds(**self.move(inp)).float().cpu().numpy()
                check_array(output, 1); result.append(output); lengths.append(len(tensor))
        return np.concatenate(result), lengths

    def pilot(self, frames, segments):
        # Original baseline's batch-versus-single check on real media.
        count = min(2,len(frames)); batch = self.visual(frames[:count])
        single = np.concatenate([self.visual([x]) for x in frames[:count]])
        cosine = (batch*single).sum(1)/(np.linalg.norm(batch,axis=1)*np.linalg.norm(single,axis=1))
        relative = np.abs(np.linalg.norm(batch,axis=1)/np.linalg.norm(single,axis=1)-1)
        assert np.all(cosine >= .999) and np.all(relative <= .01), (cosine,relative)
        audio, lengths = self.audio(segments[:count])
        return dict(passed=True, frozen=all(not p.requires_grad for p in self.model.parameters()),
            model_eval=not self.model.training, face_shape=list(batch.shape),voice_shape=list(audio.shape),
            face_batch_single_cosine=cosine.tolist(),face_batch_single_relative_norm=relative.tolist(),
            audio_48000_samples=lengths, no_arcface_alignment=True, no_temporal_video_input=True,
            audio_batch=1, contract=self.contract)
