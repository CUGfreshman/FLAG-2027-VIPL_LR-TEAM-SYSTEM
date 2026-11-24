"""Train the released English projection with the fixed production recipe."""
import json
from pathlib import Path
import random
import sys
import time

import numpy as np
import torch

from .data import load_training_data
from .model import make_model
from .original.losses import contrastive_loss, sample_negative_permutations
from .runtime_io import csvwrite, save as save_payload, state_cpu, write


def setup():
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    random.seed(2)
    np.random.seed(2)
    torch.manual_seed(2)
    torch.cuda.manual_seed_all(2)


def save_checkpoint(path, payload):
    payload = dict(payload, python_rng=random.getstate(), numpy_rng=np.random.get_state(),
                   media_rng_formula='np.default_rng(2+1009*epoch)')
    save_payload(path, payload)


def same_recipe(left, right):
    """Compare training settings, allowing an explicitly selected older run."""
    metadata = {'model_source', 'sampler_source', 'loss_source', 'M_model_factory', 'model_factory',
                'task', 'arm', 'fold', 'initial_checkpoint'}
    def settings(config):
        result = {key: value for key, value in config.items() if key not in metadata}
        if 'data' in result:
            result['data'] = {key: value for key, value in result['data'].items() if key != 'fold'}
        return result

    return settings(left) == settings(right)


def train_one_epoch(model, opt, sampler, epoch, steps, labels, sources, label_gpu,
                    source_gpu, negative_rng, tensors, device, index, exposure, trace):
    model.train()
    rng = np.random.default_rng(2+1009*epoch)
    statistics = []
    for step in range(steps):
        batch = sampler.draw(rng)
        bi = torch.as_tensor(batch, device=device)
        neg = sample_negative_permutations(labels[batch], sources[batch], generator=negative_rng)
        f, a, *_ = model(tensors['face'][bi], tensors['voice'][bi])
        loss, detail = contrastive_loss(f, a, label_gpu[bi], source_gpu[bi], 'T16', neg)
        assert torch.isfinite(loss)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        if step == 0:
            assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
        opt.step()
        np.add.at(exposure, batch, 1)
        statistics.append(dict(loss=float(loss.detach()), **detail))
        if epoch in (1, 300) and step in (0, steps-1):
            trace.append(dict(epoch=epoch, step=step, global_rows=index[batch].tolist(), source=sources[batch].tolist(),
                negative_f=neg.image_to_audio[:2, :16].tolist(), negative_a=neg.audio_to_image[:2, :16].tolist()))
    return statistics


def run_training(data_directory, dest):
    import fcntl

    setup()
    dest.mkdir(parents=True, exist_ok=True)
    with (dest / 'writer.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (dest / 'complete.json').exists():
            return
        device = torch.device('cuda:0')
        assert torch.cuda.device_count() == 1
        invocation = time.monotonic()
        rows, index, labels, sources, sampler, raw, info = load_training_data(data_directory)
        cfg = dict(seed=2, epochs=300, fixed_endpoint=300,
            data={key: value for key, value in info.items() if key != 'fold'},
            architecture='Linear[input]-256/BN/ReLU/Dropout0.5/L2',
            input_preprocessing='PEAV-final and identity blocks separately L2; concat1536/1280; no concat L2',
            optimizer='Adam', lr=1e-5, weight_decay=.01, betas=[.9, .999], eps=1e-8, scheduler=None,
            p=32, k=4, q=16, margin=.2, positive='original row only', ignore_sameID_offdiagonal=True,
            negatives='independent two-direction uniform differentID within source v1/v2/v4; languages may mix within source',
            source_labels={'v1':0, 'v2':1, 'v4':2}, identity_CE=False, gender_reweighting=False, AMP=False, TF32=False,
            model_source='triplet/original/original_model.py', sampler_source='triplet/original/original_objective.py',
            loss_source='triplet/original/losses.py', model_factory='triplet/model.py',
            validation='No fold or local EER; direct package fixed300; retain complete English training BN',
            held_or_official_media_used_for_training=False, AL=False, COHORT=False)
        config_path = dest / 'config.json'
        if config_path.exists():
            assert same_recipe(json.loads(config_path.read_text(encoding='utf-8')), cfg)
        else:
            write(config_path, cfg)
        model = make_model()
        initial = dest / 'initial_epoch0000.pth.tar'
        if not initial.exists():
            save_checkpoint(initial, dict(seed=2, model_state=state_cpu(model),
                torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all(),
                input_widths={'face': 1536, 'voice': 1280}))
        model = model.to(device)
        opt = torch.optim.Adam(model.parameters(), lr=1e-5, weight_decay=.01, betas=(.9, .999), eps=1e-8)
        tensors = {m: torch.from_numpy(x).to(device) for m, x in raw.items()}
        del raw
        label_gpu, source_gpu = labels.to(device), sources.to(device)
        negative_rng = torch.Generator(device='cpu').manual_seed(2)
        steps, start, prior = info['steps_per_epoch'], 1, 0.
        history, evaluated, trace = [], [], []
        exposure = np.zeros(len(rows), np.int64)
        latest = dest / 'checkpoint_latest.pth.tar'
        if latest.exists():
            ck = torch.load(latest, map_location='cpu', weights_only=False)
            assert same_recipe(ck['config'], cfg)
            model.load_state_dict(ck['model_state'])
            opt.load_state_dict(ck['optimizer_state'])
            for state in opt.state.values():
                for k, v in state.items():
                    if torch.is_tensor(v) and k != 'step':
                        state[k] = v.to(device)
            torch.set_rng_state(ck['torch_rng'])
            torch.cuda.set_rng_state_all(ck['cuda_rng'])
            negative_rng.set_state(ck['negative_rng'])
            start, prior = ck['epoch'] + 1, ck['elapsed_seconds']
            history, evaluated, trace, exposure = ck['history'], ck['evaluation'], ck['trace'], ck['exposure']
        begin = time.monotonic()
        for epoch in range(start, 301):
            epoch_start = time.monotonic()
            statistics = train_one_epoch(model, opt, sampler, epoch, steps, labels, sources, label_gpu,
                                         source_gpu, negative_rng, tensors, device, index, exposure, trace)
            row = dict(epoch=epoch, global_step=epoch*steps, elapsed_seconds=prior+time.monotonic()-begin,
                train_epoch_seconds=time.monotonic()-epoch_start, evaluation_seconds=0.,
                **{k:float(np.mean([s[k] for s in statistics])) for k in statistics[0]})
            history.append(row)
            row['elapsed_seconds'] = prior+time.monotonic()-begin
            csvwrite(dest / 'training_log.csv', history)
            write(dest / 'progress.json', row)
            if epoch % 10 == 0:
                payload = dict(config=cfg, epoch=epoch, global_step=epoch*steps, model_state=state_cpu(model),
                    optimizer_state=opt.state_dict(), torch_rng=torch.get_rng_state(),
                    cuda_rng=torch.cuda.get_rng_state_all(), negative_rng=negative_rng.get_state(),
                    elapsed_seconds=row['elapsed_seconds'], history=history, evaluation=evaluated, trace=trace, exposure=exposure)
                save_checkpoint(latest, payload)
                if epoch == 300:
                    save_checkpoint(dest / 'epoch0300.pth.tar', payload)
                print('PROGRESS', epoch, epoch*steps, row['loss'], row['elapsed_seconds'], flush=True)
        assert all(int(model.state_dict()[m+'.block.1.num_batches_tracked']) == steps*300 for m in ('face', 'voice'))
        final_checkpoint = dest / 'epoch0300.pth.tar'
        if not final_checkpoint.exists():
            final_payload = torch.load(latest, map_location='cpu', weights_only=False)
            assert final_payload['epoch'] == 300 and final_payload['global_step'] == steps*300
            save_checkpoint(final_checkpoint, final_payload)
        candidate = np.concatenate(list(sampler.rows.values()))
        np.save(dest / 'source_exposures.npy', exposure)
        write(dest / 'sampling_trace.json', trace)
        write(dest / 'complete.json', dict(complete=True, epoch=300,
            updates=steps*300, steps_per_epoch=steps, checkpoint=str(final_checkpoint), training_rows=len(rows),
            training_identities=info['training_identity_count'], unique_candidate_pairs=len(candidate),
            candidate_pairs_seen=int((exposure[candidate]>0).sum()), all_unique_pairs_seen=bool(np.all(exposure[candidate]>0)),
            actual_sample_exposures=int(exposure.sum()), elapsed_seconds=prior+time.monotonic()-begin,
            invocation_seconds=time.monotonic()-invocation, BN='saved training running stats; no recalibration',
            fixed_endpoint=True, held_training_overlap=False, official_read=False))
        print('COMPLETE', final_checkpoint, flush=True)


def train(data_directory, output_directory):
    """Train on one visible CUDA device, then export the tensor-only endpoint."""
    if sys.platform == 'win32':
        raise RuntimeError('Train on Linux/WSL: the original loop uses fcntl writer locking')
    data_directory = Path(data_directory).resolve()
    endpoint = Path(output_directory).resolve()
    if any(path.is_file() for path in endpoint.glob('outputs/*/fold*/config.json')):
        raise FileExistsError('A nested training run already exists. Pass its exact run directory as --output to resume, '
                              'or choose a new output directory.')
    run_training(data_directory, endpoint)
    payload = torch.load(endpoint / 'epoch0300.pth.tar', map_location='cpu', weights_only=False)
    if payload['epoch'] != 300 or payload['global_step'] != 53700:
        raise ValueError('Unexpected new-training endpoint')
    released = dict(format='flag2027-triplet-inference-v1', epoch=300, global_step=53700,
                    input_widths={'face': 1536, 'voice': 1280}, model_state=payload['model_state'])
    inference_file = endpoint / 'epoch0300.pth'
    if inference_file.exists():
        stored = torch.load(inference_file, map_location='cpu', weights_only=True)
        if any(not torch.equal(value, stored['model_state'][key]) for key, value in released['model_state'].items()):
            raise RuntimeError('Existing inference endpoint differs from the completed run')
    else:
        save_payload(inference_file, released)
