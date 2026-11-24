"""Pinned SpeechBrain language model, with its original classify_batch path."""
from pathlib import Path
import json
import sys

import numpy as np

REVISION='0253049ae131d6a4be1c4f0d8b0ff483a0f8c8e9'


class LanguageModel:
    def __init__(self,root,device='cpu',threads=2):
        root=Path(root);sys.path.insert(0,str(root/'vendor'))
        import torch
        import speechbrain
        from hyperpyyaml import load_hyperpyyaml
        from speechbrain.inference.classifiers import EncoderClassifier
        torch.set_num_threads(threads);torch.manual_seed(2)
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        self.torch=torch;self.device=device
        folder=root/'model'
        api=json.loads((folder/'model_api.json').read_text())
        assert api['sha']==REVISION
        with (folder/'hyperparams.yaml').open() as f:
            params=load_hyperpyyaml(f,overrides={'pretrained_path':str(folder)})
        for name in ['embedding_model','classifier']:
            state=torch.load(folder/(name+'.ckpt'),map_location='cpu',weights_only=True)
            params[name].load_state_dict(state,strict=True)
        params['label_encoder'].load(str(folder/'label_encoder.txt'))
        self.model=EncoderClassifier(modules=params['modules'],hparams=params,
            run_opts={'device':device},freeze_params=True)
        self.model.eval().requires_grad_(False)
        self.labels=[self.model.hparams.label_encoder.ind2lab[i] for i in range(107)]
        assert self.labels[20]=='en: English'
        self.contract=dict(repo='speechbrain/lang-id-voxlingua107-ecapa',revision=REVISION,
            license='Apache-2.0, pinned model card',speechbrain=speechbrain.__version__,
            torch=torch.__version__,device=device,threads=threads,dtype='float32',
            preprocessing='original SpeechBrain Fbank60+sentence mean normalization; complete mono16k waveform float32/32768',
            output='EncoderClassifier.classify_batch original log posterior -> exp; English index20',
            threshold=0.9,acceptance='top1=en and English posterior>=0.90',
            batching='only exact equal-length waveforms together; lengths=1, no artificial padding',tf32=False)

    def predict(self,waveforms):
        torch=self.torch
        assert waveforms and len({len(w) for w in waveforms})==1
        values=np.stack(waveforms).astype(np.float32)
        assert np.isfinite(values).all() and values.ndim==2
        with torch.inference_mode():
            scores,_,_,_=self.model.classify_batch(torch.from_numpy(values).to(self.device),
                torch.ones(len(values),device=self.device))
            scores=scores.float().cpu()
        assert scores.shape==(len(values),107) and torch.isfinite(scores).all()
        assert torch.max(torch.abs(torch.logsumexp(scores,dim=1))).item()<1e-4
        posterior=scores.exp().numpy()
        results=[]
        for probability in posterior:
            top=int(probability.argmax());english=float(probability[20])
            label=self.labels[top]
            results.append(dict(top1_language=label.split(':',1)[0],top1_label=label,
                top1_confidence=float(probability[top]),english_posterior=english,
                accepted_english=top==20 and english>=.90))
        return results,posterior
