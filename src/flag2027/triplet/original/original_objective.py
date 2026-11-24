"""Unique-file identity sampling and a differentiable audio-prototype ranking loss."""
import numpy as np
import torch
from torch.nn import functional as F


class UniquePairSampler:
    """Choose P identities uniformly, then at most K original paired rows each.

    Resolved face/audio paths are separately unique within each identity's draw.
    A person with one actual audio contributes one row, never K alias copies.
    """
    def __init__(self, identities, face_paths, voice_paths, p=32, k=4):
        self.identities = np.asarray(identities)
        self.face_paths, self.voice_paths = np.asarray(face_paths), np.asarray(voice_paths)
        self.people = np.unique(self.identities)
        assert len(self.people) >= p
        self.p, self.k = p, k
        self.rows = {}
        for identity in self.people:
            unique = {}
            for row in np.flatnonzero(self.identities == identity):
                unique.setdefault((face_paths[row], voice_paths[row]), int(row))
            self.rows[identity] = np.array(list(unique.values()), dtype=np.int64)

    def draw(self, rng):
        selected = []
        for identity in rng.choice(self.people, self.p, replace=False):
            faces, voices = set(), set()
            for row in rng.permutation(self.rows[identity]):
                f, v = self.face_paths[row], self.voice_paths[row]
                if f in faces or v in voices:
                    continue
                selected.append(int(row)); faces.add(f); voices.add(v)
                if len(faces) == self.k:
                    break
            assert len(faces) >= 1
        result = np.array(selected, dtype=np.int64)
        assert len(np.unique(self.identities[result])) == self.p
        return result


def audio_prototype_triplet(face, voice, identity, margin=.2):
    """Both anchor directions, with equal identity and identity-pair weights.

    Audio prototype i is unit(mean(unit audio files of i)). A face anchor ranks
    its own prototype against each other identity's prototype. An audio
    prototype ranks every own-identity face against every other-identity face.
    The latter averages hinges over positive files, not distances before hinge.
    """
    people, owner, counts = torch.unique(identity, sorted=True, return_inverse=True, return_counts=True)
    p, b = len(people), len(face)
    assert p >= 2 and b == len(voice)
    sums = voice.new_zeros((p, voice.shape[1])).index_add(0, owner, voice)
    prototype = F.normalize(sums / counts[:, None], dim=1)
    d = torch.cdist(prototype, face, p=2)
    pos = d[owner, torch.arange(b, device=d.device)]
    face_weight = counts[owner].to(d.dtype).reciprocal()
    other_proto = owner[:, None] != torch.arange(p, device=d.device)[None, :]
    face_hinge = F.relu(pos[:, None] - d.T + margin)
    face_loss = (face_hinge * other_proto * face_weight[:, None]).sum() / (p * (p-1))
    other_face = owner[:, None] != owner[None, :]
    audio_hinge = F.relu(pos[:, None] - d[owner] + margin)
    pair_weight = face_weight[:, None] * face_weight[None, :]
    audio_loss = (audio_hinge * other_face * pair_weight).sum() / (p * (p-1))
    total = face_loss + audio_loss
    with torch.no_grad():
        stat = dict(triplet=float(total.detach()),triplet_video_anchor_loss=float(face_loss.detach()),
            triplet_audio_anchor_loss=float(audio_loss.detach()),
            positive_distance=float((pos*face_weight).sum()/p),
            negative_distance=float((d.T*other_proto*face_weight[:, None]).sum()/(p*(p-1))),
            active_rate=float((((face_hinge>0)*other_proto*face_weight[:,None]).sum()
                               + ((audio_hinge>0)*other_face*pair_weight).sum())/(2*p*(p-1))),
            identities=p,batch_rows=b,singleton_identities=int((counts==1).sum()))
    return total, stat
