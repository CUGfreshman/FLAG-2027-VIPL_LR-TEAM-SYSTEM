"""Original symmetric-triplet model with a clean metric path from one forward.

State names, initialization, dropout CE path, and default forward match the
historical tri_symmetric300 model. The only optional intervention is whether
the unchanged triplet objective sees pre- or post-dropout branch embeddings.
"""
import torch
from torch import nn
from torch.nn import functional as F


class Forward_Block(nn.Module):
    def __init__(self, input_dim=256, output_dim=256, p_val=0.):
        super().__init__()
        self.block=nn.Sequential(nn.Linear(input_dim,output_dim),nn.BatchNorm1d(output_dim),
                                 nn.ReLU(),nn.Dropout(p=p_val))

    def forward(self,x):
        return self.block(x)


class GatedFusion(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_face=nn.Sequential(); self.linear_voice=nn.Sequential(); self.final_transform=nn.Sequential()
        self.attention=nn.Sequential(Forward_Block(512,256),nn.Linear(256,256))

    def forward(self,face_input,voice_input):
        attention_out=torch.sigmoid(self.attention(torch.cat((face_input,voice_input),dim=1)))
        face_trans=torch.tanh(self.linear_face(face_input))
        voice_trans=torch.tanh(self.linear_voice(voice_input))
        out=face_trans*attention_out+(1.-attention_out)*voice_trans
        return self.final_transform(out),face_trans,voice_trans


class Projection(nn.Module):
    def __init__(self):
        super().__init__()
        self.block=nn.Sequential(nn.Linear(1024,256),nn.BatchNorm1d(256),nn.ReLU(inplace=True),nn.Dropout(.5))

    def forward(self,x):
        return F.normalize(self.block(x),p=2,dim=1)

    def forward_with_clean(self,x):
        hidden=self.block[2](self.block[1](self.block[0](x)))
        clean=F.normalize(hidden,p=2,dim=1)
        dropped=F.normalize(self.block[3](hidden),p=2,dim=1)
        return dropped,clean


class TripletModel(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.face=Projection(); self.voice=Projection()
        self.use_fop=bool(config['use_fop'])
        if self.use_fop:
            self.fusion=GatedFusion(); self.classifier=nn.Linear(256,config['n_class'])
            for module in [self.fusion,self.classifier]:
                for layer in module.modules():
                    if isinstance(layer,nn.Linear):
                        nn.init.xavier_uniform_(layer.weight); nn.init.constant_(layer.bias,.01)

    def forward(self,faces,voices):
        face=self.face(faces); voice=self.voice(voices)
        if not self.use_fop:
            return face,voice,None,None,None
        fused,fs,vs=self.fusion(face,voice)
        return face,voice,fused,self.classifier(fused),(fs,vs)

    def forward_with_clean(self,faces,voices):
        face,clean_face=self.face.forward_with_clean(faces)
        voice,clean_voice=self.voice.forward_with_clean(voices)
        fused,fs,vs=self.fusion(face,voice)
        return face,voice,fused,self.classifier(fused),(fs,vs),clean_face,clean_voice


def matrix_triplet_loss(face,voice,identities,margin,gender=None,negative_selection=None):
    similarity=voice@face.T
    distance=torch.sqrt((2.-2.*similarity).clamp_min(0.))
    negative=identities[:,None].ne(identities[None,:])
    pos=distance.diagonal()
    hv=F.relu(pos[None,:]-distance+margin)
    ha=F.relu(pos[:,None]-distance+margin)
    # Default exactly matches historical all-pairs means. A selected-negative
    # intervention may use distinct row/column masks, preserving both anchors.
    mv,ma=(negative,negative) if negative_selection is None else negative_selection
    if torch.any(mv & ~negative) or torch.any(ma & ~negative):
        raise ValueError('Selected negatives include same-identity pairs')
    if not mv.any() or not ma.any():
        raise ValueError('No valid negatives')
    lv=hv[mv].mean(); la=ha[ma].mean(); total=lv+la
    parts={}
    with torch.no_grad():
        stats={'triplet_loss':float(total.detach()),'triplet_video_anchor_loss':float(lv.detach()),
               'triplet_audio_anchor_loss':float(la.detach()),'positive_distance':float(pos.mean()),
               'negative_distance':float(distance[negative].mean()),
               'active_rate':float(((hv[mv]>0).float().mean()+(ha[ma]>0).float().mean())/2),
               'video_anchor_active_rate':float((hv[mv]>0).float().mean()),
               'audio_anchor_active_rate':float((ha[ma]>0).float().mean()),
               'positive_pairs':len(pos),'negative_pairs':int(negative.sum()),
               'selected_video_negative_pairs':int(mv.sum()),'selected_audio_negative_pairs':int(ma.sum()),
               'ignored_same_identity_offdiag':int((~negative).sum()-len(pos))}
    if gender is not None:
        same=gender[:,None].eq(gender[None,:])
        for name,gm in [('same_gender',same),('different_gender',~same)]:
            x,y=mv & gm,ma & gm
            parts[name]=hv[x].sum()/mv.sum()+ha[y].sum()/ma.sum()
            with torch.no_grad():
                count=int(x.sum()+y.sum())
                stats[name+'_candidate_count']=count
                stats[name+'_active_count']=int((hv[x]>0).sum()+(ha[y]>0).sum())
                stats[name+'_active_rate']=stats[name+'_active_count']/count if count else 0.
                stats[name+'_loss_contribution']=float(parts[name].detach())
                stats[name+'_negative_distance']=float(distance[negative & gm].mean()) if (negative & gm).any() else float('nan')
    return total,stats,parts
