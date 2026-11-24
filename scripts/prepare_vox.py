"""Restore the exact frozen Vox selection using public utterance keys.

This command does not reselect identities/utterances. The optional zero-audio
mask is branch-specific: FaceNet-WeSpeaker retained one silent item; the
other views skipped it.
"""
from pathlib import Path
import argparse,csv,gzip,json

REPO=Path(__file__).resolve().parents[1]
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mp4-root',type=Path,required=True)
    p.add_argument('--selection',type=Path,default=REPO/'data_lists/vox_selected.csv.gz')
    p.add_argument('--excluded-identities',type=Path,
        help='Optional owner-provided exclusion CSV for an alternative selection')
    p.add_argument('--split',choices=['train','held','all'],default='train')
    p.add_argument('--drop-zero-audio',action='store_true',help='Use only for branches whose original input mask skipped silent union104443')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--check-exists',action='store_true')
    args=p.parse_args()
    open_selection = gzip.open if args.selection.suffix == '.gz' else open
    with open_selection(args.selection,'rt',encoding='utf-8-sig',newline='') as stream:
        rows=list(csv.DictReader(stream))
    excluded=set()
    if args.excluded_identities:
        with args.excluded_identities.open(encoding='utf-8-sig',newline='') as stream:
            excluded={r['identity'] for r in csv.DictReader(stream)}
    output=[];dropped=[]
    for row in rows:
        held=row['split']=='held'
        if args.split=='train' and held:continue
        if args.split=='held' and not held:continue
        if row['identity'] in excluded or row['lid_status']!='ok' or row['top1_language'] in {'bn','as'}:
            raise ValueError('Frozen selected row violates its original filtering rules')
        if args.drop_zero_audio and row.get('audio_all_zero','').lower()=='true':
            dropped.append(row['union_index']);continue
        key=Path(row['canonical_key'])
        if key.is_absolute() or '..' in key.parts:
            raise ValueError('Vox canonical key must be relative to the media root')
        path=args.mp4_root/key
        if args.check_exists and not path.is_file():raise FileNotFoundError(path)
        output.append({'trial_id':row['union_index'],'subset':'vox_held' if held else 'vox_train',
            'face_path':str(path),'audio_path':str(path),'identity':row['identity'],
            'gender':row['gender'],'source':'vox','language':row['top1_language'],
            'canonical_key':row['canonical_key'],'union_index':row['union_index']})
    if not output:raise ValueError('No rows for the requested Vox selection')
    if args.output.exists():raise FileExistsError(args.output)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x',encoding='utf-8',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(output[0]));writer.writeheader();writer.writerows(output)
    info={'rows':len(output),'identities':len({r['identity'] for r in output}),'split':args.split,
          'dropped_zero_audio_union_indices':dropped,'selection_resampled':False}
    args.output.with_suffix('.json').write_text(json.dumps(info,indent=2),encoding='utf-8')
    print(json.dumps(info))

if __name__=='__main__':main()
