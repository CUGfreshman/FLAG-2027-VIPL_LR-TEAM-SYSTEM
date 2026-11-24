"""Extract frozen six-branch encoder features for a caller-owned media manifest."""
from pathlib import Path
import argparse,json,sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from flag2027.encoders import FEATURE_DIMENSIONS,read_manifest,extract_manifest

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest',type=Path,required=True)
    parser.add_argument('--config',type=Path,default=Path(__file__).resolve().parents[1]/'configs/encoders.example.json')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--device',default='cuda:0')
    parser.add_argument('--encoders',nargs='+',choices=['peav','facenet','wespeaker','imagebind','arcface','dinov2','whisper13'])
    parser.add_argument('--validate-only',action='store_true',help='Check manifest/schema without importing or executing model frameworks')
    args=parser.parse_args();rows=read_manifest(args.manifest)
    config=json.loads(args.config.read_text(encoding='utf-8'))
    if args.validate_only:
        print(json.dumps({'manifest_rows':len(rows),'dimensions':FEATURE_DIMENSIONS,'models_executed':False},indent=2));return
    result=extract_manifest(rows,config,args.output,args.device,args.encoders)
    args.output.with_suffix('.contract.json').write_text(json.dumps(result,indent=2),encoding='utf-8')

if __name__=='__main__':main()
