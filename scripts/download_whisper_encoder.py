"""Download exact public Whisper encoder tensors, without its decoder."""
from pathlib import Path
import argparse
import concurrent.futures as cf
import json
import struct
import urllib.request

REVISION='06f233fe06e710322aca913c1bc4249a0d71fce1'
BASE=f'https://huggingface.co/openai/whisper-large-v3/resolve/{REVISION}/'

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--workers',type=int,default=6)
    args=parser.parse_args()
    out=args.output;out.mkdir(parents=True,exist_ok=True)
    if (out/'encoder.safetensors').exists():raise FileExistsError(out/'encoder.safetensors')
    opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
    def request(name,first=None,last=None):
        headers={} if first is None else {'Range':f'bytes={first}-{last}'}
        response=opener.open(urllib.request.Request(BASE+name,headers=headers),timeout=120)
        if first is not None and (response.status!=206 or not response.headers.get('Content-Range','').startswith(f'bytes {first}-{last}/')):
            response.close();raise ValueError('Public endpoint did not honor exact tensor byte range')
        with response:data=response.read()
        if first is not None and len(data)!=last-first+1:raise ValueError('Incomplete byte range')
        return data
    for name in ['config.json','preprocessor_config.json','README.md']:
        if not (out/name).exists():
            with (out/name).open('xb') as stream:stream.write(request(name))
    header_length=struct.unpack('<Q',request('model.safetensors',0,7))[0]
    header=json.loads(request('model.safetensors',8,header_length+7))
    chosen={k.removeprefix('model.encoder.'):dict(v) for k,v in header.items() if k.startswith('model.encoder.')}
    if len(chosen)<=400:raise ValueError('Unexpected public encoder tensor set')
    lo=min(v['data_offsets'][0] for v in chosen.values());hi=max(v['data_offsets'][1] for v in chosen.values())
    if sum(v['data_offsets'][1]-v['data_offsets'][0] for v in chosen.values())!=hi-lo:
        raise ValueError('Encoder tensors are not contiguous')
    for value in chosen.values():value['data_offsets']=[x-lo for x in value['data_offsets']]
    tensor_count=len(chosen)
    chosen['__metadata__']={'format':'pt','source_repository':'openai/whisper-large-v3',
        'source_revision':REVISION,'selection':'model.encoder.* only, names prefix removed'}
    encoded=json.dumps(chosen,separators=(',',':')).encode();encoded+=b' '*((-len(encoded))%8)
    parts=out/'parts';parts.mkdir(exist_ok=True)
    jobs=[(i,a,min(hi,a+16*1024*1024)) for i,a in enumerate(range(lo,hi,16*1024*1024))]
    def part(job):
        i,a,b=job;path=parts/f'{i:04d}.bin'
        if path.exists():
            if path.stat().st_size!=b-a:raise ValueError('Existing incomplete range part: '+str(path))
            return
        data=request('model.safetensors',8+header_length+a,8+header_length+b-1)
        with path.open('xb') as stream:stream.write(data)
        print(json.dumps({'chunk':i,'bytes':len(data)}),flush=True)
    with cf.ThreadPoolExecutor(max_workers=args.workers) as pool:list(pool.map(part,jobs))
    with (out/'encoder.safetensors').open('xb') as stream:
        stream.write(struct.pack('<Q',len(encoded)));stream.write(encoded)
        for i,_,_ in jobs:stream.write((parts/f'{i:04d}.bin').read_bytes())
    receipt={'repository':'openai/whisper-large-v3','revision':REVISION,'encoder_tensors':tensor_count,
        'encoder_data_bytes':hi-lo,'encoder_only':True,'decoder_downloaded':False,
        'exact_public_tensor_bytes':True,'source_range':[8+header_length+lo,8+header_length+hi-1]}
    (out/'DOWNLOAD_READY.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(receipt))

if __name__=='__main__':main()
