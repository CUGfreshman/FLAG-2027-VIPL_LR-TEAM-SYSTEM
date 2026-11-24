"""Complete-MP4 local language classification; no source files are modified."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import argparse,collections,csv,fcntl,json,math,os,subprocess,sys,time,traceback
import numpy as np

from lid_model import LanguageModel,REVISION

def write_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n');tmp.replace(path)

def decode(row):
    t=time.monotonic()
    try:
        p=subprocess.run(['ffmpeg','-nostdin','-v','error','-threads','1','-i',row['path'],'-map','0:a:0','-vn','-ac','1','-ar','16000','-f','f32le','-acodec','pcm_f32le','pipe:1'],stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=180)
        if p.returncode:raise RuntimeError(p.stderr.decode('utf8','replace')[-1500:])
        audio=np.frombuffer(p.stdout,dtype='<f4').copy()
        if len(audio)<400 or not np.isfinite(audio).all():raise ValueError('empty/short/nonfinite decoded waveform')
        return row,audio,time.monotonic()-t,None
    except Exception as e:return row,None,time.monotonic()-t,str(e)

def decode_av(row):
    import av
    t=time.monotonic();row=dict(row)
    try:
        pieces=[]
        with av.open(row['path'],timeout=60) as media:
            stream=next((s for s in media.streams if s.type=='audio'),None)
            if stream is None:raise ValueError('MP4 has no audio stream')
            stream.codec_context.thread_count=1
            row.update(source_audio_codec=stream.codec_context.name,source_audio_rate=stream.codec_context.sample_rate,source_audio_channels=stream.codec_context.channels)
            resampler=av.AudioResampler(format='flt',layout='mono',rate=16000)
            for frame in media.decode(stream):
                for part in resampler.resample(frame):pieces.append(part.to_ndarray().reshape(-1))
            for part in resampler.resample(None):pieces.append(part.to_ndarray().reshape(-1))
        if not pieces:raise ValueError('empty decoded audio')
        audio=np.concatenate(pieces).astype(np.float32,copy=False)
        if len(audio)<400 or not np.isfinite(audio).all():raise ValueError('empty/short/nonfinite decoded waveform')
        return row,audio,time.monotonic()-t,None
    except Exception as e:return row,None,time.monotonic()-t,str(e)

def classify(model,decoded,max_audio_seconds=64,max_batch=16):
    groups=collections.defaultdict(list);records=[];posteriors={};sizes=collections.Counter();oom_splits=0
    for row,audio,seconds,error in decoded:
        record=dict(row_index=int(row['row_index']),key=row['key'],identity=row['identity'],youtube_id=row['youtube_id'],utterance=row['utterance'],path=row['path'],status='ok',model_revision=REVISION,decode_seconds=seconds,**{k:v for k,v in row.items() if k.startswith('source_audio_')})
        if error:record.update(status='decode_error',error_reason=error);records.append(record)
        else:
            record.update(audio_samples=len(audio),sample_rate=16000,audio_duration_seconds=len(audio)/16000)
            groups[len(audio)].append((record,audio))
    bn_index=next(i for i,s in enumerate(model.labels) if s.startswith('bn:'))
    def predict(items):
        nonlocal oom_splits
        try:
            prediction,posterior=model.predict([audio for _,audio in items]);sizes[len(items)]+=1
            for (r,_),p,prob in zip(items,prediction,posterior):
                p.pop('accepted_english',None);r.update(p);r['bengali_posterior']=float(prob[bn_index]);r['error_reason']='';records.append(r);posteriors[r['row_index']]=prob
        except model.torch.cuda.OutOfMemoryError as e:
            model.torch.cuda.empty_cache()
            if len(items)>1:
                oom_splits+=1;middle=len(items)//2;predict(items[:middle]);predict(items[middle:])
            else:
                r,_=items[0];r.update(status='inference_error',error_reason='CUDA OOM on whole utterance; not truncated');records.append(r)
        except Exception as e:
            if len(items)>1:
                middle=len(items)//2;predict(items[:middle]);predict(items[middle:])
            else:
                r,_=items[0];r.update(status='inference_error',error_reason=str(e));records.append(r)
    for samples,items in sorted(groups.items()):
        batch=max(1,min(max_batch,int(max_audio_seconds/(samples/16000))))
        for start in range(0,len(items),batch):predict(items[start:start+batch])
    records.sort(key=lambda r:r['row_index'])
    return records,posteriors,dict(batch_size_counts=dict(sizes),oom_splits=oom_splits)

def sample_rows(manifest,n,random=False):
    total=json.loads((ROOT/'inventory.json').read_text())['rows']
    targets=set((np.random.default_rng(2).choice(total,n,replace=False) if random else np.linspace(0,total-1,n,dtype=np.int64)).tolist())
    with manifest.open(newline='') as f:return [r for r in csv.DictReader(f) if int(r['row_index']) in targets]

def benchmark(args,model):
    selected=sample_rows(ROOT/'manifest.csv',args.benchmark_count*2,random=args.decoder=='pyav')
    selected=selected[args.worker::2]
    write_json(ROOT/f'benchmark/worker{args.worker}_selected.json',selected)
    started=time.monotonic()
    with ThreadPoolExecutor(max_workers=args.decode_workers) as pool:
        decoded=list(pool.map(decode_av if args.decoder=='pyav' else decode,selected))
    decoded_at=time.monotonic();records,posterior,stats=classify(model,decoded,args.audio_seconds,args.max_batch)
    finished=time.monotonic()
    # Compare already decoded real, exact-length batches against single utterances.
    same=collections.defaultdict(list)
    for row,audio,_,error in decoded:
        if error is None:same[len(audio)].append((row,audio))
    comparison=[]
    for _,items in sorted(same.items(),key=lambda z:-len(z[1]))[:3]:
        subset=items[:min(len(items),4)]
        if len(subset)<2:continue
        _,batch=model.predict([a for _,a in subset])
        for (row,a),prob in zip(subset,batch):
            _,single=model.predict([a]);comparison.append(float(np.abs(prob-single[0]).max()))
    value=dict(complete=True,worker=args.worker,pid=os.getpid(),rows=len(records),successful=sum(r['status']=='ok' for r in records),decode_seconds=decoded_at-started,inference_seconds=finished-decoded_at,end_to_end_seconds=finished-started,rows_per_second=len(records)/(finished-started),audio_hours=sum(r.get('audio_duration_seconds',0) for r in records)/3600,mean_audio_seconds=np.mean([r.get('audio_duration_seconds',0) for r in records]),batch_vs_single_max_posterior_difference=max(comparison) if comparison else None,batch_vs_single_compared=len(comparison),decode_workers=args.decode_workers,max_audio_seconds=args.audio_seconds,max_batch=args.max_batch,stats=stats,model_contract=model.contract,started_monotonic=started,finished_monotonic=finished)
    value['decoder']=args.decoder
    assert not comparison or max(comparison)<2e-5
    write_json(ROOT/f'benchmark/{args.decoder}_worker{args.worker}.json',value)
    write_json(ROOT/f'benchmark/{args.decoder}_worker{args.worker}_results.json',records)
    print(json.dumps(value),flush=True)

def work_chunks(manifest,size):
    with manifest.open(newline='') as f:
        chunk=[];index=0
        for row in csv.DictReader(f):
            chunk.append(row)
            if len(chunk)==size:yield index,chunk;index+=1;chunk=[]
        if chunk:yield index,chunk

def full(args,model):
    destination=ROOT/'chunks';destination.mkdir(exist_ok=True)
    count=0;reused=0;new=0;audio_seconds=0;failures=0;started=time.monotonic()
    total=json.loads((ROOT/'inventory.json').read_text())['rows']
    with ThreadPoolExecutor(max_workers=args.decode_workers) as pool:
        for shard,rows in work_chunks(ROOT/'manifest.csv',args.shard_size):
            if shard%2!=args.worker:continue
            output=destination/f'{shard:06d}.json'
            if output.exists():
                old=json.loads(output.read_text());assert old['complete'] and old['model_revision']==REVISION
                assert [r['key'] for r in old['records']]==[r['key'] for r in rows]
                count+=len(rows);reused+=len(rows);failures+=sum(r['status']!='ok' for r in old['records']);continue
            t=time.monotonic();decoded=list(pool.map(decode_av if args.decoder=='pyav' else decode,rows));decoded_at=time.monotonic()
            records,posterior,stats=classify(model,decoded,args.audio_seconds,args.max_batch)
            assert len(records)==len(rows) and [r['key'] for r in records]==[r['key'] for r in rows]
            if posterior:
                indices=np.asarray(sorted(posterior),dtype=np.int64)
                temp=destination/f'{shard:06d}.posterior.tmp.npz'
                np.savez_compressed(temp,row_indices=indices,posterior=np.stack([posterior[int(i)] for i in indices]),labels=np.asarray(model.labels))
                temp.replace(destination/f'{shard:06d}.posterior.npz')
            value=dict(complete=True,shard=shard,worker=args.worker,model_revision=REVISION,rows=len(rows),records=records,stats=stats,decode_seconds=decoded_at-t,inference_seconds=time.monotonic()-decoded_at)
            write_json(output,value)
            count+=len(rows);new+=len(rows);audio_seconds+=sum(r.get('audio_duration_seconds',0) for r in records);failures+=sum(r['status']!='ok' for r in records)
            progress=dict(state='running',worker=args.worker,pid=os.getpid(),last_shard=shard,completed_rows=count,new_rows=new,reused_rows=reused,total_rows=total,elapsed_seconds=time.monotonic()-started,new_rows_per_second=new/(time.monotonic()-started),new_audio_hours=audio_seconds/3600,errors=failures,physical_gpu_uuid=os.environ.get('CUDA_VISIBLE_DEVICES'),updated_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()))
            write_json(ROOT/f'progress_gpu{args.worker}.json',progress);print(json.dumps(progress),flush=True)
    progress=dict(state='complete',worker=args.worker,pid=os.getpid(),completed_rows=count,new_rows=new,reused_rows=reused,total_rows=total,elapsed_seconds=time.monotonic()-started,new_audio_hours=audio_seconds/3600,errors=failures,physical_gpu_uuid=os.environ.get('CUDA_VISIBLE_DEVICES'),updated_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()))
    write_json(ROOT/f'progress_gpu{args.worker}.json',progress)

def main():
    p=argparse.ArgumentParser();p.add_argument('--work-dir',type=Path,required=True);p.add_argument('--model-dir',type=Path,required=True);p.add_argument('--worker',type=int,choices=[0,1],required=True);p.add_argument('--benchmark-count',type=int,default=0);p.add_argument('--decode-workers',type=int,default=16);p.add_argument('--shard-size',type=int,default=1024);p.add_argument('--audio-seconds',type=float,default=64);p.add_argument('--max-batch',type=int,default=16);p.add_argument('--decoder',choices=['pyav','ffmpeg'],default='pyav');args=p.parse_args()
    global ROOT
    ROOT = args.work_dir.resolve()
    root=ROOT/'locks';root.mkdir(exist_ok=True)
    lock=(root/f'worker{args.worker}.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    model=LanguageModel(args.model_dir.resolve(),device='cuda:0',threads=2)
    model.contract.pop('threshold',None);model.contract.pop('acceptance',None)
    model.contract.update(input='complete first MP4 audio stream decoded/resampled mono16k float32; no VAD/crop/truncation',preprocessing='original SpeechBrain Fbank60+sentence mean normalization; normalized floating PCM decoded directly, no second division by32768',decoder=args.decoder,selection='none; all languages classified',full107_posterior_saved=True)
    write_json(ROOT/f'contract_gpu{args.worker}.json',model.contract)
    if args.benchmark_count:benchmark(args,model)
    else:full(args,model)

if __name__=='__main__':main()
