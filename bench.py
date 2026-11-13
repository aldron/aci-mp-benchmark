import argparse, ctypes, csv, json, os, time, resource
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parent
class ACI:
 def __init__(self,x,group_size=4,bins=16):
  self.lib=ctypes.CDLL(str(ROOT/'libaci.so'))
  f32=np.ctypeslib.ndpointer(dtype=np.float32,flags='C_CONTIGUOUS')
  self.lib.aci_create.argtypes=[f32,ctypes.c_int,ctypes.c_int,ctypes.c_int];self.lib.aci_create.restype=ctypes.c_void_p
  self.lib.aci_search.argtypes=[ctypes.c_void_p,f32,ctypes.c_int,ctypes.c_int,ctypes.c_int,np.ctypeslib.ndpointer(dtype=np.int32,flags='C_CONTIGUOUS'),f32,np.ctypeslib.ndpointer(dtype=np.int32,flags='C_CONTIGUOUS')]
  self.lib.aci_search.restype=ctypes.c_int
  self.lib.aci_index_bytes.argtypes=[ctypes.c_void_p];self.lib.aci_index_bytes.restype=ctypes.c_size_t
  self.lib.aci_destroy.argtypes=[ctypes.c_void_p]
  self.ptr=self.lib.aci_create(x,len(x),group_size,bins)
  if not self.ptr:raise RuntimeError('ACI index creation failed')
 def search(self,q,k,posting,rerank):
  ids=np.empty(k,np.int32);ds=np.empty(k,np.float32);stats=np.zeros(4,np.int32)
  count=self.lib.aci_search(self.ptr,np.ascontiguousarray(q,dtype=np.float32),k,posting,rerank,ids,ds,stats)
  if count<0:raise RuntimeError('ACI search failed')
  return ids,ds,stats
 def memory(self):return self.lib.aci_index_bytes(self.ptr)
 def close(self):
  if self.ptr:self.lib.aci_destroy(self.ptr);self.ptr=None

def make_data(n,nq,seed,mode):
 rng=np.random.default_rng(seed)
 x=rng.normal(size=(n,128)).astype('float32')
 q=rng.normal(size=(nq,128)).astype('float32')
 if mode=='correlated':
  x[:,:]=.9*x[:,0:1]+.1*x;q[:,:]=.9*q[:,0:1]+.1*q
 if mode=='normalized':
  x/=np.linalg.norm(x,axis=1,keepdims=True);q/=np.linalg.norm(q,axis=1,keepdims=True)
 return x,q

def stats_summary(lat):
 return {f'p{p}_ms':round(float(np.percentile(lat,p)),4) for p in [50,95,99]}|{'qps':round(float(len(lat)/max(sum(lat)/1000,1e-12)),2)}

def main():
 ap=argparse.ArgumentParser()
 ap.add_argument('--n',type=int,default=100000);ap.add_argument('--nq',type=int,default=100)
 ap.add_argument('--k',type=int,default=10);ap.add_argument('--posting',type=int,default=20000);ap.add_argument('--rerank',type=int,default=1000)
 ap.add_argument('--groups',type=int,nargs='+',default=[2,4,8]);ap.add_argument('--bins',type=int,nargs='+',default=[8,16])
 ap.add_argument('--postings',type=int,nargs='+');ap.add_argument('--reranks',type=int,nargs='+')
 ap.add_argument('--mode',choices=['gaussian','correlated','normalized'],default='gaussian');ap.add_argument('--seed',type=int,default=42)
 ap.add_argument('--out',default='results/benchmark.csv');ap.add_argument('--threads',type=int,default=1)
 args=ap.parse_args();os.environ['OMP_NUM_THREADS']=str(args.threads)
 try:import faiss
 except ImportError:raise SystemExit('Install faiss-cpu: pip install -r requirements.txt')
 faiss.omp_set_num_threads(args.threads)
 x,q=make_data(args.n,args.nq,args.seed,args.mode)
 flat=faiss.IndexFlatL2(128);flat.add(x)
 truth_d,truth=flat.search(q,args.k)
 rows=[]
 def measure(name,search,memory,build_s,params):
  for _ in range(min(5,args.nq)):search(q[_])
  latency=[];recs=[];entries=[];unique=[];reranked=[];probes=[]
  for i in range(args.nq):
   start=time.perf_counter_ns();ids,st=search(q[i]);latency.append((time.perf_counter_ns()-start)/1e6)
   recs.append(len(set(ids[ids>=0])&set(truth[i]))/args.k)
   if st is not None:entries.append(st[0]);unique.append(st[1]);reranked.append(st[2]);probes.append(st[3])
  row={'algorithm':name,'n':args.n,'nq':args.nq,'k':args.k,'mode':args.mode,'recall':round(float(np.mean(recs)),5),'build_s':round(build_s,3),'index_bytes':int(memory),**stats_summary(latency),**params}
  if entries:row.update(postings_read_mean=float(np.mean(entries)),candidates_mean=float(np.mean(unique)),reranked_mean=float(np.mean(reranked)),probes_mean=float(np.mean(probes)))
  rows.append(row);print(json.dumps(row),flush=True)
 measure('FlatL2',lambda v:(flat.search(v[None],args.k)[1][0],None),args.n*128*4,0,{})
 start=time.perf_counter();h=faiss.IndexHNSWFlat(128,32);h.hnsw.efConstruction=200;h.add(x);build=time.perf_counter()-start
 for ef in [64,128,256,512]:
  h.hnsw.efSearch=ef
  measure('HNSWFlat',lambda v:(h.search(v[None],args.k)[1][0],None),faiss.serialize_index(h).nbytes,build,{'efSearch':ef,'M':32})
 nlist=min(256,max(4,int(np.sqrt(args.n))))
 for kind in ['IVFFlat','IVFPQ']:
  quant=faiss.IndexFlatL2(128)
  ix=faiss.IndexIVFFlat(quant,128,nlist) if kind=='IVFFlat' else faiss.IndexIVFPQ(quant,128,nlist,16,8)
  start=time.perf_counter();ix.train(x);ix.add(x);build=time.perf_counter()-start
  for npb in [4,8,16,32,64,128]:
   ix.nprobe=min(npb,nlist)
   measure(kind,lambda v:(ix.search(v[None],args.k)[1][0],None),faiss.serialize_index(ix).nbytes,build,{'nlist':nlist,'nprobe':ix.nprobe})
 for gs in args.groups:
  for bins in args.bins:
   start=time.perf_counter();ix=ACI(x,gs,bins);build=time.perf_counter()-start
   for posting in args.postings or [args.posting]:
    for rerank in args.reranks or [args.rerank]:
     measure('ACI-MP',lambda v:(lambda z:(z[0],z[2]))(ix.search(v,args.k,posting,rerank)),ix.memory(),build,{'group_size':gs,'bins':bins,'posting_budget':posting,'rerank_budget':rerank})
   ix.close()
 path=Path(args.out);path.parent.mkdir(parents=True,exist_ok=True)
 fields=list(dict.fromkeys(key for row in rows for key in row))
 with path.open('w',newline='') as f:
  writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(rows)
 print('Saved:',path,'peak_rss_kib:',resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
 print('95%-recall qualifying methods:',[(r['algorithm'],r['p95_ms'],r.get('group_size'),r.get('efSearch')) for r in rows if r['recall']>=.95])
if __name__=='__main__':main()
