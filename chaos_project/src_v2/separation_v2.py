"""Check the separation assumption (Prop. 2) and count local optima on the ground-truth tables."""
import os, json, itertools, numpy as np
HERE=os.path.dirname(os.path.abspath(__file__)); DATA=os.path.join(HERE,'..','data_v2'); OUT=os.path.join(HERE,'..','..','results')
N=14; n=2**N
nb=[]
for d in (1,2):
    for c in itertools.combinations(range(N),d): nb.append(sum(1<<b for b in c))
nb=np.array(nb)
idx=np.arange(n)
NB=idx[:,None]^nb[None,:]  # radius-2 neighbours
out={}
for key in ['chaos_orig','poisson_orig','self_similar_orig','chaos_zero_floor']:
    g=np.load(os.path.join(DATA,f'gt_{key}.npy'))
    unstable=g[:,3]>0.5   # majority of long runs flag instability
    for col,name in [(1,'pen'),(0,'perf')]:
        U=g[:,col]
        locopt=U>=U[NB].max(1)
        better_stable=((U[NB]>U[:,None]) & (~unstable[NB])).any(1)
        sep_fail=unstable & ~better_stable
        out[f'{key}_{name}']={'n_unstable':int(unstable.sum()),'n_local_opt':int(locopt.sum()),
            'n_local_opt_unstable':int((locopt&unstable).sum()),'n_sep_violations':int(sep_fail.sum()),
            'global_opt_unstable':bool(unstable[U.argmax()])}
s=json.load(open(os.path.join(OUT,'summary_v2.json')))
d=json.load(open(os.path.join(DATA,'allocation_v2.json')));H=d['horizons'];R=d['results']
def ci(x): x=np.asarray(x); return [round(x.mean(),3), round(1.96*x.std(ddof=1)/np.sqrt(len(x)),3)]
for obj in ['pen','perf']:
    rnd=np.array([r['shipped'] for r in R[f'{obj}|random']])*100
    for k in [1,2,3,4,5]:
        a=np.array([r['shipped'] for r in R[f'{obj}|new{k}']])*100
        out[f'{obj}_new{k}_vs_random']={h:ci(a[:,j]-rnd[:,j]) for j,h in enumerate(H)}
        out[f'{obj}_new{k}_shipped']={h:ci(a[:,j]) for j,h in enumerate(H)}
json.dump(out,open(os.path.join(OUT,'separation_v2.json'),'w'),indent=1)
for k,v in out.items(): print(k,v)
