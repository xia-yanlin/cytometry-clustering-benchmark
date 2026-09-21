from __future__ import annotations
import argparse,hashlib,json,platform,sys
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score

METRICS=("ari","target_precision","target_recall","target_f1","target_f2")
CONTRASTS=((10,2),(40,2),(40,10))
def sha256(p:Path)->str:
 h=hashlib.sha256()
 with p.open("rb") as f:
  while b:=f.read(1024*1024):h.update(b)
 return h.hexdigest()
def write_json(p:Path,v:object):p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
def main()->int:
 ap=argparse.ArgumentParser();ap.add_argument("--parent",type=Path,required=True);ap.add_argument("--protocol",type=Path,required=True);ap.add_argument("--output",type=Path,required=True);a=ap.parse_args();parent,protocol,output=a.parent.resolve(),a.protocol.resolve(),a.output.resolve();output.mkdir(parents=True,exist_ok=False)
 progress=json.loads((parent/"progress_manifest.json").read_text(encoding="utf-8"));index=pd.read_csv(parent/"run_index.csv")
 if progress.get("all_runs_complete_and_passed") is not True or len(index)!=180:raise RuntimeError("EXP-034 parent not eligible")
 summaries=[]
 for (d,k),g in index.groupby(["dataset","k"],sort=True):
  for metric in (*METRICS,"lower_bound"):
   x=g[metric].to_numpy(float);summaries.append({"dataset":d,"k":int(k),"metric":metric,"n":len(x),"mean":float(x.mean()),"sd":float(x.std(ddof=1)),"median":float(np.median(x)),"q25":float(np.quantile(x,.25)),"q75":float(np.quantile(x,.75)),"min":float(x.min()),"max":float(x.max())})
 summary=pd.DataFrame(summaries);summary.to_csv(output/"distribution_summary.csv",index=False)
 rng=np.random.default_rng(20_260_911);paired=[];contrast=[]
 for d in ("Nilsson_rare","Mosmann_rare"):
  p=index[index.dataset==d].pivot(index="seed",columns="k",values=list(METRICS))
  for hi,lo in CONTRASTS:
   for metric in METRICS:
    diff=p[(metric,hi)].to_numpy()-p[(metric,lo)].to_numpy()
    for seed,val in zip(p.index,diff,strict=True):paired.append({"dataset":d,"contrast":f"K{hi}_minus_K{lo}","metric":metric,"seed":int(seed),"difference":float(val)})
    draws=rng.choice(diff,size=(10000,len(diff)),replace=True).mean(axis=1);contrast.append({"dataset":d,"contrast":f"K{hi}_minus_K{lo}","metric":metric,"n":len(diff),"mean_difference":float(diff.mean()),"sd_difference":float(diff.std(ddof=1)),"bootstrap_ci_low":float(np.quantile(draws,.025)),"bootstrap_ci_high":float(np.quantile(draws,.975)),"bootstrap_replicates":10000})
 pd.DataFrame(paired).to_csv(output/"paired_differences.csv",index=False);pd.DataFrame(contrast).to_csv(output/"paired_contrast_summary.csv",index=False)
 pairs=[];pair_summary=[];parent_hashes=[]
 for d in ("Nilsson_rare","Mosmann_rare"):
  for k in (2,10,40):
   arrays=[]
   for seed in range(30):
    path=parent/"runs"/d/f"K{k:02d}"/f"seed{seed:03d}"/"component_labels_all_events.npy";arrays.append(np.load(path,allow_pickle=False));parent_hashes.append({"dataset":d,"k":k,"seed":seed,"path":str(path),"bytes":path.stat().st_size,"sha256":sha256(path)})
   vals=[]
   for i in range(30):
    for j in range(i+1,30):
     v=float(adjusted_rand_score(arrays[i],arrays[j]));pairs.append({"dataset":d,"k":k,"seed_a":i,"seed_b":j,"partition_ari":v});vals.append(v)
   x=np.asarray(vals);pair_summary.append({"dataset":d,"k":k,"pairs":len(x),"mean":float(x.mean()),"sd":float(x.std(ddof=1)),"median":float(np.median(x)),"min":float(x.min()),"max":float(x.max()),"exact_one_pairs":int(np.sum(x==1))})
 pd.DataFrame(pairs).to_csv(output/"partition_pairwise_ari.csv",index=False);pd.DataFrame(pair_summary).to_csv(output/"partition_pairwise_summary.csv",index=False);pd.DataFrame(parent_hashes).to_csv(output/"parent_prediction_hashes.csv",index=False)
 extremes=[]
 for (d,k),g in index.groupby(["dataset","k"],sort=True):
  for direction,idx in (("min",g.target_f1.idxmin()),("max",g.target_f1.idxmax())):
   r=index.loc[idx];extremes.append({"dataset":d,"k":int(k),"direction":direction,"seed":int(r.seed),"target_f1":float(r.target_f1),"target_precision":float(r.target_precision),"target_recall":float(r.target_recall),"ari":float(r.ari)})
 pd.DataFrame(extremes).to_csv(output/"f1_extreme_runs.csv",index=False)
 checks=[]
 def check(n,p,d):checks.append({"check":n,"passed":bool(p),"detail":d})
 check("parent",len(index)==180 and index.all_checks_passed.astype(bool).all() and index.converged.astype(bool).all(),f"rows={len(index)}");check("cells",set(index.groupby(["dataset","k"]).size())=={30},str(index.groupby(["dataset","k"]).size().to_dict()));check("summary",len(summary)==36,"6 cells x 6 metrics");check("paired",len(paired)==900 and len(contrast)==30,f"{len(paired)};{len(contrast)}");check("pairwise",len(pairs)==2610 and all(-1<=r["partition_ari"]<=1 for r in pairs),str(len(pairs)));check("hashes",len(parent_hashes)==180 and all(sha256(Path(r["path"]))==r["sha256"] for r in parent_hashes),str(len(parent_hashes)));check("label_permission",index.loc[index.k==2,"label_information_used_for_regime_definition"].astype(bool).all() and not index.loc[index.k!=2,"label_information_used_for_regime_definition"].astype(bool).any(),"K2 only");check("metric_ranges",index.ari.between(-1,1).all() and all(index[m].between(0,1).all() for m in METRICS[1:]),"legal")
 cf=pd.DataFrame(checks);cf.to_csv(output/"checks.csv",index=False);result={"experiment_id":"EXP-034A","completed_utc":datetime.now(timezone.utc).isoformat(),"parent":str(parent),"parent_progress_sha256":sha256(parent/"progress_manifest.json"),"protocol":str(protocol),"protocol_sha256":sha256(protocol),"script":str(Path(__file__).resolve()),"script_sha256":sha256(Path(__file__).resolve()),"checks_passed":int(cf.passed.sum()),"checks_total":len(cf),"all_checks_passed":bool(cf.passed.all()),"python":sys.version,"python_executable":sys.executable,"platform":platform.platform()};write_json(output/"run_manifest.json",result)
 (output/"scientific_summary.md").write_text(f"# EXP-034A GMM稀有K敏感性\n\n180次父运行；900个配对差、30个bootstrap摘要、2,610个事件分区ARI；检查{result['checks_passed']}/{result['checks_total']}。\n",encoding="utf-8");write_json(output/"artifact_hashes.json",{p.name:sha256(p) for p in output.iterdir() if p.is_file()});print(json.dumps(result,ensure_ascii=False));return 0 if result["all_checks_passed"] else 1
if __name__=="__main__":raise SystemExit(main())
