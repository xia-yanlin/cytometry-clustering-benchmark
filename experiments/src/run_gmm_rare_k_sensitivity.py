from __future__ import annotations

import argparse, hashlib, json, platform, sys, time, traceback
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn import __version__ as sklearn_version
from sklearn.metrics import adjusted_rand_score
from sklearn.mixture import GaussianMixture

DATASETS={
 "Nilsson_rare":{"rows":44140,"markers":13,"target":"HSCs","target_events":358,"sha":"e74c804a6cb040fd5cf1bea3c9d1ce382b266d860209d443f7f3960f129c918c"},
 "Mosmann_rare":{"rows":396460,"markers":14,"target":"activated","target_events":109,"sha":"684541587d408477b89da4882e237000ab03353bc9cb440cd4a1bc27736d81ae"},
}
KS=(2,10,40);SEEDS=tuple(range(30));EXPECTED=180

def sha256(path:Path)->str:
 h=hashlib.sha256()
 with path.open("rb") as f:
  while b:=f.read(1024*1024):h.update(b)
 return h.hexdigest()
def write_json(path:Path,value:object)->None:path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
def effective(counts:np.ndarray)->float:
 p=counts[counts>0].astype(float);p/=p.sum();return float(np.exp(-np.sum(p*np.log(p))))
def evaluate(labels:np.ndarray,clusters:np.ndarray,target:str)->tuple[dict[str,object],pd.DataFrame]:
 unique,inverse=np.unique(clusters,return_inverse=True);sizes=np.bincount(inverse);actual=labels==target;n=int(actual.sum());tp=np.bincount(inverse[actual],minlength=len(unique));fp=sizes-tp;fn=n-tp
 precision=np.divide(tp,tp+fp,out=np.zeros(len(unique),float),where=(tp+fp)>0);recall=tp/n
 f1=np.divide(2*precision*recall,precision+recall,out=np.zeros(len(unique),float),where=(precision+recall)>0);f2=np.divide(5*precision*recall,4*precision+recall,out=np.zeros(len(unique),float),where=(4*precision+recall)>0)
 order=np.lexsort((unique,-recall,-f1));b=int(order[0]);diag=pd.DataFrame({"component":unique,"component_size":sizes,"tp":tp,"fp":fp,"fn":fn,"precision":precision,"recall":recall,"f1":f1,"f2":f2})
 return {"n_total_events":len(labels),"n_components_occupied":len(unique),"target":target,"target_events":n,"target_prevalence":float(actual.mean()),"ari":float(adjusted_rand_score(labels,clusters)),"selected_target_component":int(unique[b]),"selected_component_size":int(sizes[b]),"target_tp":int(tp[b]),"target_fp":int(fp[b]),"target_fn":int(fn[b]),"target_precision":float(precision[b]),"target_recall":float(recall[b]),"target_f1":float(f1[b]),"target_f2":float(f2[b]),"target_overlapping_components":int(np.sum(tp>0)),"target_effective_components":effective(tp),"target_dominant_component_capture":float(tp.max()/n)},diag
def load_data(config:Path,dataset:str):
 cfg=json.loads(config.read_text(encoding="utf-8"));spec=cfg["datasets"][dataset];source=(Path(cfg["data_root"])/dataset/spec["filename"]).resolve();frame=pd.read_csv(source,sep="," if spec["separator"]=="comma" else "\t",low_memory=False);markers=[c for c in frame.columns if c not in set(spec["exclude_columns"])];matrix=np.arcsinh(frame[markers].to_numpy(float)/float(spec["cofactor"]));labels=frame.label.astype(str).to_numpy();return source,markers,matrix,labels
def rebuild(output:Path)->pd.DataFrame:
 rows=[]
 for d in DATASETS:
  for k in KS:
   for seed in SEEDS:
    run=output/"runs"/d/f"K{k:02d}"/f"seed{seed:03d}";mp=run/"run_manifest.json"
    if mp.is_file():
     m=json.loads(mp.read_text(encoding="utf-8"));v=pd.read_csv(run/"run_level_metrics.csv").iloc[0].to_dict();rows.append({"dataset":d,"k":k,"seed":seed,"runtime_seconds":m["runtime_seconds"],"converged":m["converged"],"n_iter":m["n_iter"],"lower_bound":m["lower_bound"],"all_checks_passed":m["all_checks_passed"],**v})
 f=pd.DataFrame(rows);f.to_csv(output/"run_index.csv",index=False);return f
def progress(output:Path,protocol:Path,config:Path,parent:Path,failed:str|None=None)->dict[str,object]:
 idx=rebuild(output);passed=int(idx.all_checks_passed.astype(bool).sum()) if len(idx) else 0;r={"experiment_id":"EXP-034","updated_utc":datetime.now(timezone.utc).isoformat(),"status":"failed" if failed else ("complete" if len(idx)==passed==EXPECTED else "in_progress"),"protocol":str(protocol),"protocol_sha256":sha256(protocol),"script":str(Path(__file__).resolve()),"script_sha256":sha256(Path(__file__).resolve()),"config":str(config),"config_sha256":sha256(config),"training_index_parent":str(parent),"training_index_parent_progress_sha256":sha256(parent/"progress_manifest.json"),"datasets":list(DATASETS),"k_values":list(KS),"seeds":list(SEEDS),"expected_runs":EXPECTED,"completed_runs":len(idx),"passed_runs":passed,"failed_run":failed,"all_runs_complete_and_passed":len(idx)==passed==EXPECTED and failed is None};write_json(output/"progress_manifest.json",r);return r
def main()->int:
 p=argparse.ArgumentParser();p.add_argument("--config",type=Path,required=True);p.add_argument("--protocol",type=Path,required=True);p.add_argument("--training-index-parent",type=Path,required=True);p.add_argument("--output",type=Path,required=True);a=p.parse_args();config,protocol,parent,output=a.config.resolve(),a.protocol.resolve(),a.training_index_parent.resolve(),a.output.resolve()
 pm=json.loads((parent/"progress_manifest.json").read_text(encoding="utf-8"));
 if pm.get("all_runs_complete_and_passed") is not True:raise RuntimeError("EXP-033A training index parent not eligible")
 parent_inventory=json.loads((parent/"artifact_hashes.json").read_text(encoding="utf-8"))
 if output.exists():
  old=json.loads((output/"progress_manifest.json").read_text(encoding="utf-8"));
  if old.get("protocol_sha256")!=sha256(protocol) or old.get("script_sha256")!=sha256(Path(__file__).resolve()):raise RuntimeError("resume contract differs")
 else:(output/"runs").mkdir(parents=True)
 progress(output,protocol,config,parent)
 for d,c in DATASETS.items():
  source,markers,matrix,labels=load_data(config,d);idx_path=parent/"runs"/d/"training_indices.npy";indices=np.load(idx_path,allow_pickle=False)
  if sha256(source)!=c["sha"] or matrix.shape!=(c["rows"],c["markers"]) or int(np.sum(labels==c["target"]))!=c["target_events"] or len(indices)!=20000:raise ValueError(f"input contract failed:{d}")
  train=matrix[indices]
  for k in KS:
   for seed in SEEDS:
    run=output/"runs"/d/f"K{k:02d}"/f"seed{seed:03d}"
    if (run/"run_manifest.json").is_file():continue
    if run.exists():raise RuntimeError(f"incomplete run preserved:{run}")
    run.mkdir(parents=True)
    try:
     started=time.perf_counter();model=GaussianMixture(n_components=k,covariance_type="full",tol=1e-3,reg_covar=1e-6,max_iter=500,n_init=1,init_params="kmeans",random_state=seed).fit(train);pred=model.predict(matrix).astype(np.int16);runtime=time.perf_counter()-started
     metrics,diag=evaluate(labels,pred,c["target"]);np.save(run/"component_labels_all_events.npy",pred);np.save(run/"weights.npy",model.weights_);np.save(run/"means.npy",model.means_);np.save(run/"covariances.npy",model.covariances_);diag.to_csv(run/"component_target_diagnostics.csv",index=False)
     checks=[]
     def check(n,p,d):checks.append({"check":n,"passed":bool(p),"detail":d})
     index_key=str(idx_path.relative_to(parent));check("source_hash",sha256(source)==c["sha"],sha256(source));check("training_indices",parent_inventory.get(index_key)==sha256(idx_path),f"{index_key}:{sha256(idx_path)}");check("converged",bool(model.converged_),f"iter={model.n_iter_}");check("event_shape",pred.shape==(c["rows"],),str(pred.shape));check("parameter_shapes",model.weights_.shape==(k,) and model.means_.shape==(k,c["markers"]) and model.covariances_.shape==(k,c["markers"],c["markers"]),str(model.means_.shape));check("component_contract",pred.min()>=0 and pred.max()<k and np.unique(pred).size==k,f"occupied={np.unique(pred).size}");check("target_counts",metrics["target_events"]==c["target_events"] and metrics["target_tp"]+metrics["target_fn"]==c["target_events"],str(metrics["target_events"]));check("metric_ranges",-1<=metrics["ari"]<=1 and all(0<=metrics[x]<=1 for x in ("target_precision","target_recall","target_f1","target_f2")),"legal");check("label_permission",True,"labels used only for input count and posthoc evaluation; K2 flag recorded")
     cf=pd.DataFrame(checks);cf.to_csv(run/"checks.csv",index=False);row={"dataset":d,"algorithm":"sklearn_GaussianMixture_1.6.1","k":k,"seed":seed,"label_information_used_for_regime_definition":k==2,"labels_used_for_sampling_or_fit":False,"training_size":20000,"converged":bool(model.converged_),"n_iter":int(model.n_iter_),"lower_bound":float(model.lower_bound_),"runtime_seconds":runtime,**metrics};pd.DataFrame([row]).to_csv(run/"run_level_metrics.csv",index=False)
     m={"experiment_id":"EXP-034","dataset":d,"k":k,"seed":seed,"completed_utc":datetime.now(timezone.utc).isoformat(),"source":str(source),"source_sha256":sha256(source),"training_indices":str(idx_path),"training_indices_sha256":sha256(idx_path),"protocol":str(protocol),"protocol_sha256":sha256(protocol),"script":str(Path(__file__).resolve()),"script_sha256":sha256(Path(__file__).resolve()),"fit_policy":"fixed_unlabeled_20000_then_predict_all_events","label_information_used_for_regime_definition":k==2,"labels_used_for_sampling_or_fit":False,"n_components":k,"covariance_type":"full","tol":1e-3,"reg_covar":1e-6,"max_iter":500,"n_init":1,"init_params":"kmeans","converged":bool(model.converged_),"n_iter":int(model.n_iter_),"lower_bound":float(model.lower_bound_),"runtime_seconds":runtime,"sklearn":sklearn_version,"python":sys.version,"python_executable":sys.executable,"platform":platform.platform(),"checks_passed":int(cf.passed.sum()),"checks_total":len(cf),"all_checks_passed":bool(cf.passed.all())};write_json(run/"run_manifest.json",m);write_json(run/"artifact_hashes.json",{x.name:sha256(x) for x in run.iterdir() if x.is_file()})
     if not m["all_checks_passed"]:raise RuntimeError("run checks failed")
     print(json.dumps({"dataset":d,"k":k,"seed":seed,"runtime":runtime,"converged":model.converged_,"f1":metrics["target_f1"],"checks":"9/9"},ensure_ascii=False),flush=True)
    except Exception as e:
     (run/"failure_traceback.txt").write_text(traceback.format_exc(),encoding="utf-8");write_json(run/"failure.json",{"exception":repr(e),"scientific_output_eligible":False});progress(output,protocol,config,parent,str(run.relative_to(output)));raise
    progress(output,protocol,config,parent)
 final=progress(output,protocol,config,parent);art=[x for x in output.rglob("*") if x.is_file() and x.name!="artifact_hashes.json"];write_json(output/"artifact_hashes.json",{str(x.relative_to(output)):sha256(x) for x in art});print(json.dumps(final,ensure_ascii=False));return 0 if final["all_runs_complete_and_passed"] else 1
if __name__=="__main__":raise SystemExit(main())
