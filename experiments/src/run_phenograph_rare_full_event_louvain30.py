from __future__ import annotations
import argparse,hashlib,json,platform,sys,time,traceback
from datetime import datetime,timezone
from importlib.metadata import version
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import sparse as sp
from sklearn.metrics import adjusted_rand_score
from phenograph.cluster import run_louvain,sort_by_size
from phenograph.core import find_neighbors,jaccard_kernel,neighbor_graph

CONTRACTS={"Nilsson_rare":{"rows":44140,"markers":13,"target":"HSCs","target_events":358,"sha":"e74c804a6cb040fd5cf1bea3c9d1ce382b266d860209d443f7f3960f129c918c"},"Mosmann_rare":{"rows":396460,"markers":14,"target":"activated","target_events":109,"sha":"684541587d408477b89da4882e237000ab03353bc9cb440cd4a1bc27736d81ae"}};REPEATS=tuple(range(30));K=30;MIN_SIZE=10
def sha256(p:Path)->str:
 h=hashlib.sha256()
 with p.open("rb") as f:
  while b:=f.read(1024*1024):h.update(b)
 return h.hexdigest()
def write_json(p:Path,v:object):p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
def evaluate(labels,clusters,target):
 actual=labels==target;n=int(actual.sum());all_ari=float(adjusted_rand_score(labels,clusters));rows=[]
 for c in np.unique(clusters):
  pred=clusters==c;tp=int(np.sum(actual&pred));fp=int(np.sum(~actual&pred));fn=n-tp;p=tp/(tp+fp) if tp+fp else 0;r=tp/n;f1=2*p*r/(p+r) if p+r else 0;f2=5*p*r/(4*p+r) if 4*p+r else 0;rows.append({"community":int(c),"community_size":int(pred.sum()),"is_outlier":c<0,"tp":tp,"fp":fp,"fn":fn,"precision":p,"recall":r,"f1":f1,"f2":f2})
 frame=pd.DataFrame(rows);valid=frame[~frame.is_outlier].sort_values(["f1","recall","community"],ascending=[False,False,True]);best=valid.iloc[0];target_valid=frame.loc[~frame.is_outlier,"tp"].to_numpy(int);out=int(frame.loc[frame.is_outlier,"tp"].sum());captured=int(target_valid.sum());q=target_valid[target_valid>0]/captured if captured else np.array([],float);eff=float(np.exp(-np.sum(q*np.log(q)))) if len(q) else 0
 return {"n_total_events":len(labels),"n_valid_communities":int(np.sum(np.unique(clusters)>=0)),"outlier_events":int(np.sum(clusters<0)),"target":target,"target_events":n,"target_outlier_events":out,"target_outlier_fraction":out/n,"ari":all_ari,"selected_target_community":int(best.community),"selected_community_size":int(best.community_size),"target_tp":int(best.tp),"target_fp":int(best.fp),"target_fn":int(best.fn),"target_precision":float(best.precision),"target_recall":float(best.recall),"target_f1":float(best.f1),"target_f2":float(best.f2),"target_overlapping_valid_communities":int(np.sum(target_valid>0)),"target_effective_valid_communities":eff,"target_dominant_valid_community_capture":float(target_valid.max()/n) if len(target_valid) else 0},frame
def load(config,d):
 cfg=json.loads(config.read_text(encoding="utf-8"));s=cfg["datasets"][d];p=(Path(cfg["data_root"])/d/s["filename"]).resolve();f=pd.read_csv(p,sep="," if s["separator"]=="comma" else "\t",low_memory=False);m=[x for x in f.columns if x not in set(s["exclude_columns"])];return p,m,np.arcsinh(f[m].to_numpy(float)/float(s["cofactor"])),f.label.astype(str).to_numpy()
def index(output):
 rows=[]
 for r in REPEATS:
  d=output/"runs"/f"repeat{r:03d}";mp=d/"run_manifest.json"
  if mp.is_file():m=json.loads(mp.read_text(encoding="utf-8"));v=pd.read_csv(d/"run_level_metrics.csv").iloc[0].to_dict();rows.append({"repeat":r,"runtime_seconds":m["runtime_seconds"],"all_checks_passed":m["all_checks_passed"],**v})
 f=pd.DataFrame(rows);f.to_csv(output/"run_index.csv",index=False);return f
def progress(output,expid,protocol,qualification,source,graph,neighbors,failed=None):
 ix=index(output);passed=int(ix.all_checks_passed.astype(bool).sum()) if len(ix) else 0;r={"experiment_id":expid,"updated_utc":datetime.now(timezone.utc).isoformat(),"status":"failed" if failed else ("complete" if len(ix)==passed==30 else "in_progress"),"protocol":str(protocol),"protocol_sha256":sha256(protocol),"script":str(Path(__file__).resolve()),"script_sha256":sha256(Path(__file__).resolve()),"qualification":str(qualification),"qualification_manifest_sha256":sha256(qualification/"run_manifest.json"),"source":str(source),"source_sha256":sha256(source),"fixed_graph":str(graph),"fixed_graph_sha256":sha256(graph),"neighbors":str(neighbors),"neighbors_sha256":sha256(neighbors),"k":K,"min_cluster_size":MIN_SIZE,"q_tol":1e-3,"louvain_time_limit":2000,"evaluation_scope":"all_events","repeat_indices":list(REPEATS),"seed_control":"unavailable","expected_runs":30,"completed_runs":len(ix),"passed_runs":passed,"failed_run":failed,"all_runs_complete_and_passed":len(ix)==passed==30 and failed is None};write_json(output/"progress_manifest.json",r);return r
def main():
 ap=argparse.ArgumentParser();ap.add_argument("--config",type=Path,required=True);ap.add_argument("--dataset",choices=sorted(CONTRACTS),required=True);ap.add_argument("--protocol",type=Path,required=True);ap.add_argument("--qualification",type=Path,required=True);ap.add_argument("--experiment-id",required=True);ap.add_argument("--output",type=Path,required=True);a=ap.parse_args();config,protocol,qualification,output=a.config.resolve(),a.protocol.resolve(),a.qualification.resolve(),a.output.resolve();c=CONTRACTS[a.dataset];qm=json.loads((qualification/"run_manifest.json").read_text(encoding="utf-8"))
 if qm.get("experiment_id")!="EXP-016-R2" or qm.get("all_checks_passed") is not True:raise RuntimeError("PhenoGraph default Louvain qualification required")
 source,markers,matrix,labels=load(config,a.dataset)
 if sha256(source)!=c["sha"] or matrix.shape!=(c["rows"],c["markers"]) or int(np.sum(labels==c["target"]))!=c["target_events"]:raise ValueError("dataset contract failed")
 if output.exists():
  old=json.loads((output/"progress_manifest.json").read_text(encoding="utf-8")) if (output/"progress_manifest.json").is_file() else None
  if old is not None and (old.get("protocol_sha256")!=sha256(protocol) or old.get("experiment_id")!=a.experiment_id):raise RuntimeError("resume contract differs")
 else:(output/"runs").mkdir(parents=True)
 graph=output/"fixed_full_event_graph.npz";neighbors=output/"neighbor_indices_k30.npy"
 if not graph.is_file():
  started=time.perf_counter();_,nn=find_neighbors(matrix,k=K,metric="euclidean",method="kdtree",n_jobs=1);np.save(neighbors,nn.astype(np.int32));raw=neighbor_graph(jaccard_kernel,{"idx":nn});g=sp.tril((raw+raw.transpose()).multiply(.5),-1).tocoo();sp.save_npz(graph,g);write_json(output/"graph_manifest.json",{"build_seconds":time.perf_counter()-started,"shape":list(g.shape),"nnz":int(g.nnz),"weight_min":float(g.data.min()),"weight_max":float(g.data.max()),"strict_lower_triangle":bool(np.all(g.row>g.col)),"finite_positive_weights":bool(np.isfinite(g.data).all() and np.all(g.data>0)),"labels_used":False})
 g=sp.load_npz(graph).tocoo();nn=np.load(neighbors,allow_pickle=False)
 if g.shape!=(c["rows"],c["rows"]) or nn.shape!=(c["rows"],K) or not np.all(g.row>g.col):raise ValueError("graph contract failed")
 progress(output,a.experiment_id,protocol,qualification,source,graph,neighbors)
 for repeat in REPEATS:
  run=output/"runs"/f"repeat{repeat:03d}"
  if (run/"run_manifest.json").is_file():continue
  if run.exists():raise RuntimeError(f"incomplete run preserved:{run}")
  run.mkdir()
  try:
   started=time.perf_counter();raw,q=run_louvain(g,1e-3,2000);communities=np.asarray(sort_by_size(np.asarray(raw),MIN_SIZE),dtype=np.int32);runtime=time.perf_counter()-started;metrics,diag=evaluate(labels,communities,c["target"]);np.save(run/"raw_membership_all_events.npy",np.asarray(raw,dtype=np.int32));np.save(run/"community_labels_all_events.npy",communities);diag.to_csv(run/"community_target_diagnostics.csv",index=False)
   checks=[]
   def chk(n,p,d):checks.append({"check":n,"passed":bool(p),"detail":d})
   chk("qualification",qm["all_checks_passed"] is True,"EXP-016-R2");chk("source",sha256(source)==c["sha"],sha256(source));chk("graph",g.shape==(c["rows"],c["rows"]) and np.all(g.row>g.col),f"nnz={g.nnz}");chk("neighbors",nn.shape==(c["rows"],K),str(nn.shape));chk("membership",communities.shape==(c["rows"],) and communities.min()>=-1,str(communities.shape));chk("relabel_partition",adjusted_rand_score(raw,communities)==1,"ARI=1");chk("quality",np.isfinite(q),str(q));chk("target",metrics["target_events"]==c["target_events"] and metrics["target_tp"]+metrics["target_fn"]==c["target_events"],str(metrics["target_events"]));chk("metrics",-1<=metrics["ari"]<=1 and all(0<=metrics[x]<=1 for x in ("target_precision","target_recall","target_f1","target_f2")),"legal");chk("labels_permission",True,"labels used only for input target count and posthoc evaluation")
   cf=pd.DataFrame(checks);cf.to_csv(run/"checks.csv",index=False);pd.DataFrame([{"dataset":a.dataset,"algorithm":"PhenoGraph_v1.5.7_default_Louvain","repeat":repeat,"seed_control":"unavailable","k_neighbors":K,"evaluation_scope":"all_events","quality_q":float(q),"runtime_seconds":runtime,**metrics}]).to_csv(run/"run_level_metrics.csv",index=False);m={"experiment_id":a.experiment_id,"dataset":a.dataset,"repeat":repeat,"completed_utc":datetime.now(timezone.utc).isoformat(),"source":str(source),"source_sha256":sha256(source),"graph_sha256":sha256(graph),"neighbors_sha256":sha256(neighbors),"protocol":str(protocol),"protocol_sha256":sha256(protocol),"script":str(Path(__file__).resolve()),"script_sha256":sha256(Path(__file__).resolve()),"qualification":str(qualification),"qualification_manifest_sha256":sha256(qualification/"run_manifest.json"),"k_neighbors":K,"q_tol":1e-3,"louvain_time_limit":2000,"min_cluster_size":MIN_SIZE,"seed_control":"unavailable","labels_used_for_graph_or_clustering":False,"runtime_seconds":runtime,"quality_q":float(q),"packages":{x:version(x) for x in ("PhenoGraph","numpy","scipy","scikit-learn")},"python":sys.version,"python_executable":sys.executable,"platform":platform.platform(),"checks_passed":int(cf.passed.sum()),"checks_total":len(cf),"all_checks_passed":bool(cf.passed.all())};write_json(run/"run_manifest.json",m);write_json(run/"artifact_hashes.json",{p.name:sha256(p) for p in run.iterdir() if p.is_file()});
   if not m["all_checks_passed"]:raise RuntimeError("checks failed")
   print(json.dumps({"dataset":a.dataset,"repeat":repeat,"runtime":runtime,"q":q,"communities":metrics["n_valid_communities"],"target_f1":metrics["target_f1"],"checks":"10/10"},ensure_ascii=False),flush=True)
  except Exception as e:
   (run/"failure_traceback.txt").write_text(traceback.format_exc(),encoding="utf-8");write_json(run/"failure.json",{"exception":repr(e),"scientific_output_eligible":False});progress(output,a.experiment_id,protocol,qualification,source,graph,neighbors,str(run.relative_to(output)));raise
  progress(output,a.experiment_id,protocol,qualification,source,graph,neighbors)
 final=progress(output,a.experiment_id,protocol,qualification,source,graph,neighbors);art=[p for p in output.rglob("*") if p.is_file() and p.name!="artifact_hashes.json"];write_json(output/"artifact_hashes.json",{str(p.relative_to(output)):sha256(p) for p in art});print(json.dumps(final,ensure_ascii=False));return 0 if final["all_runs_complete_and_passed"] else 1
if __name__=="__main__":raise SystemExit(main())
