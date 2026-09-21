from __future__ import annotations

import argparse, hashlib, json, platform, subprocess, sys, time, zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score

from evaluate_xshift_cross_dataset import evaluate_multiclass
from run_xshift_nilsson_fixedK20_30repeat import bootstrap_ci, class_methodrefs, read_fcs, sha256


def main()->int:
    ap=argparse.ArgumentParser()
    for name in ("parent","source-repo","release-jar","java","data","protocol","output"): ap.add_argument("--"+name,type=Path,required=True)
    ap.add_argument("--experiment-id",required=True);ap.add_argument("--dataset",required=True);ap.add_argument("--parent-id",required=True);ap.add_argument("--input-fcs-name",required=True)
    ap.add_argument("--separator",choices=["comma","tab"],required=True);ap.add_argument("--label-mode",choices=["numeric_1_to_n","not_unassigned"],required=True)
    ap.add_argument("--n-events",type=int,required=True);ap.add_argument("--n-markers",type=int,required=True);ap.add_argument("--n-populations",type=int,required=True);ap.add_argument("--expected-evaluable",type=int,required=True)
    ap.add_argument("--workers",type=int,default=2);ap.add_argument("--timeout-seconds",type=int,default=1200);a=ap.parse_args()
    parent=a.parent.resolve();repo=a.source_repo.resolve();jar=a.release_jar.resolve();java=a.java.resolve();data=a.data.resolve();protocol=a.protocol.resolve();out=a.output.resolve();out.mkdir(parents=True,exist_ok=False);runs=out/"runs";runs.mkdir()
    helper=Path(__file__).with_name("run_xshift_nilsson_fixedK20_30repeat.py");evaluator=Path(__file__).with_name("evaluate_xshift_cross_dataset.py")
    pm=json.loads((parent/"run_manifest.json").read_text(encoding="utf-8"));input_fcs=parent/a.input_fcs_name;config=parent/"importConfig.txt";matrix,names=read_fcs(input_fcs)
    if pm["experiment_id"]!=a.parent_id or not pm["all_checks_passed"] or matrix.shape!=(a.n_events,a.n_markers):raise ValueError("parent/input contract")
    parent_outputs=list((parent/"out").glob("*.fcs"))
    if len(parent_outputs)!=1:raise ValueError("expected one parent output FCS")
    parent_out,parent_out_names=read_fcs(parent_outputs[0]);parent_labels=np.rint(parent_out[:,-1]).astype(np.int32)
    if parent_out.shape!=(a.n_events,a.n_markers+1) or parent_out_names[:-1]!=names or not np.array_equal(parent_out[:,:-1],matrix):raise ValueError("parent output contract")
    parent_jar=Path(json.loads((parent/"xshift_execution.json").read_text(encoding="utf-8"))["command"][3])
    if sha256(parent_jar)!=sha256(jar):raise ValueError("jar mismatch")
    commit=subprocess.run(["git","-C",str(repo),"rev-parse","29-Jun-2017^{}"],capture_output=True,text=True,check=True).stdout.strip();source=subprocess.run(["git","-C",str(repo),"show","29-Jun-2017^{}:src/vortex/clustering/XShiftClustering.java"],capture_output=True,text=True,check=True).stdout
    with zipfile.ZipFile(jar) as z:cls=z.read("util/Shuffle.class")
    refs=class_methodrefs(cls);contract={"tag_commit":commit,"source_sha256":hashlib.sha256(source.encode()).hexdigest(),"source_has_shuffle_initialization":"new Shuffle<Datapoint>()).shuffleCopyArray" in source,"source_has_math_random_fallback":"Math.random() * (numCells)" in source,"shuffle_class_sha256":hashlib.sha256(cls).hexdigest(),"java_util_random_methodrefs":sorted({r for r in refs if r[0]=="java/util/Random"}),"default_random_constructor_present":('java/util/Random','<init>','()V') in refs,"seeded_random_constructor_present":('java/util/Random','<init>','(J)V') in refs,"cli_seed_interface":False};(out/"randomness_source_contract.json").write_text(json.dumps(contract,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    if commit!="fda75cf79980222da663185e2a6a72b442b9aff3" or not contract["source_has_shuffle_initialization"] or not contract["source_has_math_random_fallback"] or not contract["default_random_constructor_present"] or contract["seeded_random_constructor_present"]:raise ValueError("random contract")
    truth_raw=pd.read_csv(data,sep="\t" if a.separator=="tab" else ",",usecols=["label"])["label"]
    if a.label_mode=="numeric_1_to_n":
        numeric=pd.to_numeric(truth_raw,errors="coerce");ev=(numeric.between(1,a.n_populations)&(numeric%1==0)).to_numpy();truth=np.array([str(int(x)) for x in numeric[ev]])
    else:
        all_truth=truth_raw.astype(str).to_numpy();ev=all_truth!="unassigned";truth=all_truth[ev]
    if len(truth_raw)!=a.n_events or int(ev.sum())!=a.expected_evaluable or np.unique(truth).size!=a.n_populations:raise ValueError("truth contract")
    cmd=[str(java),"-Xmx4G","-cp",str(jar),"standalone.Xshift","20"];inputs={"input_fcs":sha256(input_fcs),"config":sha256(config),"release_jar":sha256(jar),"java":sha256(java),"data":sha256(data),"helper":sha256(helper),"evaluator":sha256(evaluator)};(out/"input_hashes.json").write_text(json.dumps(inputs,indent=2)+"\n",encoding="utf-8")
    def one(rep:int)->dict:
        d=runs/f"repeat{rep:03d}";d.mkdir();(d/"importConfig.txt").write_bytes(config.read_bytes());(d/"fcsFileList.txt").write_text(str(input_fcs)+"\n",encoding="utf-8");start=time.perf_counter()
        try:r=subprocess.run(cmd,cwd=d,capture_output=True,text=True,timeout=a.timeout_seconds);timed=False
        except subprocess.TimeoutExpired as e:r=None;timed=True;(d/"xshift_stdout.log").write_text(e.stdout or "",encoding="utf-8");(d/"xshift_stderr.log").write_text(e.stderr or "",encoding="utf-8")
        runtime=time.perf_counter()-start
        if r is not None:(d/"xshift_stdout.log").write_text(r.stdout,encoding="utf-8");(d/"xshift_stderr.log").write_text(r.stderr,encoding="utf-8");code=r.returncode
        else:code=None
        cand=list((d/"out").glob("*.fcs")) if (d/"out").is_dir() else [];ok=False;detail="";nc=None;lh=None
        if not timed and code==0 and len(cand)==1:
            m,n=read_fcs(cand[0]);v=m[:,-1];ok=m.shape==(a.n_events,a.n_markers+1) and n[:-1]==names and n[-1].lower().replace("_","")=="clusterid" and np.all(np.isfinite(v)) and np.allclose(v,np.rint(v)) and np.array_equal(m[:,:-1],matrix)
            if ok:
                lab=np.rint(v).astype(np.int32);np.save(d/"cluster_ids_all_events.npy",lab);lh=sha256(d/"cluster_ids_all_events.npy");u,c=np.unique(lab,return_counts=True);nc=len(u);pd.DataFrame({"cluster_id":u,"events":c}).to_csv(d/"cluster_sizes.csv",index=False)
            detail=f"shape={m.shape}; marker_exact={np.array_equal(m[:,:-1],matrix)}"
        rec={"repeat":rep,"command":cmd,"cwd":str(d),"exit_code":code,"timed_out":timed,"runtime_seconds":runtime,"output_fcs_count":len(cand),"all_checks_passed":ok,"detail":detail,"n_clusters":nc,"labels_sha256":lh};(d/"run_manifest.json").write_text(json.dumps(rec,ensure_ascii=False,indent=2)+"\n",encoding="utf-8");return rec
    records=[]
    with ThreadPoolExecutor(max_workers=a.workers) as pool:
        fs={pool.submit(one,r):r for r in range(30)}
        for f in as_completed(fs):rec=f.result();records.append(rec);print(f"repeat={rec['repeat']:03d} pass={rec['all_checks_passed']} runtime={rec['runtime_seconds']:.2f}s",flush=True)
    records.sort(key=lambda x:x["repeat"]);pd.DataFrame(records).to_csv(out/"execution_index.csv",index=False)
    rows=[];pops=[];parts=[];parent_comp=[]
    for rec in records:
        if not rec["all_checks_passed"]:continue
        rep=rec["repeat"];lab=np.load(runs/f"repeat{rep:03d}"/"cluster_ids_all_events.npy",allow_pickle=False);parts.append(lab);met,pf,_,_,_=evaluate_multiclass(truth,lab[ev],lab);rows.append({"repeat":rep,"runtime_seconds":rec["runtime_seconds"],"partition_sha256":hashlib.sha256(lab.tobytes()).hexdigest(),**met});pf.insert(0,"repeat",rep);pops.append(pf);parent_comp.append({"repeat":rep,"parent_partition_ari":adjusted_rand_score(parent_labels,lab),"parent_exact_label_fraction":float(np.mean(parent_labels==lab))})
    rf=pd.DataFrame(rows);rf.to_csv(out/"run_level_metrics.csv",index=False);pd.concat(pops,ignore_index=True).to_csv(out/"population_level_metrics.csv",index=False);pd.DataFrame(parent_comp).to_csv(out/"parent_partition_comparison.csv",index=False)
    pairs=[]
    for i in range(len(parts)):
        for j in range(i+1,len(parts)):pairs.append({"repeat_a":int(rf.iloc[i]["repeat"]),"repeat_b":int(rf.iloc[j]["repeat"]),"partition_ari":adjusted_rand_score(parts[i],parts[j])})
    pair=pd.DataFrame(pairs);pair.to_csv(out/"pairwise_partition_ari.csv",index=False)
    summary=[];metric_names=("runtime_seconds","n_predicted_clusters_all_events","ari","macro_precision","macro_recall","macro_f1","weighted_f1","hungarian_accuracy","weighted_evaluable_cluster_purity")
    for i,k in enumerate(metric_names):v=rf[k].to_numpy(float);lo,hi=bootstrap_ci(v,20260911+i);summary.append({"metric":k,"n":len(v),"mean":v.mean(),"sd":v.std(ddof=1),"median":np.median(v),"min":v.min(),"max":v.max(),"bootstrap_mean_ci_low":lo,"bootstrap_mean_ci_high":hi})
    pd.DataFrame(summary).to_csv(out/"endpoint_descriptive_statistics.csv",index=False)
    success=sum(bool(r["all_checks_passed"]) for r in records);unique=int(rf.partition_sha256.nunique()) if len(rf) else 0;checks=[]
    def ck(n,p,d):checks.append({"check":n,"passed":bool(p),"detail":d})
    ck("parent_manifest",pm["experiment_id"]==a.parent_id and pm["all_checks_passed"],pm["experiment_id"]);ck("randomness_source",contract["default_random_constructor_present"] and not contract["seeded_random_constructor_present"],json.dumps(contract));ck("execution_accounting",len(records)==30 and {r["repeat"] for r in records}==set(range(30)),f"rows={len(records)}");ck("all_runs_successful",success==30,f"success={success}");ck("run_metrics",len(rf)==success and rf.repeat.nunique()==success,f"rows={len(rf)}");ck("population_metrics",sum(len(x) for x in pops)==success*a.n_populations,f"rows={sum(len(x) for x in pops)}");ck("pairwise_count",len(pair)==success*(success-1)//2,f"rows={len(pair)}");ck("input_hash_uniformity",all(sha256(runs/f"repeat{r:03d}"/"importConfig.txt")==inputs["config"] for r in range(30)),"30/30");ck("labels_valid",len(rf)==30 and rf.n_predicted_clusters_all_events.ge(1).all(),"valid");ck("finite_primary_metrics",np.isfinite(rf[["ari","macro_precision","macro_recall","macro_f1","weighted_f1","hungarian_accuracy"]].to_numpy()).all(),"finite");cf=pd.DataFrame(checks);cf.to_csv(out/"checks.csv",index=False);pv=pair.partition_ari.to_numpy(float)
    manifest={"experiment_id":a.experiment_id,"completed_utc":datetime.now(timezone.utc).isoformat(),"dataset":a.dataset,"protocol":str(protocol),"protocol_sha256":sha256(protocol),"script":str(Path(__file__).resolve()),"script_sha256":sha256(Path(__file__).resolve()),"helper_script":str(helper),"helper_script_sha256":sha256(helper),"evaluator_script":str(evaluator),"evaluator_script_sha256":sha256(evaluator),"parent":str(parent),"parent_manifest_sha256":sha256(parent/"run_manifest.json"),"parent_output_fcs":str(parent_outputs[0]),"parent_output_fcs_sha256":sha256(parent_outputs[0]),"source_repo":str(repo),"tag_commit":commit,"release_jar":str(jar),"release_jar_sha256":sha256(jar),"java":str(java),"java_sha256":sha256(java),"data":str(data),"data_sha256":sha256(data),"n_events":a.n_events,"n_markers":a.n_markers,"n_evaluable":a.expected_evaluable,"n_populations":a.n_populations,"workers":a.workers,"repeat_not_seed":True,"runs_attempted":30,"runs_successful":success,"runs_failed":30-success,"unique_partition_hashes":unique,"pairwise_partition_ari_mean":float(pv.mean()),"pairwise_partition_ari_min":float(pv.min()),"pairwise_partition_ari_max":float(pv.max()),"checks_passed":int(cf.passed.sum()),"checks_total":len(cf),"all_checks_passed":bool(cf.passed.all()),"python":sys.version,"python_executable":sys.executable,"platform":platform.platform()};(out/"run_manifest.json").write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+"\n",encoding="utf-8");(out/"scientific_summary.md").write_text(f"# {a.experiment_id} X-shift {a.dataset}固定K=20三十次原生重复\n\n成功{success}/30；唯一全事件分区哈希{unique}；435个两两ARI均值/范围={pv.mean():.6f}/[{pv.min():.6f},{pv.max():.6f}]；Macro F1均值/范围={rf.macro_f1.mean():.6f}/[{rf.macro_f1.min():.6f},{rf.macro_f1.max():.6f}]。运行编号为native repeat而非seed。\n",encoding="utf-8");arts=[p for p in out.rglob("*") if p.is_file()];(out/"artifact_hashes.json").write_text(json.dumps({str(p.relative_to(out)):sha256(p) for p in arts},ensure_ascii=False,indent=2)+"\n",encoding="utf-8");print(json.dumps(manifest,ensure_ascii=False));return 0 if manifest["all_checks_passed"] else 1

if __name__=="__main__":raise SystemExit(main())
