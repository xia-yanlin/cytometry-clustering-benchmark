"""Post hoc metric check of frozen contingency tables, 20 September 2026.
No clustering is re-run. Primary overlap-count Hungarian matching is retained.
Usage: python matching_sensitivity.py --source /path/to/extracted/source --out ./out
Dependencies: numpy, pandas, scipy. The input archive contains all required tables.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment


def measures(a: np.ndarray, objective: str) -> dict:
    nref, ncl = a.shape
    rs, cs = a.sum(1), a.sum(0)
    f = np.divide(2*a, rs[:,None]+cs[None,:], out=np.zeros_like(a,dtype=float), where=(rs[:,None]+cs[None,:])!=0)
    ri, ci = linear_sum_assignment(-(a if objective=='overlap' else f))
    p=np.zeros(nref); r=np.zeros(nref); fs=np.zeros(nref)
    p[ri]=np.divide(a[ri,ci],cs[ci],out=np.zeros(len(ri)),where=cs[ci]!=0)
    r[ri]=np.divide(a[ri,ci],rs[ri],out=np.zeros(len(ri)),where=rs[ri]!=0)
    fs[ri]=f[ri,ci]
    return {'macro_precision':p.mean(),'macro_recall':r.mean(),'macro_f1':fs.mean(),
            'matched_event_fraction':a[ri,ci].sum()/a.sum(), 'assignment':list(zip(ri.tolist(),ci.tolist()))}


def main():
    pa=argparse.ArgumentParser();pa.add_argument('--source',type=Path,required=True);pa.add_argument('--out',type=Path,required=True)
    ar=pa.parse_args();ar.out.mkdir(parents=True,exist_ok=True)
    specs=[('EXP-045A_', 'PhenoGraph_fixed_graph','Samusik_01'),('EXP-045B_','PhenoGraph_fixed_graph','Levine_13dim'),('EXP-045C_','PhenoGraph_fixed_graph','Levine_32dim'),('EXP-046A-R1_','R_FlowSOM_full','Samusik_01'),('EXP-046B-R1_','R_FlowSOM_full','Levine_32dim'),('EXP-015-R2_','Python_FlowSOM_inclusion','Levine_13dim'),('EXP-011C_','Xshift_K20','Levine_32dim')]
    records=[]
    for pref,method,dataset in specs:
        dirs=list((ar.source/'experiment_records/runs').glob(pref+'*'))
        if len(dirs)!=1: raise ValueError(f'Expected one directory for {pref}, found {dirs}')
        paths=sorted(dirs[0].rglob('contingency_true_by_predicted.csv'))
        for p in paths:
            if pref=='EXP-015-R2_' and p.parent.name not in ['labeled_direct_a','all_direct']: continue
            a=pd.read_csv(p,index_col=0).to_numpy(dtype=float)
            if not np.isfinite(a).all() or (a<0).any() or not np.equal(a,np.floor(a)).all(): raise ValueError(str(p))
            x=measures(a,'overlap'); y=measures(a,'f1')
            if y['macro_f1']+1e-12 <x['macro_f1']:raise AssertionError('F1 optimum is lower than primary')
            rec={'method':method,'dataset':dataset,'run':p.parent.name,'n_evaluable':int(a.sum()),'n_reference':a.shape[0],'n_predicted_evaluable':a.shape[1], 'source':p.relative_to(ar.source).as_posix(),}
            for k in ['macro_precision','macro_recall','macro_f1','matched_event_fraction']:
                rec['overlap_'+k]=x[k];rec['f1optimal_'+k]=y[k];rec['delta_'+k]=y[k]-x[k]
            rec['assignment_changed']=x['assignment']!=y['assignment']
            records.append(rec)
    df=pd.DataFrame(records);df.to_csv(ar.out/'matching_run_results.csv',index=False)
    summary=df.groupby(['method','dataset'],sort=False).agg(n=('run','size'),overlap_f1_mean=('overlap_macro_f1','mean'),f1optimal_f1_mean=('f1optimal_macro_f1','mean'),delta_f1_mean=('delta_macro_f1','mean'),delta_f1_min=('delta_macro_f1','min'),delta_f1_max=('delta_macro_f1','max'),assignments_changed=('assignment_changed','sum')).reset_index()
    summary.to_csv(ar.out/'matching_summary.csv',index=False)
    # Confirm primary values against the frozen data plotted in Figure 3.
    checks=[]
    for f, method in [('figure3a_flowsom_r_run_points.csv','R_FlowSOM_full'),('figure3b_phenograph_run_points.csv','PhenoGraph_fixed_graph')]:
        ref=pd.read_csv(ar.source/'package/figure_data'/f)
        for ds, g in df[df.method==method].groupby('dataset'):
            vals=ref[ref.dataset==ds].macro_f1.to_numpy()
            diff=float(np.max(np.abs(np.sort(vals)-np.sort(g.overlap_macro_f1.to_numpy()))))
            checks.append({'check':f'{method} {ds} primary macro-F1 reconstruction','max_abs_difference':diff,'pass':diff<1e-12})
    met=pd.read_csv(ar.source/'package/inputs/flowsom_auto_multiclass_runs.csv')
    met=met[(met.selector_seed==12345)&met.regime.isin(['official_auto_max40','official_fixed_ktrue','official_fixed_k40'])].copy()
    if len(met)!=180: print('Meta filters found',len(met),'rows; regimes:',pd.read_csv(ar.source/'package/inputs/flowsom_auto_multiclass_runs.csv').regime.unique())
    met['matching_ceiling']=np.minimum(met.n_predicted_clusters_evaluable,met.n_true_populations)/met.n_true_populations
    met['ceiling_satisfied']=met.macro_f1<=met.matching_ceiling+1e-12
    met.to_csv(ar.out/'metacluster_matching_ceiling.csv',index=False)
    pd.DataFrame(checks).to_csv(ar.out/'reconstruction_checks.csv',index=False)
    if not all(c['pass'] for c in checks):raise AssertionError('Primary reconstruction failed')
    if not met.ceiling_satisfied.all():raise AssertionError('Matching ceiling check failed')
    print(summary.to_string(index=False))
    print("Computed", len(df), "contingency-table comparisons and", len(met), "metaclustering ceilings")
if __name__=='__main__':main()
