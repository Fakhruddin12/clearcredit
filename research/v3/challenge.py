"""Post-review, retrospective ClearCredit comparisons; not fresh validation.

Default path reproduces v2 if its generated fitted artifacts are absent.
See protocol.json and README.md for the frozen experimental specification.
"""
from __future__ import annotations
import argparse, hashlib, json, os, platform, subprocess, sys, time
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import scipy, sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score, brier_score_loss, log_loss
from sklearn.model_selection import train_test_split, StratifiedGroupKFold
from lightgbm import LGBMClassifier
import lightgbm
import torch
from torch import nn

V2=Path(__file__).resolve().parents[1]/'v2'
sys.path.insert(0,str(V2))
import experiment as old

PRIMARY_SEEDS=[42,137,2026]
SENSITIVITY_SEEDS=[718,9001]
SOURCE_SHA='30c6be3abd8dcfd3e6096c828bad8c2f011238620f5369220bd60cfc82700933'
PROTOCOL={
 'study':'Post-review challenge dated 31 October 2026, designed after inspecting v2 reporting results.',
 'status':'Exploratory retrospective follow-up. Same reused cohort; not confirmatory or prospective evidence.',
 'data_and_splits':'Exact v2 workbook and disjoint financial-input-group partitions; no reporting-driven retuning.',
 'primary_objective':'Defaults captured at exactly 20% account-review capacity; AUC, AP, Brier and log loss secondary.',
 'simple_rules':'Binary latest status>=1; nonnegative latest status severity. Seed20261031 random tie-break fixed without labels. Also show tie expectations and 200 alternate tie draws.',
 'latest_models':'Credit limit plus latest status, balance and payment and two limit ratios; logistic and LightGBM search the same three v2 development presets. Single-status logistic uses only PAY_0 categories, C=1.',
 'history_controls':'Original three-seed GRU; same 17-channel monthly inputs flattened into an MLP(32,32); GRU with jointly permuted monthly vectors within each account, seed7331; GRU with all status channels removed.',
 'neural_budget':'Original selected width24, dropout0.2, 26 fixed epochs for GRU controls; MLP widths32,32 dropout0.2, 26 epochs. AdamW lr0.001 wd0.001, batch256, clip5. No new neural architecture search.',
 'seeds':PRIMARY_SEEDS+SENSITIVITY_SEEDS,
 'primary_ensemble':'First three original seeds. Additional two seeds are single-fit sensitivity only and do not change ensemble membership.',
 'order_ablation':'One fixed independent within-account permutation, applied to all four numeric and thirteen status channels jointly. Fit/calibration/reporting use the permuted representation; this is not inference-only corruption.',
 'subgroup':'PAY_0<1 means no latest code>=1, not absence of prior distress. Report within-group AUC/AP and exact20% subgroup queue separately from global-policy capture.',
 'fixed_policy':'For new common comparison, threshold is 80th percentile of calibration scores. This differs from the v2 tuning-score policy; legacy85.2% is not carried into this policy.',
 'uncertainty':'500 paired bootstrap draws of full financial-input groups, conditional on fitted models. No multiplicity adjustment or retraining uncertainty. Neural single-fit ranges are separate.',
 'practical_decision':'GRU complexity requires at least2 percentage points of capture advantage over full-history LightGBM, positive paired lower95% bound, and no more than2x median inference time. This review-stage criterion is not prospectively validated.',
 'no_new_cohort':'No new untouched outcomes are available. A later multi-vintage dataset remains required; arbitrary ID or feature-month splitting is not a valid replacement.',
 'stop':'Run this fixed comparison once, retain negative findings, do not promote a reporting winner as a newly development-selected champion.'
}

def latest_features(df):
 x=df[['LIMIT_BAL','PAY_0','BILL_AMT1','PAY_AMT1']].copy()
 x['UTIL_1']=x.BILL_AMT1/x.LIMIT_BAL.clip(lower=1)
 x['PAY_LIMIT_1']=x.PAY_AMT1/x.LIMIT_BAL.clip(lower=1)
 return x

def tabular_prep(x):
 from sklearn.compose import ColumnTransformer
 from sklearn.impute import SimpleImputer
 from sklearn.pipeline import make_pipeline
 from sklearn.preprocessing import FunctionTransformer, StandardScaler, OneHotEncoder
 status=[c for c in x if c in old.STATUS]
 num=[c for c in x if c not in status]
 parts=[]
 if num:parts.append(('num',make_pipeline(SimpleImputer(strategy='median'),FunctionTransformer(old.signedlog),StandardScaler()),num))
 if status:parts.append(('status',OneHotEncoder(categories=[[-99]+list(range(-2,10))]*len(status),handle_unknown='ignore',sparse_output=False),status))
 return ColumnTransformer(parts,sparse_threshold=0)

def permute_months(seq,seed=7331):
 rng=np.random.default_rng(seed)
 order=np.argsort(rng.random(seq.shape[:2]),axis=1)
 return np.take_along_axis(seq,order[:,:,None],axis=1)

def representation(monthly,kind):
 seq,static=monthly
 if kind=='flat_mlp':return np.concatenate([seq.reshape(len(seq),-1),static],axis=1).astype('float32')
 if kind=='no_status_gru':return seq[:,:,:4].copy(),static
 if kind=='shuffled_gru':return permute_months(seq),static
 return seq,static

def subset(arr,ix):
 return tuple(a[ix] for a in arr) if isinstance(arr,tuple) else arr[ix]

class FlatMLP(nn.Module):
 def __init__(self,d):
  super().__init__();self.layers=nn.Sequential(nn.Linear(d,32),nn.ReLU(),nn.Dropout(.2),nn.Linear(32,32),nn.ReLU(),nn.Dropout(.2),nn.Linear(32,1))
 def forward(self,x):return self.layers(x).squeeze(-1)

def fit_fixed(kind,arr,y,seed):
 torch.manual_seed(seed);np.random.seed(seed)
 model=FlatMLP(arr.shape[1]) if kind=='flat_mlp' else old.MonthlyGRU(d=arr[0].shape[2],width=24,dropout=.2)
 optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.001)
 xx=old.tensors(arr);yy=torch.as_tensor(np.asarray(y),dtype=torch.float32)
 rng=np.random.default_rng(seed);losses=[]
 for epoch in range(26):
  model.train();order=rng.permutation(len(yy));total=0.
  for start in range(0,len(yy),256):
   ix=order[start:start+256];optimizer.zero_grad(set_to_none=True)
   loss=nn.functional.binary_cross_entropy_with_logits(model(*(a[ix] for a in xx)),yy[ix]);loss.backward()
   nn.utils.clip_grad_norm_(model.parameters(),5.);optimizer.step();total+=float(loss.detach())*len(ix)
  losses.append(total/len(yy))
 return model,losses

def queue(y,score,tie,fraction=.2):
 y=np.asarray(y);score=np.asarray(score);k=max(1,int(np.ceil(len(y)*fraction)))
 ix=np.lexsort((np.asarray(tie),-score))[:k];captured=int(y[ix].sum())
 return {'reviewed':k,'defaults_captured':captured,'capture_rate':captured/int(y.sum()),'precision':captured/k}

def tie_expectation(y,score,fraction=.2):
 y=np.asarray(y);score=np.asarray(score);k=max(1,int(np.ceil(len(y)*fraction)))
 cutoff=np.sort(score)[-k];above=score>cutoff;tied=score==cutoff;remaining=k-int(above.sum())
 return float(y[above].sum()+remaining*y[tied].mean())

def measures(y,p,tie,probability=True):
 y=np.asarray(y);p=np.asarray(p)
 out={'auc':float(roc_auc_score(y,p)),'ap':float(average_precision_score(y,p)),**queue(y,p,tie),'n':len(y),'defaults':int(y.sum())}
 if probability:out.update(brier=float(brier_score_loss(y,p)),log_loss=float(log_loss(y,p)),mean_prediction=float(p.mean()))
 return out

def bootstrap(y,pred,groups,tie,draws=500):
 rng=np.random.default_rng(884);unique=np.unique(groups);lookup={g:np.flatnonzero(groups==g) for g in unique}
 vals={k:{m:[] for m in ['auc','ap','capture_rate']} for k in pred}
 for b in range(draws):
  ix=np.concatenate([lookup[g] for g in rng.choice(unique,len(unique),replace=True)])
  if len(np.unique(y[ix]))<2:continue
  for name,p in pred.items():
   vals[name]['auc'].append(roc_auc_score(y[ix],p[ix]));vals[name]['ap'].append(average_precision_score(y[ix],p[ix]))
   vals[name]['capture_rate'].append(queue(y[ix],p[ix],tie[ix])['capture_rate'])
  if (b+1)%100==0:print('BOOTSTRAP',b+1,flush=True)
 comparisons=[('ordered_gru','lightgbm'),('ordered_gru','shuffled_gru'),('ordered_gru','flat_mlp'),('ordered_gru','no_status_gru'),('lightgbm','latest_lightgbm'),('lightgbm','latest_status_logistic'),('lightgbm','severity_rule'),('lightgbm','binary_rule')]
 ci={k:{m:np.quantile(v,[.025,.975]).tolist() for m,v in mm.items()} for k,mm in vals.items()}
 delta={a+' minus '+b:{m:{'mean':float(np.mean(np.asarray(vals[a][m])-vals[b][m])),'ci95':np.quantile(np.asarray(vals[a][m])-vals[b][m],[.025,.975]).tolist()} for m in vals[a]} for a,b in comparisons}
 return ci,delta

def load_data(path):
 assert hashlib.sha256(path.read_bytes()).hexdigest()==SOURCE_SHA,'Workbook changed'
 df=pd.read_excel(path,header=1).rename(columns={'default payment next month':'TARGET'});y=df.TARGET.astype(int)
 groups=pd.util.hash_pandas_object(df[old.FINANCIAL],index=False).to_numpy()
 dev,test=train_test_split(np.arange(len(df)),test_size=.2,stratify=y,random_state=42)
 dev=dev[~np.isin(groups[dev],groups[test])]
 folds=list(StratifiedGroupKFold(5,shuffle=True,random_state=2026).split(df.iloc[dev],y.iloc[dev],groups[dev]))
 tune=dev[folds[0][1]];cal=dev[folds[1][1]];fit=np.setdiff1d(dev,np.r_[tune,cal]);refit=np.r_[fit,tune]
 parts={'fit':fit,'tune':tune,'calibration':cal,'reporting':test}
 for a,ia in parts.items():
  for b,ib in parts.items():
   if a!=b:assert not set(groups[ia])&set(groups[ib])
 manifest=json.loads((V2/'results/data_manifest.json').read_text())
 for name,ix in parts.items():assert hashlib.sha256(ix.tobytes()).hexdigest()==manifest['partitions'][name]['indices_sha256']
 return df,y,groups,parts,refit

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--data',type=Path,default=V2/'data/source.xls');parser.add_argument('--baseline-results',type=Path,default=V2/'results_reproduced');parser.add_argument('--out',type=Path,default=Path('results_reproduced'));args=parser.parse_args()
 out=args.out;out.mkdir(parents=True,exist_ok=True);(out/'models').mkdir(exist_ok=True)
 old.dump(out/'protocol.json',PROTOCOL) # Freeze before fitting any new candidate.
 if not (args.baseline_results/'models/gru_42.pt').exists():
  subprocess.run([sys.executable,str(V2/'experiment.py'),'--data',str(args.data),'--out',str(args.baseline_results)],check=True)
 start=time.perf_counter();torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
 df,y,groups,parts,refit=load_data(args.data);fit,tune,cal,test=(parts[k] for k in ['fit','tune','calibration','reporting'])
 base=args.baseline_results;original=json.loads((base/'results.json').read_text());selected=json.loads((base/'selection_locked_before_reporting.json').read_text())
 assert selected['configs']['gru']=={'width':24,'dropout':.2} and selected['median_epochs']['gru']==26, 'Baseline GRU selection changed; review the frozen control budget before proceeding.'
 saved=np.load(base/'split_indices.npz')
 for k,v in parts.items():assert np.array_equal(saved[k],v)
 preprocess=joblib.load(base/'models/preprocessing.joblib');calibrators=joblib.load(base/'models/calibrators.joblib')
 monthly=preprocess['monthly'].transform(df);full=old.features(df)
 raw={};prob={};predictors={};fit_seconds={};seed_metrics={};seed_predictions={};parameter_counts={};tuning=[]
 for name in ['logistic','lightgbm']:
  model=joblib.load(base/f'models/{name}.joblib')
  def predict(frame,m=model,name=name):
   xx=old.features(frame)
   if name=='logistic':xx=preprocess['flat'].transform(xx).astype('float32')
   return m.predict_proba(xx)[:,1]
  raw[name]=predict(df);prob[name]=old.calibrate(calibrators[name],raw[name]);predictors[name]=predict
  for metric,expected in original['models'][name]['calibrated'].items():
   if metric=='roc_auc':assert abs(roc_auc_score(y.iloc[test],prob[name][test])-expected)<1e-10
 ordered_models=[];ordered_pp=[]
 for seed in PRIMARY_SEEDS:
  model=old.MonthlyGRU(**selected['configs']['gru']);model.load_state_dict(torch.load(base/f'models/gru_{seed}.pt',weights_only=True,map_location='cpu'))
  ordered_models.append(model);ordered_pp.append(old.neural_predict(model,monthly))
 raw['ordered_gru']=np.mean(ordered_pp,axis=0);prob['ordered_gru']=old.calibrate(calibrators['gru'],raw['ordered_gru'])
 assert np.allclose(prob['ordered_gru'][test],pd.read_csv(base/'reporting_predictions.csv').gru,atol=1e-7)
 print('BASELINE_RESTORED',len(test),'accounts',flush=True)

 for name in ['latest_status_logistic','latest_logistic','latest_lightgbm']:
  x=df[['PAY_0']].copy() if name=='latest_status_logistic' else latest_features(df)
  prep=tabular_prep(x).fit(x.iloc[fit]);z=prep.transform(x).astype('float32')
  configs=([{'C':1.}] if name=='latest_status_logistic' else [{'C':v} for v in [.01,.1,1.]] if name=='latest_logistic' else [
   {'num_leaves':15,'min_child_samples':100,'reg_lambda':10.}, {'num_leaves':31,'min_child_samples':100,'reg_lambda':10.}, {'num_leaves':15,'min_child_samples':200,'reg_lambda':20.}])
  def make(cfg):return LGBMClassifier(**cfg,n_estimators=450,learning_rate=.03,colsample_bytree=.9,random_state=42,n_jobs=2,verbosity=-1) if name=='latest_lightgbm' else LogisticRegression(**cfg,max_iter=1500,solver='lbfgs')
  best=-1;chosen=None
  for cfg in configs:
   xx=x if name=='latest_lightgbm' else z;model=make(cfg);model.fit(xx.iloc[fit] if isinstance(xx,pd.DataFrame) else xx[fit],y.iloc[fit]);p=model.predict_proba(xx.iloc[tune] if isinstance(xx,pd.DataFrame) else xx[tune])[:,1]
   ap=average_precision_score(y.iloc[tune],p);tuning.append({'name':name,'config':cfg,'development_ap':float(ap)})
   if ap>best:best=ap;chosen=cfg
  prep=tabular_prep(x).fit(x.iloc[refit]);xx=x if name=='latest_lightgbm' else prep.transform(x).astype('float32')
  begin=time.perf_counter();model=make(chosen).fit(xx.iloc[refit] if isinstance(xx,pd.DataFrame) else xx[refit],y.iloc[refit]);fit_seconds[name]=time.perf_counter()-begin
  pp=model.predict_proba(xx)[:,1];cc=old.calibration(y.iloc[cal],pp[cal]);raw[name]=pp;prob[name]=old.calibrate(cc,pp)
  joblib.dump({'model':model,'preprocessing':prep,'calibrator':cc,'config':chosen},out/f'models/{name}.joblib')
  def predict(frame,name=name,m=model,prep=prep):
   xf=frame[['PAY_0']] if name=='latest_status_logistic' else latest_features(frame)
   return m.predict_proba(xf if name=='latest_lightgbm' else prep.transform(xf).astype('float32'))[:,1]
  predictors[name]=predict
  print('LATEST_FIT',name,chosen,flush=True)

 seed_metric_fields=['auc','ap','capture_rate'];tie=np.random.default_rng(20261031).random(len(df));yy=y.iloc[test].to_numpy()
 for kind in ['ordered_gru','flat_mlp','shuffled_gru','no_status_gru']:
  arr=representation(monthly,kind);models=[];predictions=[];seed_metrics[kind]=[];losses={};begin=time.perf_counter()
  for i,seed in enumerate(PRIMARY_SEEDS+SENSITIVITY_SEEDS):
   if kind=='ordered_gru' and i<3:model=ordered_models[i];pp=ordered_pp[i];hist=None
   else:model,hist=fit_fixed(kind,subset(arr,refit),y.iloc[refit],seed);pp=old.neural_predict(model,arr)
   models.append(model);predictions.append(pp);losses[str(seed)]=hist
   cc=old.calibration(y.iloc[cal],pp[cal]);pp_cal=old.calibrate(cc,pp[test]);seed_metrics[kind].append({'seed':seed,'predictions':pp_cal})
   torch.save(model.state_dict(),out/f'models/{kind}_{seed}.pt');print('NEURAL_FIT',kind,seed,flush=True)
  fit_seconds[kind]=time.perf_counter()-begin
  primary=models[:3];raw[kind]=np.mean(predictions[:3],axis=0);cc=old.calibration(y.iloc[cal],raw[kind][cal]);prob[kind]=old.calibrate(cc,raw[kind])
  joblib.dump(cc,out/f'models/{kind}_calibrator.joblib');seed_predictions[kind]=predictions
  parameter_counts[kind]=sum(p.numel() for p in models[0].parameters())
  def predict(frame,kind=kind,models=primary):
   a=representation(preprocess['monthly'].transform(frame),kind)
   return np.mean([old.neural_predict(m,a) for m in models],axis=0)
  # The shuffled representation must use full-dataset row permutations. Its
  # predictor is excluded from operational latency claims on arbitrary subsets.
  if kind!='shuffled_gru':predictors[kind]=predict
  old.dump(out/f'{kind}_training_losses.json',losses)
  if kind=='ordered_gru':assert np.allclose(prob[kind][test],pd.read_csv(base/'reporting_predictions.csv').gru,atol=1e-7)
 old.dump(out/'development_search.json',tuning)
 old.dump(out/'selection.json',{'original_selection_unchanged':selected,'new_neural_configs':'frozen in protocol','new_latest_search':tuning,'no_reporting_winner_promotion':True})
 joblib.dump(preprocess,out/'models/preprocessing.joblib')
 print('FITTING_COMPLETE; START_REPORTING',flush=True)
 for kind,rows in seed_metrics.items():
  seed_metrics[kind]=[{'seed':row['seed'],**measures(yy,row['predictions'],tie[test])} for row in rows]
 scores={name:p[test] for name,p in prob.items()}
 scores['binary_rule']=df.iloc[test].PAY_0.ge(1).to_numpy(float)
 scores['severity_rule']=df.iloc[test].PAY_0.clip(lower=0).to_numpy(float)
 results={name:measures(yy,p,tie[test],name not in ['binary_rule','severity_rule']) for name,p in scores.items()}
 simple={}
 for name in ['binary_rule','severity_rule']:
  counts=[queue(yy,scores[name],np.random.default_rng(s).random(len(test)))['defaults_captured'] for s in range(1001,1201)]
  simple[name]={'expected_defaults_at20':tie_expectation(yy,scores[name]),'expected_capture_at20':tie_expectation(yy,scores[name])/int(yy.sum()),'tie_seed_range_defaults':[min(counts),max(counts)],'tie_draws':200,'note':'Tie sensitivity range, not a confidence interval.'}
 flag=df.iloc[test].PAY_0.ge(1).to_numpy();simple['unconstrained_binary']={'reviewed':int(flag.sum()),'review_fraction':float(flag.mean()),'defaults_captured':int(yy[flag].sum()),'capture_rate':float(yy[flag].sum()/yy.sum())}
 fixed={};subgroups={};mask=df.iloc[test].PAY_0.lt(1).to_numpy()
 for name,p in scores.items():
  if name not in ['binary_rule','severity_rule']:
   threshold=float(np.quantile(prob[name][cal],.8));fixed[name]=old.fixed_policy(yy,p,threshold)
   fixed[name]['no_latest_delay']=old.fixed_policy(yy[mask],p[mask],threshold)
  subgroups[name]=measures(yy[mask],p[mask],tie[test][mask],name not in ['binary_rule','severity_rule'])
 ci,delta=bootstrap(yy,scores,groups[test],tie[test]);sg_ci,sg_delta=bootstrap(yy[mask],{k:p[mask] for k,p in scores.items()},groups[test][mask],tie[test][mask])
 latency={}
 for name in ['latest_status_logistic','latest_lightgbm','lightgbm','ordered_gru','flat_mlp']:
  fn=predictors[name];fn(df.iloc[test]);tt=[]
  for _ in range(7):
   begin=time.perf_counter();fn(df.iloc[test]);tt.append(time.perf_counter()-begin)
  latency[name]={'median_ms_6000_accounts':1000*float(np.median(tt)),'repetitions':7,'preprocessing_included':True,'calibration_excluded_for_all':True}
 advantage=results['ordered_gru']['capture_rate']-results['lightgbm']['capture_rate']
 qualifies=(advantage>=.02 and delta['ordered_gru minus lightgbm']['capture_rate']['ci95'][0]>0 and latency['ordered_gru']['median_ms_6000_accounts']<=2*latency['lightgbm']['median_ms_6000_accounts'])
 report={'protocol':PROTOCOL,'metrics':results,'simple_rules':simple,'cluster_bootstrap_ci95':ci,'paired_differences':delta,'fixed_calibration_policy':fixed,'no_latest_delay':{'definition':'PAY_0<1; previous delays may exist','n':int(mask.sum()),'defaults':int(yy[mask].sum()),'accounts_with_earlier_delay':int(df.iloc[test].loc[mask,old.STATUS[1:]].ge(1).any(axis=1).sum()),'metrics':subgroups,'cluster_bootstrap_ci95':sg_ci,'paired_differences':sg_delta},'seed_metrics':seed_metrics,'parameter_counts':parameter_counts,'latency':latency,'fit_seconds':fit_seconds,'practical_decision':{'gru_qualifies_for_added_complexity':bool(qualifies),'gru_minus_lightgbm_capture':advantage,'criterion':PROTOCOL['practical_decision'],'note':'Descriptive research recommendation; independent validation still absent.'},'elapsed_seconds':time.perf_counter()-start}
 old.dump(out/'results.json',report)
 old.dump(out/'environment.json',{'python':platform.python_version(),'torch':torch.__version__,'numpy':np.__version__,'pandas':pd.__version__,'lightgbm':lightgbm.__version__,'scikit_learn':sklearn.__version__,'scipy':scipy.__version__,'threads':2,'source_sha256':SOURCE_SHA,'baseline_source_commit':'99f3fbb17c352f1bc2192e6647d8431cf00a7440'})
 pd.DataFrame({'row_index':test,'financial_group':groups[test],'y':yy,'tie_key':tie[test],**scores}).to_csv(out/'reporting_predictions.csv',index=False)
 old.dump(out/'data_manifest.json',{'source_sha256':SOURCE_SHA,'partitions':{k:{'n':len(v),'indices_sha256':hashlib.sha256(v.tobytes()).hexdigest()} for k,v in parts.items()},'reporting_defaults':int(yy.sum())})
 print('FINAL',json.dumps({'metrics':results,'decision':report['practical_decision'],'elapsed_seconds':report['elapsed_seconds']},indent=2),flush=True)

if __name__=='__main__':main()
