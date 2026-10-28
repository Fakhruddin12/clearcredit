"""ClearCredit v2: behavioural risk research, not a lending decision system.

Run: python experiment.py --data source.xls --out results
All model/hyperparameter choices use development labels only. The existing
6,000-row reporting set is reused and is NOT a pristine external test set.
"""
from __future__ import annotations
import argparse, copy, hashlib, json, platform, time, urllib.request, zipfile
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from scipy.special import expit, logit
from scipy.stats import norm
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, roc_auc_score, log_loss,
                             brier_score_loss, roc_curve)
from sklearn.model_selection import train_test_split, StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer
from lightgbm import LGBMClassifier
import lightgbm
import torch
from torch import nn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

STATUS = ['PAY_0','PAY_2','PAY_3','PAY_4','PAY_5','PAY_6']  # newest first
BILLS = [f'BILL_AMT{i}' for i in range(1,7)]
PAYMENTS = [f'PAY_AMT{i}' for i in range(1,7)]
FINANCIAL = ['LIMIT_BAL'] + STATUS + BILLS + PAYMENTS
SEEDS = [42,137,2026]
PROTOCOL = {
 'purpose':'Rank existing credit-card accounts for hypothetical risk review; predict source-defined next-month default.',
 'data':'UCI 350, Taiwan April-September 2005 account history; 30,000 rows.',
 'reporting_set':'Original stratified random 20% seed=42, already examined in v1; descriptive internal re-evaluation only.',
 'development':'Remove financial-feature duplicate groups shared with reporting set. StratifiedGroupKFold(5, shuffle=True, random_state=2026): fold0 tuning, fold1 calibration, folds2-4 fitting.',
 'selection':'Highest tuning average precision. Family winners selected without reporting labels. No reporting-set tuning.',
 'models':'Unweighted logistic regression; unweighted LightGBM; custom tabular residual network; monthly GRU with static branch; fixed 50:50 tree/strongest-neural blend.',
 'neural':'Two presets per family; max60 epochs, patience8 on tuning log loss; seeds42/137/2026 for selected preset. Custom implementations inspired by literature, not paper replications.',
 'refit':'Family winners refit on fitting+tuning rows; neural epochs fixed to median best epoch from development runs. Calibration rows excluded from all base-model fitting.',
 'calibration':'One sigmoid calibrator per family fit to its predictions on separate calibration rows; report before/after diagnostics.',
 'demographics':'ID, sex, age, education and marriage excluded from predictors. Sex and age retained only for descriptive audits. Exclusion does not prove fairness.',
 'policy':'Illustrative review capacity20%; tuning score quantile sets fixed threshold; also report exact top20% ranking on reporting set. No causal treatment or profit claim.',
 'uncertainty':'500 paired cluster-bootstrap draws over identical financial-feature groups; conditional on fitted models, not training or selection uncertainty.',
 'stress':'Label-prior reweighting to35% defaults; latest-month masking and +1 month delinquency perturbation are sensitivity scenarios, not causal or macroeconomic forecasts.',
 'stop_rule':'No model changes after first reporting evaluation. Negative deep-learning results retained.',
}

def dump(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, allow_nan=False, default=lambda v: v.item() if isinstance(v,np.generic) else str(v)))

def signedlog(x):
    x=np.asarray(x,dtype=float)
    return np.sign(x)*np.log1p(np.abs(x))

def features(frame):
    """Only features available at the common observation date, no target use."""
    x=frame[FINANCIAL].copy()
    limit=x.LIMIT_BAL.clip(lower=1)
    for i in range(1,7):
        x[f'UTIL_{i}']=x[f'BILL_AMT{i}']/limit
        x[f'PAY_LIMIT_{i}']=x[f'PAY_AMT{i}']/limit
    x['UTIL_CHANGE_1_6']=(x.BILL_AMT1-x.BILL_AMT6)/limit
    x['DELAY_COUNT']=x[STATUS].ge(1).sum(axis=1)
    x['SEVERE_DELAY_COUNT']=x[STATUS].ge(2).sum(axis=1)
    x['MAX_RECORDED_DELAY']=x[STATUS].clip(lower=0).max(axis=1)
    x['MEAN_PAY_LIMIT']=x[PAYMENTS].mean(axis=1)/limit
    return x

def prep(x):
    numerical=[c for c in x.columns if c not in STATUS]
    return ColumnTransformer([
      ('numeric',make_pipeline(SimpleImputer(strategy='median', add_indicator=False),
                              FunctionTransformer(signedlog),StandardScaler()), numerical),
      ('status',make_pipeline(SimpleImputer(strategy='constant',fill_value=-99),
                  OneHotEncoder(categories=[[-99]+list(range(-2,10))]*6,
                                handle_unknown='ignore',sparse_output=False)),STATUS)
    ],sparse_threshold=0)

class MonthlyPrep:
    def fit(self, frame):
        seq,stat=self.raw(frame)
        self.seq_imputer=SimpleImputer(strategy='median').fit(seq.reshape(-1,4))
        filled=self.seq_imputer.transform(seq.reshape(-1,4))
        self.seq_scaler=StandardScaler().fit(filled)
        self.stat_scaler=StandardScaler().fit(stat)
        return self
    def raw(self,frame):
        limit=frame.LIMIT_BAL.to_numpy(float).clip(1)
        bill=frame[BILLS].to_numpy(float)[:,::-1]
        payment=frame[PAYMENTS].to_numpy(float)[:,::-1]
        seq=np.stack([signedlog(bill),signedlog(payment),bill/limit[:,None],payment/limit[:,None]],axis=-1)
        return seq,signedlog(limit[:,None])
    def transform(self,frame):
        seq,stat=self.raw(frame)
        z=self.seq_scaler.transform(self.seq_imputer.transform(seq.reshape(-1,4))).reshape(len(frame),6,4)
        status=frame[STATUS].to_numpy(float)[:,::-1]
        cats=np.arange(-2,10)
        onehot=(status[:,:,None]==cats).astype(float)
        missing=(~np.isin(status,cats))[:,:,None].astype(float)
        return (np.concatenate([z,onehot,missing],axis=-1).astype('float32'),
                self.stat_scaler.transform(stat).astype('float32'))

class ResidualNet(nn.Module):
    def __init__(self, d, width=64, dropout=.2):
        super().__init__()
        self.input=nn.Linear(d,width)
        self.blocks=nn.ModuleList([nn.Sequential(nn.LayerNorm(width),nn.Linear(width,width*2),
                   nn.ReLU(),nn.Dropout(dropout),nn.Linear(width*2,width),nn.Dropout(dropout)) for _ in range(2)])
        self.head=nn.Sequential(nn.LayerNorm(width),nn.ReLU(),nn.Linear(width,1))
    def forward(self,x):
        h=self.input(x)
        for block in self.blocks: h=h+block(h)
        return self.head(h).squeeze(-1)

class MonthlyGRU(nn.Module):
    def __init__(self,d=17,width=32,dropout=.2):
        super().__init__()
        self.gru=nn.GRU(d,width,batch_first=True)
        self.static=nn.Sequential(nn.Linear(1,16),nn.ReLU())
        self.head=nn.Sequential(nn.Linear(width+16,32),nn.ReLU(),nn.Dropout(dropout),nn.Linear(32,1))
    def forward(self,seq,static):
        _,h=self.gru(seq)
        return self.head(torch.cat([h[-1],self.static(static)],dim=1)).squeeze(-1)

def tensors(arr):
    if not isinstance(arr,tuple): arr=(arr,)
    return tuple(torch.from_numpy(np.asarray(a,dtype='float32')) for a in arr)

def neural_predict(model,arr):
    model.eval();tx=tensors(arr);out=[]
    with torch.no_grad():
        for start in range(0,len(tx[0]),1024):
            out.append(torch.sigmoid(model(*(a[start:start+1024] for a in tx))).numpy())
    return np.concatenate(out)

def fit_neural(kind, arrays, y, config, seed, validation=None, fixed_epochs=None):
    torch.manual_seed(seed);np.random.seed(seed)
    if kind=='resnet': model=ResidualNet(np.asarray(arrays).shape[1],**config)
    else: model=MonthlyGRU(**config)
    optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.001)
    tx=tensors(arrays);ty=torch.tensor(y.to_numpy(),dtype=torch.float32)
    rng=np.random.default_rng(seed); best_loss=float('inf');best=None;best_epoch=1;stale=0;history=[]
    for epoch in range(1,(fixed_epochs or 60)+1):
        model.train();order=rng.permutation(len(y));total=0.
        for start in range(0,len(y),256):
            ix=order[start:start+256];optimizer.zero_grad(set_to_none=True)
            loss=nn.functional.binary_cross_entropy_with_logits(model(*(a[ix] for a in tx)),ty[ix])
            loss.backward();nn.utils.clip_grad_norm_(model.parameters(),5.);optimizer.step()
            total+=float(loss.detach())*len(ix)
        rec={'epoch':epoch,'train_logloss':total/len(y)}
        if validation is not None:
            p=neural_predict(model,validation[0]);vl=log_loss(validation[1],p)
            rec['validation_logloss']=float(vl)
            if vl<best_loss-1e-5:
                best_loss=vl;best=copy.deepcopy(model.state_dict());best_epoch=epoch;stale=0
            else: stale+=1
            history.append(rec)
            if stale>=8: break
        else: history.append(rec)
    if best is not None: model.load_state_dict(best)
    return model,best_epoch if validation is not None else fixed_epochs,history

def calibration(y,p):
    p=np.clip(p,1e-6,1-1e-6)
    m=LogisticRegression(C=1e6,max_iter=1000).fit(logit(p).reshape(-1,1),y)
    return m

def calibrate(m,p):
    return m.predict_proba(logit(np.clip(p,1e-6,1-1e-6)).reshape(-1,1))[:,1]

def wilson(k,n):
    if not n:return [None,None]
    z=norm.ppf(.975);d=1+z*z/n;mid=(k/n+z*z/(2*n))/d
    half=z*np.sqrt((k/n)*(1-k/n)/n+z*z/(4*n*n))/d
    return [float(max(0,mid-half)),float(min(1,mid+half))]

def top_stats(y,p,fraction=.2):
    y=np.asarray(y);ix=np.argsort(-p,kind='stable')[:max(1,int(np.ceil(len(y)*fraction)))]
    captured=int(y[ix].sum());precision=float(y[ix].mean())
    return {'reviewed':len(ix),'defaults_captured':captured,'capture_rate':float(captured/y.sum()),
            'precision':precision,'lift':float(precision/y.mean())}

def metrics(y,p,reference_rate=None):
    y=np.asarray(y);p=np.clip(p,1e-6,1-1e-6);m=calibration(y,p)
    fpr,tpr,_=roc_curve(y,p)
    baseline=reference_rate if reference_rate is not None else y.mean()
    return {'roc_auc':float(roc_auc_score(y,p)),'average_precision':float(average_precision_score(y,p)),
            'log_loss':float(log_loss(y,p)),'brier':float(brier_score_loss(y,p)),
            'probability_mse_reduction_vs_constant':float(1-brier_score_loss(y,p)/brier_score_loss(y,np.repeat(baseline,len(y)))),
            'ks':float(np.max(tpr-fpr)),'calibration_intercept':float(m.intercept_[0]),
            'calibration_slope':float(m.coef_[0,0]),'predicted_rate':float(p.mean()),
            'observed_rate':float(y.mean()),'observed_over_expected':float(y.mean()/p.mean()),
            'top20':top_stats(y,p),'top10':top_stats(y,p,.1)}

def bins(y,p,n=10):
    frame=pd.DataFrame({'y':np.asarray(y),'p':p})
    frame['bin']=pd.qcut(frame.p.rank(method='first'),n,labels=False)
    rows=[]
    for b,g in frame.groupby('bin'):
        k=int(g.y.sum());ci=wilson(k,len(g))
        rows.append({'risk_decile':int(b+1),'n':len(g),'defaults':k,'mean_prediction':float(g.p.mean()),
                     'observed_rate':float(g.y.mean()),'observed_wilson95':ci})
    return rows

def fixed_policy(y,p,threshold):
    y=np.asarray(y);review=p>=threshold
    return {'threshold':float(threshold),'review_fraction':float(review.mean()),
     'defaults_captured':int(y[review].sum()),'capture_rate':float(y[review].sum()/y.sum()),
     'missed_defaults':int(y[~review].sum()),'nondefault_reviews':int(((y==0)&review).sum())}

def audit(y,p,g,threshold):
    y=np.asarray(y);g=np.asarray(g).astype(str);rows=[]
    for name in sorted(set(g)):
        mask=g==name;n=int(mask.sum());yy=y[mask];pp=p[mask];d=pp>=threshold
        rows.append({'group':name,'n':n,'defaults':int(yy.sum()),'observed_rate':float(yy.mean()),
         'observed_wilson95':wilson(int(yy.sum()),n),'predicted_rate':float(pp.mean()),
         'auc':float(roc_auc_score(yy,pp)) if len(set(yy))==2 else None,
         'review_fraction':float(d.mean()),'missed_default_rate':float((~d[yy==1]).mean()) if yy.sum() else None,
         'nondefault_review_rate':float(d[yy==0].mean()) if (yy==0).any() else None,
         'low_support':n<200 or yy.sum()<30})
    return rows

def paired_bootstrap(y, probs,groups, champion, draws=500):
    rng=np.random.default_rng(884);y=np.asarray(y);unique=np.unique(groups)
    lookup={g:np.flatnonzero(groups==g) for g in unique}
    targets=['roc_auc','average_precision','top20_capture']
    values={name:{t:[] for t in targets} for name in probs}
    for _ in range(draws):
        ix=np.concatenate([lookup[g] for g in rng.choice(unique,len(unique),replace=True)])
        if len(np.unique(y[ix]))<2:continue
        for name,p in probs.items():
            values[name]['roc_auc'].append(roc_auc_score(y[ix],p[ix]))
            values[name]['average_precision'].append(average_precision_score(y[ix],p[ix]))
            values[name]['top20_capture'].append(top_stats(y[ix],p[ix])['capture_rate'])
    out={}
    for name in probs:
        out[name]={t:{'ci95':np.quantile(values[name][t],[.025,.975]).tolist(),
          'delta_vs_champion_ci95':np.quantile(np.array(values[name][t])-values[champion][t],[.025,.975]).tolist()}
          for t in targets}
    return out

def prior_shift(y,p,new_prior,old_prior):
    """Pure label shift: fixed p(x|y), new P(y); not a recession model."""
    # Resampling weights use THIS sample's empirical prior. The odds correction
    # separately uses the reference prior of the calibrated model.
    sample_prior=float(np.mean(y))
    w=np.where(np.asarray(y)==1,new_prior/sample_prior,(1-new_prior)/(1-sample_prior))
    corrected=expit(logit(np.clip(p,1e-6,1-1e-6))+logit(new_prior)-logit(old_prior))
    return {'assumed_default_rate':new_prior,'reference_calibration_prior':old_prior,
      'reporting_sample_prior':sample_prior,
      'weighted_observed_rate':float(np.average(y,weights=w)),
      'unadjusted_predicted_rate':float(np.average(p,weights=w)),
      'odds_adjusted_predicted_rate':float(np.average(corrected,weights=w)),
      'unadjusted_brier':float(np.average((p-y)**2,weights=w)),
      'odds_adjusted_brier':float(np.average((corrected-y)**2,weights=w))}

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--data',type=Path,default=Path('data/source.xls'))
    parser.add_argument('--out',type=Path,default=Path('results'));args=parser.parse_args()
    out=args.out;out.mkdir(parents=True,exist_ok=True);(out/'models').mkdir(exist_ok=True)
    dump(out/'protocol.json',PROTOCOL)  # written before any model is fitted
    if not args.data.exists():
        args.data.parent.mkdir(parents=True,exist_ok=True)
        url='https://archive.ics.uci.edu/static/public/350/default+of+credit+card+clients.zip'
        archive=args.data.with_suffix('.zip');urllib.request.urlretrieve(url,archive)
        with zipfile.ZipFile(archive) as z:
            args.data.write_bytes(z.read(next(n for n in z.namelist() if n.lower().endswith('.xls'))))
    checksum=hashlib.sha256(args.data.read_bytes()).hexdigest()
    expected='30c6be3abd8dcfd3e6096c828bad8c2f011238620f5369220bd60cfc82700933'
    if checksum!=expected:raise ValueError('UCI workbook checksum changed; review data before fitting.')
    df=pd.read_excel(args.data,header=1);df=df.rename(columns={'default payment next month':'TARGET'})
    y=df.TARGET.astype(int)
    assert df.shape==(30000,25) and df.ID.is_unique and set(y)=={0,1}
    assert df[FINANCIAL].notna().all().all() and df.LIMIT_BAL.gt(0).all()
    groups=pd.util.hash_pandas_object(df[FINANCIAL],index=False).to_numpy()
    dev,test=train_test_split(np.arange(len(df)),test_size=.2,stratify=y,random_state=42)
    overlaps=np.isin(groups[dev],groups[test]);removed=dev[overlaps];dev=dev[~overlaps]
    split=StratifiedGroupKFold(n_splits=5,shuffle=True,random_state=2026)
    folds=list(split.split(df.iloc[dev],y.iloc[dev],groups[dev]))
    tune=dev[folds[0][1]];cal=dev[folds[1][1]]
    fit=np.setdiff1d(dev,np.r_[tune,cal]);refit=np.r_[fit,tune]
    partitions={'fit':fit,'tune':tune,'calibration':cal,'reporting':test}
    for a,ia in partitions.items():
        for b,ib in partitions.items():
            if a!=b:assert not set(groups[ia])&set(groups[ib])
    manifest={'source_sha256':checksum,'removed_development_rows_matching_reporting_features':len(removed),
      'duplicate_feature_rows':int(pd.Series(groups).duplicated().sum()),
      'conflicting_label_feature_groups':int(pd.DataFrame({'g':groups,'y':y}).groupby('g').y.nunique().gt(1).sum()),
      'partitions':{name:{'n':len(ix),'default_rate':float(y.iloc[ix].mean()),'indices_sha256':hashlib.sha256(ix.tobytes()).hexdigest()}
                    for name,ix in partitions.items()}}
    dump(out/'data_manifest.json',manifest)
    np.savez(out/'split_indices.npz',**partitions,excluded_development=removed)
    dump(out/'environment.json',{'python':platform.python_version(),'torch':torch.__version__,
                                'lightgbm':lightgbm.__version__,'numpy':np.__version__,
                                'pandas':pd.__version__,'seeds':SEEDS,'threads':2})
    print('SPLITS',manifest,flush=True)
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    X=features(df);pflat=prep(X.iloc[fit]).fit(X.iloc[fit]);mp=MonthlyPrep().fit(df.iloc[fit])
    flat=pflat.transform(X).astype('float32');monthly=mp.transform(df)
    vals={};chosen={};tuning=[];neural_epochs={};timing={}
    start=time.time()
    for family in ['logistic','lightgbm']:
        best=-1.;begin=time.time()
        configs=([{'C':v} for v in [.01,.1,1.]] if family=='logistic' else
                 [{'num_leaves':15,'min_child_samples':100,'reg_lambda':10.},
                  {'num_leaves':31,'min_child_samples':100,'reg_lambda':10.},
                  {'num_leaves':15,'min_child_samples':200,'reg_lambda':20.}])
        for cfg in configs:
            if family=='logistic':m=LogisticRegression(**cfg,max_iter=1500,solver='lbfgs')
            else:m=LGBMClassifier(**cfg,n_estimators=450,learning_rate=.03,colsample_bytree=.9,
                                  random_state=42,n_jobs=2,verbosity=-1)
            xx=flat if family=='logistic' else X
            m.fit(xx[fit] if isinstance(xx,np.ndarray) else xx.iloc[fit],y.iloc[fit])
            pp=m.predict_proba(xx[tune] if isinstance(xx,np.ndarray) else xx.iloc[tune])[:,1]
            ap=average_precision_score(y.iloc[tune],pp)
            tuning.append({'family':family,'config':cfg,'ap':float(ap)})
            print('TUNE',family,cfg,ap,flush=True)
            if ap>best:best=ap;chosen[family]=cfg;vals[family]=pp
        timing[family+'_tuning_seconds']=time.time()-begin
    for family in ['resnet','gru']:
        begin=time.time();best=-1.
        configs=([{'width':64,'dropout':.2},{'width':128,'dropout':.3}] if family=='resnet' else
                 [{'width':24,'dropout':.2},{'width':48,'dropout':.3}])
        arr=flat if family=='resnet' else monthly
        select=lambda ix: arr[ix] if isinstance(arr,np.ndarray) else tuple(a[ix] for a in arr)
        for cfg in configs:
            m,ep,hist=fit_neural(family,select(fit),y.iloc[fit],cfg,42,validation=(select(tune),y.iloc[tune]))
            pp=neural_predict(m,select(tune));ap=average_precision_score(y.iloc[tune],pp)
            tuning.append({'family':family,'config':cfg,'seed':42,'ap':float(ap),'best_epoch':ep,'history':hist})
            print('TUNE',family,cfg,ap,'epochs',ep,flush=True)
            if ap>best:best=ap;chosen[family]=cfg;vals[family]=pp;neural_epochs[family]=[ep]
        ps=[vals[family]]
        for seed in SEEDS[1:]:
            m,ep,hist=fit_neural(family,select(fit),y.iloc[fit],chosen[family],seed,validation=(select(tune),y.iloc[tune]))
            pp=neural_predict(m,select(tune));ps.append(pp);neural_epochs[family].append(ep)
            tuning.append({'family':family,'selected_config':True,'seed':seed,'ap':float(average_precision_score(y.iloc[tune],pp)),
                           'best_epoch':ep,'history':hist})
            print('SEED',family,seed,'epochs',ep,flush=True)
        vals[family]=np.mean(ps,axis=0);timing[family+'_tuning_seconds']=time.time()-begin
    bestnn=max(['resnet','gru'],key=lambda k:average_precision_score(y.iloc[tune],vals[k]))
    vals['blend']=(vals['lightgbm']+vals[bestnn])*.5
    champion=max(vals,key=lambda k:average_precision_score(y.iloc[tune],vals[k]))
    selected={'champion':champion,'best_neural':bestnn,'configs':chosen,
              'median_epochs':{k:int(np.median(v)) for k,v in neural_epochs.items()},
              'tuning_average_precision':{k:float(average_precision_score(y.iloc[tune],v)) for k,v in vals.items()},
              'review_thresholds':{k:float(np.quantile(v,.8)) for k,v in vals.items()},
              'note':'Thresholds are raw-score development thresholds; final-refit scale may change, which is measured.'}
    dump(out/'tuning.json',tuning);dump(out/'selection_locked_before_reporting.json',selected)
    print('LOCKED_SELECTION',selected,flush=True)
    # No choice below depends on reporting labels.
    pf=prep(X.iloc[refit]).fit(X.iloc[refit]);mf=MonthlyPrep().fit(df.iloc[refit])
    zz=pf.transform(X).astype('float32');mm=mf.transform(df)
    final={};raw={};cals={};test_probs={};calprobs={};test_raw={}
    joblib.dump({'flat':pf,'monthly':mf},out/'models/preprocessing.joblib')
    for family in ['logistic','lightgbm','resnet','gru']:
        begin=time.time()
        if family=='logistic':
            m=LogisticRegression(**chosen[family],max_iter=1500,solver='lbfgs').fit(zz[refit],y.iloc[refit])
            final[family]=m;raw[family]=m.predict_proba(zz)[:,1]
        elif family=='lightgbm':
            m=LGBMClassifier(**chosen[family],n_estimators=450,learning_rate=.03,colsample_bytree=.9,
                            random_state=42,n_jobs=2,verbosity=-1).fit(X.iloc[refit],y.iloc[refit])
            final[family]=m;raw[family]=m.predict_proba(X)[:,1]
        else:
            arr=zz if family=='resnet' else mm
            subset=arr[refit] if isinstance(arr,np.ndarray) else tuple(a[refit] for a in arr)
            models=[];pp=[]
            for seed in SEEDS:
                m,_,_=fit_neural(family,subset,y.iloc[refit],chosen[family],seed,
                                fixed_epochs=selected['median_epochs'][family])
                pp.append(neural_predict(m,arr));models.append(m)
                torch.save(m.state_dict(),out/f'models/{family}_{seed}.pt')
            final[family]=models;raw[family]=np.mean(pp,axis=0)
        timing[family+'_final_fit_seconds']=time.time()-begin
        if family in ['logistic','lightgbm']:joblib.dump(final[family],out/f'models/{family}.joblib')
        print('REFIT_DONE',family,flush=True)
    raw['blend']=.5*(raw['lightgbm']+raw[bestnn])
    thresholds={}
    for family in raw:
        c=calibration(y.iloc[cal],raw[family][cal]);cals[family]=c
        calprobs[family]=calibrate(c,raw[family][cal]);test_probs[family]=calibrate(c,raw[family][test])
        test_raw[family]=raw[family][test]
        # Map development score cut-off through the final independent calibrator.
        thresholds[family]=float(calibrate(c,np.array([selected['review_thresholds'][family]]))[0])
    joblib.dump(cals,out/'models/calibrators.joblib')
    # One scheduled reporting pass. All diagnostics below are descriptive, not used to select or retune.
    yy=y.iloc[test].to_numpy();reference=float(y.iloc[refit].mean())
    report={'selection':selected,'models':{},'calibration_sample_size':len(cal)}
    for family,p in test_probs.items():
        report['models'][family]={'raw':metrics(yy,test_raw[family],reference),
           'calibrated':metrics(yy,p,reference),'risk_deciles':bins(yy,p),
           'fixed_review_policy':fixed_policy(yy,p,thresholds[family])}
    report['cluster_bootstrap']=paired_bootstrap(yy,test_probs,groups[test],champion)
    cp=test_probs[champion];threshold=thresholds[champion]
    report['subgroups']={'sex':audit(yy,cp,df.iloc[test].SEX,threshold),
     'age':audit(yy,cp,pd.cut(df.iloc[test].AGE,[0,30,45,200],labels=['under30','30to44','45plus'],right=False).astype(str),threshold),
     'recorded_delay':audit(yy,cp,np.where(df.iloc[test].PAY_0.ge(1),'delay_code_ge1','other_status_codes'),threshold)}
    report['pure_prior_shift']=prior_shift(yy,cp,.35,float(y.iloc[cal].mean()))
    report['sensitivity']={}
    def predict_new(frame,family):
        xf=features(frame)
        if family=='logistic':return final[family].predict_proba(pf.transform(xf).astype('float32'))[:,1]
        if family=='lightgbm':return final[family].predict_proba(xf)[:,1]
        if family=='blend':return .5*(predict_new(frame,'lightgbm')+predict_new(frame,bestnn))
        arr=pf.transform(xf).astype('float32') if family=='resnet' else mf.transform(frame)
        return np.mean([neural_predict(m,arr) for m in final[family]],axis=0)
    for scenario in ['missing_latest_month','delay_codes_plus1']:
        altered=df.iloc[test].copy()
        if scenario=='missing_latest_month':altered[['PAY_0','BILL_AMT1','PAY_AMT1']]=np.nan
        else:altered['PAY_0']=np.where(altered.PAY_0.ge(0),np.minimum(altered.PAY_0+1,9),altered.PAY_0)
        pp=calibrate(cals[champion],predict_new(altered,champion))
        report['sensitivity'][scenario]={'mean_score_change':float((pp-cp).mean()),
          'median_absolute_score_change':float(np.median(np.abs(pp-cp))),
          'p95_absolute_score_change':float(np.quantile(np.abs(pp-cp),.95)),
          'review_flag_change_fraction':float(np.mean((pp>=threshold)!=(cp>=threshold))),
          'note':'Labels held fixed only for identity matching; no outcome-performance or causal claim from perturbed cases.'}
    # Grouped permutation: correlated monthly status/bill/payment fields move together.
    rng=np.random.default_rng(909);importance=[];base_ap=average_precision_score(yy,cp)
    for group_name,cols in [('repayment_status',STATUS),('billed_balances',BILLS),('payment_amounts',PAYMENTS),('credit_limit',['LIMIT_BAL'])]:
        drops=[]
        for _ in range(5):
            altered=df.iloc[test].copy();perm=rng.permutation(len(test))
            altered.loc[:,cols]=altered[cols].to_numpy()[perm]
            pp=calibrate(cals[champion],predict_new(altered,champion))
            drops.append(base_ap-average_precision_score(yy,pp))
        importance.append({'group':group_name,'mean_ap_drop':float(np.mean(drops)),
                           'sd_ap_drop':float(np.std(drops,ddof=1))})
    report['grouped_permutation']=importance
    report['timing']=timing;report['total_seconds']=time.time()-start
    pd.DataFrame({'row_index':test,'financial_group':groups[test],'y':yy,**test_probs}).to_csv(out/'reporting_predictions.csv',index=False)
    dump(out/'results.json',report)
    # Static research figures, separate from CV.
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(1,2,figsize=(10.5,4.2),layout='constrained')
    for family,p in test_probs.items():
        rows=bins(yy,p);axes[0].plot([r['mean_prediction'] for r in rows],[r['observed_rate'] for r in rows],marker='.',label=family)
        order=np.argsort(-p,kind='stable');axes[1].plot(np.arange(1,len(yy)+1)/len(yy),np.cumsum(yy[order])/yy.sum(),label=family)
    axes[0].plot([0,1],[0,1],'--',c='grey');axes[0].set(xlabel='Mean predicted default probability',ylabel='Observed default fraction',title='Probability calibration (10 risk groups)')
    axes[1].plot([0,1],[0,1],'--',c='grey');axes[1].axvline(.2,c='grey',lw=.8)
    axes[1].set(xlabel='Fraction of accounts reviewed',ylabel='Fraction of defaults captured',title='Risk-ranking concentration')
    axes[1].legend(fontsize=8);fig.savefig(out/'calibration_and_capture.png',dpi=180);plt.close(fig)
    rows=report['models'][champion]['risk_deciles'];fig,ax=plt.subplots(figsize=(8,4),layout='constrained')
    ax.bar([r['risk_decile'] for r in rows],[r['observed_rate']*100 for r in rows],color='#155e75',label='Observed')
    ax.plot([r['risk_decile'] for r in rows],[r['mean_prediction']*100 for r in rows],'o-',color='#ea580c',label='Predicted')
    ax.set(xlabel='Risk decile (1 lowest, 10 highest)',ylabel='Default rate (%)',title=f'Descriptive internal validation: {champion}');ax.legend()
    fig.savefig(out/'risk_deciles.png',dpi=180);plt.close(fig)
    print('REPORT',json.dumps({k:v['calibrated'] for k,v in report['models'].items()},indent=2),flush=True)

if __name__=='__main__':
    # Keep serialized custom preprocessing classes/functions importable rather
    # than recording script-local __main__ references in the saved artifacts.
    from experiment import main as run_experiment
    run_experiment()
