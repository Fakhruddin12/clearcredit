"""Restore fitted candidates and independently check headline counts/metrics."""
import argparse,json,sys
from pathlib import Path
import joblib,numpy as np,pandas as pd,torch
from sklearn.metrics import roc_auc_score,average_precision_score
from challenge import old,V2,PRIMARY_SEEDS,FlatMLP,representation,latest_features,measures,load_data

def main():
 p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True);p.add_argument('--results',type=Path,default=Path('results_reproduced'));p.add_argument('--baseline-results',type=Path,default=V2/'results_reproduced');a=p.parse_args()
 torch.set_num_threads(2);out=a.results
 df,y,groups,parts,refit=load_data(a.data);test=parts['reporting'];pred=pd.read_csv(out/'reporting_predictions.csv');report=json.loads((out/'results.json').read_text())
 assert np.array_equal(pred.row_index,test) and np.array_equal(pred.y,y.iloc[test])
 prep=joblib.load(out/'models/preprocessing.joblib');monthly=prep['monthly'].transform(df)
 restored={}
 for name in ['ordered_gru','flat_mlp','shuffled_gru','no_status_gru']:
  arr=representation(monthly,name);prob=[]
  for seed in PRIMARY_SEEDS:
   m=FlatMLP(arr.shape[1]) if name=='flat_mlp' else old.MonthlyGRU(d=arr[0].shape[2],width=24,dropout=.2)
   m.load_state_dict(torch.load(out/f'models/{name}_{seed}.pt',weights_only=True,map_location='cpu'))
   prob.append(old.neural_predict(m,arr))
  c=joblib.load(out/f'models/{name}_calibrator.joblib');restored[name]=old.calibrate(c,np.mean(prob,axis=0))[test]
 for name in ['latest_logistic','latest_lightgbm','latest_status_logistic']:
  saved=joblib.load(out/f'models/{name}.joblib');x=df[['PAY_0']] if name=='latest_status_logistic' else latest_features(df)
  xx=x if name=='latest_lightgbm' else saved['preprocessing'].transform(x).astype('float32')
  restored[name]=old.calibrate(saved['calibrator'],saved['model'].predict_proba(xx)[:,1])[test]
 baseline_cal=joblib.load(a.baseline_results/'models/calibrators.joblib')
 for name in ['logistic','lightgbm']:
  m=joblib.load(a.baseline_results/f'models/{name}.joblib');x=old.features(df)
  if name=='logistic':x=prep['flat'].transform(x).astype('float32')
  restored[name]=old.calibrate(baseline_cal[name],m.predict_proba(x)[:,1])[test]
 restored['binary_rule']=df.iloc[test].PAY_0.ge(1).to_numpy(float);restored['severity_rule']=df.iloc[test].PAY_0.clip(lower=0).to_numpy(float)
 for name,p in restored.items():
  assert np.allclose(p,pred[name],atol=1e-7,rtol=1e-7),name
  metric=measures(pred.y,pred[name],pred.tie_key,name not in ['binary_rule','severity_rule'])
  for k,v in metric.items():assert np.isclose(v,report['metrics'][name][k]),(name,k)
  assert metric['reviewed']==1200
  mask=df.iloc[test].PAY_0.lt(1).to_numpy()
  subgroup=measures(pred.y.to_numpy()[mask],pred[name].to_numpy()[mask],pred.tie_key.to_numpy()[mask],name not in ['binary_rule','severity_rule'])
  for k,v in subgroup.items():assert np.isclose(v,report['no_latest_delay']['metrics'][name][k]),(name,'subgroup',k)
 assert len(pred)==6000 and pred.y.sum()==1327
 print('PASS: restored all eleven candidate/rule score vectors on 6,000 accounts.')
 print('PASS: all aggregate and subgroup AUC, AP, capture, precision and probability metrics match saved results.')

if __name__=='__main__':main()
