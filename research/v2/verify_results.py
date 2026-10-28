"""Verify saved-model predictions and reported denominators without refitting."""
import argparse,json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score,average_precision_score
from experiment import MonthlyGRU,MonthlyPrep,signedlog,neural_predict,calibrate,SEEDS,top_stats

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--data',type=Path,required=True)
    parser.add_argument('--results',type=Path,default=Path('results'))
    a=parser.parse_args();p=a.results
    selection=json.loads((p/'selection_locked_before_reporting.json').read_text())
    report=json.loads((p/'results.json').read_text())
    frame=pd.read_excel(a.data,header=1)
    expected=pd.read_csv(p/'reporting_predictions.csv')
    preprocessing=joblib.load(p/'models/preprocessing.joblib')
    calibrators=joblib.load(p/'models/calibrators.joblib')
    arr=preprocessing['monthly'].transform(frame.iloc[expected.row_index])
    torch.set_num_threads(2);predictions=[]
    for seed in SEEDS:
        model=MonthlyGRU(**selection['configs']['gru'])
        model.load_state_dict(torch.load(p/f'models/gru_{seed}.pt',map_location='cpu',weights_only=True))
        predictions.append(neural_predict(model,arr))
    p_gru=calibrate(calibrators['gru'],np.mean(predictions,axis=0))
    assert np.allclose(p_gru,expected.gru,atol=1e-6,rtol=1e-6)
    metrics=report['models']['gru']['calibrated']
    assert np.isclose(roc_auc_score(expected.y,expected.gru),metrics['roc_auc'])
    assert np.isclose(average_precision_score(expected.y,expected.gru),metrics['average_precision'])
    assert top_stats(expected.y,expected.gru)==metrics['top20']
    assert len(expected)==6000 and expected.y.sum()==1327
    assert np.isclose(report['pure_prior_shift']['weighted_observed_rate'],.35)
    print('PASS: saved GRU ensemble reproduces all 6,000 calibrated predictions.')
    print('PASS: AUC, average precision, capture counts, default count and stress denominator match the report.')

if __name__=='__main__':main()
