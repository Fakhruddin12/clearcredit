import sys
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from challenge import latest_features,representation,permute_months,queue,tie_expectation,tabular_prep,PROTOCOL

def test_latest_month_excludes_history_and_target():
 frame=pd.DataFrame({'LIMIT_BAL':[100.],'PAY_0':[1],'BILL_AMT1':[30.],'PAY_AMT1':[5.],'PAY_2':[9],'TARGET':[1],'SEX':[2]})
 x=latest_features(frame)
 assert list(x)==['LIMIT_BAL','PAY_0','BILL_AMT1','PAY_AMT1','UTIL_1','PAY_LIMIT_1']
 assert x.UTIL_1.iloc[0]==.3

def test_shuffle_preserves_aligned_month_vectors():
 seq=np.arange(4*6*17).reshape(4,6,17);changed=permute_months(seq)
 for original,new in zip(seq,changed):
  assert sorted(map(tuple,original))==sorted(map(tuple,new))
 assert not np.array_equal(seq,changed)
 assert np.array_equal(changed,permute_months(seq))

def test_flatten_retains_all_inputs():
 seq=np.arange(3*6*17).reshape(3,6,17).astype('float32');static=np.ones((3,1),dtype='float32')
 flat=representation((seq,static),'flat_mlp')
 assert flat.shape==(3,103)
 assert np.array_equal(flat[:,:102].reshape(3,6,17),seq)
 assert np.array_equal(flat[:,-1:],static)

def test_no_status_removes_every_status_channel():
 seq=np.ones((3,6,17));static=np.ones((3,1))
 arr=representation((seq,static),'no_status_gru')
 assert arr[0].shape==(3,6,4)
 assert np.array_equal(arr[1],static)

def test_queue_has_exact_capacity_with_ties():
 y=np.array([1,0,1,0,0]);p=np.ones(5);tie=np.array([.9,.8,.7,.6,.5])
 assert queue(y,p,tie,.4)=={'reviewed':2,'defaults_captured':0,'capture_rate':0.,'precision':0.}
 assert np.isclose(tie_expectation(y,p,.4),.8)

def test_tie_expectation_boundary():
 y=np.array([1,1,0,0,1]);p=np.array([2.,1.,1.,0.,0.])
 assert tie_expectation(y,p,.4)==1.5

def test_queue_selection_does_not_use_labels():
 p=np.array([1.,1.,0.,0.,0.]);tie=np.arange(5.)
 a=queue(np.array([1,0,0,0,0]),p,tie,.2)
 b=queue(np.array([0,1,0,0,0]),p,tie,.2)
 assert a['defaults_captured']==1 and b['defaults_captured']==0

def test_single_status_preprocessor_accepts_one_column():
 x=pd.DataFrame({'PAY_0':[-2,-1,0,1,2]})
 z=tabular_prep(x).fit_transform(x)
 assert z.shape==(5,13)

def test_exploratory_status_and_common_primary_objective_explicit():
 assert 'not confirmatory' in PROTOCOL['status']
 assert 'exactly 20%' in PROTOCOL['primary_objective']
 assert 'First three original seeds' in PROTOCOL['primary_ensemble']
