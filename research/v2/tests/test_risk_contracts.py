"""Checks for economically meaningful mistakes, rather than estimator internals."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
from scipy.special import expit,logit
from experiment import (STATUS,BILLS,PAYMENTS,features,MonthlyPrep,top_stats,
                        fixed_policy,prior_shift,calibration,calibrate,wilson)

def account_frame():
    data={'LIMIT_BAL':[100.,200.],'TARGET':[0,1],'SEX':[1,2],
          'AGE':[22,66],'EDUCATION':[1,2],'MARRIAGE':[1,2],'ID':[11,12]}
    for i,c in enumerate(STATUS): data[c]=[float(i),float(i)]
    for i,c in enumerate(BILLS,1): data[c]=[float(i*10),float(i*20)]
    for i,c in enumerate(PAYMENTS,1): data[c]=[float(i),float(i*2)]
    return pd.DataFrame(data)

def test_outcomes_identity_and_demographics_cannot_enter_predictors():
    df=account_frame();x=features(df)
    for c in ['TARGET','SEX','AGE','EDUCATION','MARRIAGE','ID']:df[c]=999
    pd.testing.assert_frame_equal(x,features(df))
    assert not set(['TARGET','SEX','AGE','EDUCATION','MARRIAGE','ID'])&set(x.columns)

def test_months_are_oldest_to_newest_without_reversing_channel_alignment():
    df=account_frame();seq,stat=MonthlyPrep().raw(df)
    # First month is BILL_AMT6 / PAY_AMT6; last is BILL_AMT1 / PAY_AMT1.
    assert np.isclose(seq[0,0,0],np.log1p(60.))
    assert np.isclose(seq[0,0,1],np.log1p(6.))
    assert np.isclose(seq[0,-1,0],np.log1p(10.))
    assert np.isclose(seq[0,-1,2],.1)

def test_review_budget_denominator_is_all_defaults_not_reviewed_accounts():
    y=np.array([1,0,1,0,0]);p=np.array([.9,.8,.6,.3,.1])
    r=top_stats(y,p,.4)
    assert r['reviewed']==2 and r['defaults_captured']==1
    assert r['capture_rate']==.5 and r['precision']==.5
    assert np.isclose(r['lift'],1.25)
    policy=fixed_policy(y,p,.7)
    assert policy['missed_defaults']==1 and policy['nondefault_reviews']==1

def test_pure_prior_shift_is_correct_for_an_uninformative_calibrated_model():
    y=np.array([1,0,0,0,0]);p=np.full(5,.2)
    r=prior_shift(y,p,.35,.2)
    assert np.isclose(r['weighted_observed_rate'],.35)
    assert np.isclose(r['odds_adjusted_predicted_rate'],.35)
    assert r['odds_adjusted_brier']<r['unadjusted_brier']

def test_missing_status_is_not_silently_encoded_as_zero_delinquency():
    df=account_frame();prep=MonthlyPrep().fit(df)
    df.loc[0,'PAY_0']=np.nan
    seq,_=prep.transform(df)
    assert seq[0,-1,-1]==1 and seq[0,-1,4:-1].sum()==0
    assert np.isfinite(seq).all()

def test_stress_sample_weights_do_not_confuse_calibration_and_reporting_priors():
    y=np.array([1,0,0,0,0]);p=np.full(5,.22)
    r=prior_shift(y,p,.35,.22)
    assert np.isclose(r['weighted_observed_rate'],.35)
    assert r['reference_calibration_prior']==.22
    assert r['reporting_sample_prior']==.2

def test_wilson_interval_respects_probability_bounds_at_extremes():
    for k in [0,1,10]:
        lo,hi=wilson(k,10)
        assert 0<=lo<=hi<=1
        assert lo-1e-12<=k/10<=hi+1e-12
