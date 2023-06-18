# util functions for survival analysis
import os
import re
import numpy as np
import pandas as pd
import torch
import pycox.models as pm


def load_all_csv(cf, train_only=False):
    label_dir = cf['label_dir'][0]
    if re.search('.csv$', label_dir) is not None:
        label_dir = os.path.dirname(label_dir)
    label_path = cf['root']+'/'+label_dir+'/train.csv'

    train_df = pd.read_csv(label_path)
    if train_only:
        return train_df
    else:
        all_dfs = [train_df] + [pd.read_csv(cf['root']+'/'+i
                                            ) for i in cf['label_dir']]
        return pd.concat(all_dfs, axis=0)


def get_labtrans(cf):
    # select the transformation function
    if cf['loss_type'] == 'logistic_hazard':
        trans_func = pm.LogisticHazard
    elif cf['loss_type'] == 'pmf':
        trans_func = pm.PMF
    elif cf['loss_type'] == 'deephit':
        trans_func = pm.DeepHitSingle
    else:
        trans_func = None

    if trans_func is not None:
        all_dfs = load_all_csv(cf, train_only=False)
        train_df = load_all_csv(cf, train_only=True)
        out_col = cf['outcome_col']
        cf['max_duration'] = max(all_dfs[out_col+'_time'].values)

        max_time = max(all_dfs[out_col+'_time'].values)
        num_duration = np.ceil(max_time/cf['interval'])
        cf['labtrans'] = trans_func.label_transform(int(num_duration))
        ytrain = cf['labtrans'].fit_transform(
                train_df[out_col+'_time'].values,
                train_df[out_col+'_bool'].values)
        cf['trans_func'] = trans_func
    else:
        cf['labtrans'] = None
    return cf


def output2surv(xx: torch.tensor, cf: dict, epsilon: float=1e-7):
    '''
    Args:
        `xx`: output from neural network, in dimension of sample x time points
        for discrete model, and sample x 1 for continuous model
    Returns:
        a pandas dataframe of survival probabilities, time points x sample,
        rownames being the actual time point values
    Reference:
    https://github.com/havakv/pycox/blob/refactor_out_torchtuples/pycox/models/deephit.py
    '''
    if cf['loss_type'] == 'logistic_hazard':
        hazard = torch.sigmoid(xx)
        survival = (1 - hazard).add(epsilon).log().cumsum(1).exp()
    elif cf['loss_type'] == 'deephit':
        inp = xx.unsqueeze(1)  # axis 1 is for different event classes
        pmf = pm.utils.pad_col(inp.view(xx.size(0), -1)).softmax(1)[:, :-1]
        pmf = pmf.view(inp.shape).transpose(0, 1).transpose(1, 2)
        cif = pmf.cumsum(1)
        survival = (1 - cif.sum(0)).T
    elif cf['loss_type'] == 'pmf':
        pmf = pm.utils.pad_col(xx).softmax(1)[:, :-1]
        survival = 1 - pmf.cumsum(1)
    elif 'cox' in cf['loss_type']:
        survival = coxph2surv(xx, cf)

    # for discrete time analysis
    if cf['labtrans'] is not None:
        survival = pd.DataFrame(
                survival.numpy().transpose(),
                cf['labtrans'].cuts)
    return survival

# --------------------Continuous time model--------------------
def compute_baseline_hazard (output, target, cf):
    df_target = pd.DataFrame.from_dict ({'duration': target[0], 
        'status': target[1]})
    baseline_hazard = (df_target
    .assign(expg=np.exp(output))
    .groupby('duration')
    .agg({'expg': 'sum', 'status': 'sum'})
    .sort_index(ascending=False)
    .assign(expg=lambda x: x['expg'].cumsum())
    .pipe(lambda x: x['status']/x['expg'])
    .fillna(0.)
    .iloc[::-1]
    .loc[lambda x: x.index <= cf['max_duration']]
    .rename('baseline_hazards'))
    assert baseline_hazard.index.is_monotonic_increasing
    return baseline_hazard

def coxph2surv (output, cf): 
    bch = cf['baseline_hazards'].cumsum()
    bch = bch.loc[lambda x: x.index <= cf['max_duration']]
    expg = np.exp(output).reshape(1, -1)
    cumu_hazard = pd.DataFrame(bch.values.reshape(-1, 1).dot(expg), 
                        index=bch.index)
    return np.exp(-cumu_hazard)
