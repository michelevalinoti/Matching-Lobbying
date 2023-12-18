#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 25 21:09:07 2023

@author: michelev
"""
import pandas as pd

import pyblp

#%%



reg_df_mod = reg_df.loc[(pd.isna(reg_df.people_id) == False) & (pd.isna(reg_df.bill_id) == False),:]
reg_df_mod = reg_df_mod.drop_duplicates(['bill_id', 'people_id', 'principal_lobbyist', 'beneficial_client'])
reg_df_mod['lobbyist_client'] = reg_df_mod.principal_lobbyist + reg_df_mod.beneficial_client 
reg_df_mod['politician_lobbyist'] = reg_df_mod.people_id.astype(str) + reg_df_mod.principal_lobbyist
reg_df_mod['politician_client'] = reg_df_mod.people_id.astype(str) + reg_df_mod.beneficial_client


models_dict = {}
models_dict['sponsored_the_bill'] = 'prices + is_sponsor + 0'
models_dict['sponsor_rank'] = 'prices + is_sponsor_1 + is_sponsor_2 + 0'#'prices + sponsor_type + I(sponsor_type**2) + 0'

FE_dict = {'session': 'C(session_id)',
               'politician': 'C(people_id)',
               'session-politician': 'C(session_id) + C(people_id)'}

betas = {}
beta_ses = {}
zs = {}
#%%

#gg = ['session_id', 'bill_id', 'people_id', 'principal_lobbyist', 'beneficial_client']
#gg_mp = ['session_id', 'bill_id', 'principal_lobbyist', 'beneficial_client']

gg = ['session_id', 'bill_id', 'people_id']
gg_mp = ['session_id', 'bill_id']

for model_key in models_dict.keys():
    
    for FE_key in FE_dict.keys():
        
        dc_df = pd.DataFrame()
        
        dc_df['shares'] = reg_df_mod.groupby(by = gg).lobbyist_client.nunique()
        dc_df = dc_df.reset_index();
        
        dc_df = dc_df.merge(dc_df.groupby(gg_mp).shares.sum().reset_index(), how = 'left', on = gg_mp, suffixes = ('', '_within_bill'))
        
        dc_df = dc_df.merge(reg_df_mod.drop_duplicates(gg)[['session_id', 'bill_id', 'people_id', 'sponsor_type_id', 'principal_lobbyist', 'lobbyist_client', 'beneficial_client', 'politician_lobbyist', 'politician_client']], on = gg, how='left')
        dc_df['shares'] = dc_df.shares/(dc_df.shares_within_bill + 1e-6)
        #dc_df.loc[dc_df.shares==1,'shares'] = 0.99
        #dc_df.loc[dc_df.shares==0,'shares'] = 0.01
        dc_df['market_ids'] = dc_df[gg_mp].astype(str).sum(1)
        dc_df['product_ids'] = dc_df.people_id
        dc_df.loc[pd.isna(dc_df.sponsor_type_id), 'sponsor_type_id'] = 0
        dc_df['sponsor_type'] = dc_df.sponsor_type_id
        sponsor_type = dc_df['sponsor_type'].copy()
        #dc_df.loc[sponsor_type==1,'sponsor_type'] = 2
        #dc_df.loc[sponsor_type==2,'sponsor_type'] = 1
        dc_df['is_sponsor_1'] = (dc_df['sponsor_type'] == 1).astype(int)
        dc_df['is_sponsor_2'] = (dc_df['sponsor_type'] == 2).astype(int)
        dc_df['is_sponsor'] = (dc_df['sponsor_type'] != 0).astype(int)
        dc_df['prices'] = 1
        
        logit_formulation = pyblp.Formulation(models_dict[model_key], absorb=FE_dict[FE_key])
        
        problem = pyblp.Problem(logit_formulation, dc_df, add_exogenous=True)
        res = problem.solve()
        
        betas[(model_key, FE_key)] = res.beta[1:]
        beta_ses[(model_key, FE_key)] = res.beta_se[1:]
        zs[(model_key, FE_key)] = res.beta[1:]/res.beta_se[1:]



#%%



#%%

reg_df_mod = reg_df.loc[(pd.isna(reg_df.committee_id) == False) & (pd.isna(reg_df.bill_id) == False),:]
reg_df_mod = reg_df_mod.drop_duplicates(['bill_id', 'committee_id', 'principal_lobbyist', 'beneficial_client'])
reg_df_mod['lobbyist_client'] = reg_df_mod.principal_lobbyist + reg_df_mod.beneficial_client 

#%%


dc_df = pd.DataFrame()

reg_df_mod = reg_df.loc[(pd.isna(reg_df.committee_id) == False) & (pd.isna(reg_df.bill_id) == False),:]
reg_df_mod = reg_df_mod.drop_duplicates(['bill_id', 'committee_id', 'principal_lobbyist', 'beneficial_client'])
reg_df_mod['lobbyist_client'] = reg_df_mod.principal_lobbyist + reg_df_mod.beneficial_client 

dc_df = reg_df_mod.loc[:, ['session_id', 'bill_id', 'committee_id', 'event', 'principal_lobbyist', 'beneficial_client', 'lobbyist_client']]

dc_df = dc_df.groupby(['session_id', 'bill_id', 'committee_id']).lobbyist_client.nunique().reset_index()

dc_df = dc_df.merge(dc_df.groupby(['bill_id']).lobbyist_client.sum().reset_index(), how = 'left', on = 'bill_id', suffixes = ('', '_within_bill'))

dc_df = dc_df.merge(reg_df_mod.drop_duplicates(['bill_id', 'committee_id'])[['bill_id', 'committee_id', 'event']], on = ['bill_id', 'committee_id'], how='left')
dc_df['shares'] = dc_df.lobbyist_client/(dc_df.lobbyist_client_within_bill + 1e-6)
#dc_df.loc[dc_df.shares==1,'shares'] = 0.99
#dc_df.loc[dc_df.shares==0,'shares'] = 0.01
dc_df['market_ids'] = dc_df.bill_id
dc_df['product_ids'] = dc_df.committee_id
dc_df['is_referred'] = pd.isna(dc_df.event) == False
dc_df['prices'] = 1

#%%

logit_formulation = pyblp.Formulation('prices + is_referred + 0', absorb='C(product_ids)')

problem = pyblp.Problem(logit_formulation, dc_df, add_exogenous=True)
res = problem.solve()

#%%

#%%

logit_formulation = pyblp.Formulation('prices + is_referred + 0', absorb='C(session_id)')

problem = pyblp.Problem(logit_formulation, dc_df, add_exogenous=True)
res = problem.solve()

#%%


logit_formulation = pyblp.Formulation('prices + is_referred + 0', absorb='C(product_ids) + C(session_id)')

problem = pyblp.Problem(logit_formulation, dc_df, add_exogenous=True)
res = problem.solve()

#%%