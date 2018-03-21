#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  4 14:09:38 2018

@author: bkroczek
"""

import pandas as pd
import os
from os.path import join
import yaml
import numpy as np
import collections
import time
from tqdm import tqdm

os.chdir(
    join('..', '..', '..', 'Dropbox', 'Data', 'FAN_ET', 'Badanie P', '2017-05-06_Badanie_P', 'BadanieP_FAN_ET',
         'Scripts'))
# %%
ROIS = {
    'P1': [(-  15, 280), (235, 30)],
    'P2': [(-  15, 680), (235, 430)],
    'P3': [(-  15, 1080), (235, 830)],
    'A': [(+ 465, 520), (715, 270)],
    'B': [(+ 955, 520), (1205, 270)],
    'C': [(+1445, 520), (1695, 270)],
    'D': [(+ 465, 870), (715, 630)],
    'E': [(+ 955, 870), (1205, 630)],
    'F': [(+1445, 870), (1695, 630)]
}

ROIS_ORDER = ['P1', 'P2', 'P3', 'A', 'B', 'C', 'D', 'E', 'F']


def where_in_list(where, what):
    bound = len(where)
    tmp = map(lambda x: x['name'] == what, where)
    return [idx for idx, _ in zip(range(bound), tmp) if _]


def in_roi(df, roi):
    """Assume that RECTANGLED roi is defined as list [a, b] where:
        a - coordinate of upper left corner (x1, y1)
        b - coordinate of lower right corner (x2, y2)
        x, y - point in space
    """
    x, y = df.columns
    [(x1, y1), (x2, y2)] = roi
    return (df[x] > x1) & (df[x] < x2) & (df[y] < y1) & (df[y] > y2)


LAB_TO_LEV = {'TRAINING': '[1, 2, 3, 4, 5, 6]',
              'EASY': '[1, 4, 4, 5, 5, 6]',
              'MEDIUM': '[1, 3, 3, 4, 4, 6]',
              'HARD': '[1, 2, 2, 3, 3, 6]'}
LEV_TO_LAB = {v: k for k, v in LAB_TO_LEV.items()}

RESULTS = list()
TRENING_TRIALS = [1, 2, 3]
# %%
SACC_FOLDER = os.path.join('..', 'Dane trackingowe', 'sacc')
BEH_FOLDER = os.path.join('..', 'results', 'beh')
FIX_FOLDER = os.path.join('..', 'Dane trackingowe', 'fix')
RAW_FOLDER = os.path.join('..', 'Dane trackingowe', 'raw')
YAML_FOLDER = os.path.join('..', 'results', 'yaml')

sacc_files = os.listdir(SACC_FOLDER)
sacc_files = [x for x in sacc_files if x.endswith('.csv')]
beh_files = os.listdir(BEH_FOLDER)
fix_files = os.listdir(FIX_FOLDER)
raw_files = os.listdir(RAW_FOLDER)
yaml_files = os.listdir(YAML_FOLDER)

ID_GF_WMC = pd.read_csv(join('..', 'results', 'ID_GF_WMC.csv'))
import random

# sacc_files = [random.choice(sacc_files)]
with tqdm(total=len(sacc_files)) as pbar:
    for part_id in sacc_files:  # for each participant
        pbar.set_postfix(file=part_id)
        pbar.update(1)

        part_id = part_id.split('_')[0]

        sacc_data = pd.read_csv(os.path.join(SACC_FOLDER, part_id + '_sacc.csv')).drop('Unnamed: 0', 1)
        sacc_idx = sacc_data.block.unique()

        beh_data = pd.read_csv(os.path.join(BEH_FOLDER, part_id + '_beh.csv'))
        beh_data['corr_and_accept'] = (beh_data['corr'] & beh_data['ans_accept'] & (beh_data['rt'] > 10.0))

        fix_data = pd.read_csv(os.path.join(FIX_FOLDER, part_id + '_fix.csv')).drop('Unnamed: 0', 1)
        fix_idx = fix_data.block.unique()
        if part_id in ['12MALE21', '14FEMALE19', '62FEMALE39', '83MALE27', '130MALE18', '142FEMALE29',
                       '165FEMALE20']:  # no Unnamed column
            raw_data = pd.read_csv(os.path.join(RAW_FOLDER, part_id + '_raw.csv'))
        else:
            raw_data = pd.read_csv(os.path.join(RAW_FOLDER, part_id + '_raw.csv')).drop('Unnamed: 0', 1)
        raw_idx = raw_data.block.unique()

        yaml_data = yaml.load(open(os.path.join(YAML_FOLDER, part_id + '.yaml'), 'r'))
        problems = yaml_data['list_of_blocks'][0]['experiment_elements'][2:]
        problems += yaml_data['list_of_blocks'][1]['experiment_elements'][1:]
        problems += yaml_data['list_of_blocks'][2]['experiment_elements'][1:]
        part_id = part_id.split('F')[0] if 'FEMALE' in part_id else part_id.split('M')[0]

        index_data = pd.read_csv(os.path.join('..', 'results', 'FAN_ET_aggr.csv'))
        index_data = index_data[index_data.Part_id == int(part_id)]

        gf_wmc = ID_GF_WMC[ID_GF_WMC.PART_ID == int(part_id)]

        # remove broken trials
        index = set(sacc_idx).intersection(fix_idx).intersection(raw_idx)
        # remove training
        index.discard(1)
        index.discard(2)
        index.discard(3)
        index = sorted(list(index))
        index = [x for x in index if x <= 45]
        # %%

        avg_corr = beh_data[(beh_data.exp == 'experiment')]['corr_and_accept'].mean()

        trs = list()  # total
        ers = list()  # error
        for idx in index:  # iterate over index, because some items are missed, due to choosed_option == -1
            choosed_option = beh_data.ix[idx - 1]['choosed_option']
            problem = problems[idx - 1]['matrix_info']
            err = not beh_data.ix[idx - 1]['corr']
            if choosed_option == '-1':
                continue

            denom = np.sum([len(x['elements_changed']) for x in problem[1]['parameters']])
            counter = [x for x in problem if x['name'] == choosed_option][0]['parameters']
            counter = np.sum([len(x['elements_changed']) for x in counter])

            if choosed_option == 'D2':  # some magic
                rs = ((counter - 1) / denom) + 0.02
            else:
                rs = counter / denom

            trs.append(rs)
            if err:
                ers.append(rs)

        avg_trs = np.mean(trs)
        avg_ers = np.mean(ers)

        gf = gf_wmc['GF'].values[0]
        wmc = gf_wmc['WMC'].values[0]

        for idx in index:  # for each item
            part_result = collections.OrderedDict()
            sacc_item = sacc_data[sacc_data.block == idx]
            fix_item = fix_data[fix_data.block == idx]
            raw_item = raw_data[raw_data.block == idx]
            beh_item = beh_data.ix[idx - 1]
            problem = problems[idx - 1]

            # common for all items
            part_result['ID'] = part_id
            part_result['AVG_CORR'] = avg_corr
            part_result['AVG_TRS'] = avg_trs
            part_result['AVG_ERS'] = avg_ers
            part_result['GF'] = gf
            part_result['WMC'] = wmc

            # unique for all items
            part_result['KOL'] = idx
            part_result['WAR'] = LEV_TO_LAB[beh_item.answers]
            part_result['WYB'] = beh_item.choosed_option
            part_result['RESP'] = 'COR' if beh_item.choosed_option == 'D1' else 'ERR'
            part_result['LAT'] = int(round(beh_item.rt))

            # rs
            choosed_option = beh_item['choosed_option']
            prob = problem['matrix_info']

            if choosed_option == '-1':
                rs = -1
            else:
                denom = np.sum([len(x['elements_changed']) for x in prob[1]['parameters']])
                counter = [x for x in prob if x['name'] == choosed_option][0]['parameters']
                counter = np.sum([len(x['elements_changed']) for x in counter])

                if choosed_option == 'D2':  # some magic
                    rs = ((counter - 1) / denom) + 0.02
                else:
                    rs = counter / denom
            part_result['RS'] = rs
            part_result['PUP_SIZE'] = raw_item.ps.mean()

            # NT_PR (podzielone przez 3, tzn. wartość per opcja)
            # NT_COR
            # NT_SE (podzielone przez 2, tzn. wartość per opcja)
            # NT_BE (podzielone przez 2, tzn. wartość per opcja)
            # NT_CON
            level = LEV_TO_LAB[beh_item.answers]

            # high-lebel rois definitions

            cor_roi = ROIS_ORDER[where_in_list(problem['matrix_info'], 'D1')[0]]

            se_roi = {'EASY': 'D4', 'MEDIUM': 'D3', 'HARD': 'D2'}[level]
            be_roi = {'EASY': 'D5', 'MEDIUM': 'D4', 'HARD': 'D3'}[level]

            se_roi = [ROIS_ORDER[x] for x in where_in_list(problem['matrix_info'], se_roi)]
            be_roi = [ROIS_ORDER[x] for x in where_in_list(problem['matrix_info'], be_roi)]

            con_roi = ROIS_ORDER[where_in_list(problem['matrix_info'], 'D6')[0]]

            # saccades
            sacc_ends_in_pr = pd.concat([in_roi(sacc_item[['exp', 'eyp']], ROIS[roi]) for roi in ['P1', 'P2', 'P3']],
                                        axis=1).any(axis=1)
            sacc_ends_in_corr = in_roi(sacc_item[['exp', 'eyp']], ROIS[cor_roi])
            sacc_ends_in_se = pd.concat([in_roi(sacc_item[['exp', 'eyp']], ROIS[roi]) for roi in se_roi], axis=1).any(
                axis=1)
            sacc_ends_in_be = pd.concat([in_roi(sacc_item[['exp', 'eyp']], ROIS[roi]) for roi in be_roi], axis=1).any(
                axis=1)
            sacc_ends_in_con = in_roi(sacc_item[['exp', 'eyp']], ROIS[con_roi])

            part_result['NT_PR'] = sacc_ends_in_pr.sum() / 3.0
            part_result['NT_COR'] = sacc_ends_in_corr.sum()
            part_result['NT_SE'] = sacc_ends_in_se.sum() / 2.0
            part_result['NT_BE'] = sacc_ends_in_be.sum() / 2.0
            part_result['NT_CON'] = sacc_ends_in_con.sum()

            # fixations
            fix_in_pr = pd.concat([in_roi(fix_item[['axp', 'ayp']], ROIS[x]) for x in ['P1', 'P2', 'P3']], axis=1).any(
                axis=1)
            fix_in_corr = in_roi(fix_item[['axp', 'ayp']], ROIS[cor_roi])
            fix_in_se = pd.concat([in_roi(fix_item[['axp', 'ayp']], ROIS[roi]) for roi in se_roi], axis=1).any(axis=1)
            fix_in_be = pd.concat([in_roi(fix_item[['axp', 'ayp']], ROIS[roi]) for roi in be_roi], axis=1).any(axis=1)
            fix_in_con = in_roi(fix_item[['axp', 'ayp']], ROIS[con_roi])

            part_result['FIX_PR'] = fix_item[fix_in_pr].dur.sum() / 1000.0
            part_result['FIX_COR'] = fix_item[fix_in_corr].dur.sum() / 1000.0
            part_result['FIX_SE'] = fix_item[fix_in_se].dur.sum() / 1000.0
            part_result['FIX_BE'] = fix_item[fix_in_be].dur.sum() / 1000.0
            part_result['FIX_CON'] = fix_item[fix_in_con].dur.sum() / 1000.0

            part_result['DUR_PR'] = fix_item[fix_in_pr].dur.mean() / 1000.0
            part_result['DUR_COR'] = fix_item[fix_in_corr].dur.mean() / 1000.0
            part_result['DUR_SE'] = fix_item[fix_in_se].dur.mean() / 1000.0
            part_result['DUR_BE'] = fix_item[fix_in_be].dur.mean() / 1000.0
            part_result['DUR_CON'] = fix_item[fix_in_con].dur.mean() / 1000.0

            RESULTS.append(part_result)

res = pd.DataFrame(RESULTS)
res = res.fillna(0)
dat = time.localtime()
filename = '{}_{}_{}_{}:{}'.format(dat.tm_year, dat.tm_mon, dat.tm_mday, dat.tm_hour, dat.tm_min)
res.to_csv(join('results', 'item_wise_' + filename + '.csv'))
res.to_excel(join('results', 'item_wise_' + filename + '.xlsx'))
