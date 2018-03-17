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
from tqdm import tqdm

os.chdir(
    join('..', '..', '..', 'Dropbox', 'Data', 'FAN_ET', 'Badanie P', '2017-05-06_Badanie_P', 'BadanieP_FAN_ET', 'Scripts'))
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

with tqdm(total=len(sacc_files)) as pbar:
    for part_id in sacc_files: # for each participant
        pbar.set_postfix(file=part_id)
        pbar.update(1)

        part_id = part_id.split('_')[0]

        sacc_data = pd.read_csv(os.path.join(SACC_FOLDER, part_id + '_sacc.csv')).drop('Unnamed: 0', 1)
        sacc_idx = sacc_data.block.unique()

        beh_data = pd.read_csv(os.path.join(BEH_FOLDER, part_id + '_beh.csv'))

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

        avg_corr = beh_data[beh_data.exp == 'experiment']['corr'].mean()

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

        for idx in index:
            part_result = collections.OrderedDict()
            sacc_item = sacc_data[sacc_data.block == idx]
            fix_item = fix_data[fix_data.block == idx]
            raw_item = raw_data[raw_data.block == idx]
            beh_item = beh_data.ix[idx - 1]
            problem = problems[idx - 1]

            if beh_item.rt == -1:  # no ans selected
                continue

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

            #rs
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