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

Lmin = 0
Lmax = 120

res = [list() for _ in range(Lmin, Lmax)]

sacc_files = [random.choice(sacc_files)]
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

        # remove broken trials
        index = set(sacc_idx).intersection(fix_idx).intersection(raw_idx)
        # remove training
        index.discard(1)
        index.discard(2)
        index.discard(3)
        index = sorted(list(index))
        index = [x for x in index if x <= 45]
        # %%

        for idx in index:
            sacc_item = sacc_data[sacc_data.block == idx]
            fix_item = fix_data[fix_data.block == idx]
            raw_item = raw_data[raw_data.block == idx]
            beh_item = beh_data.ix[idx - 1]
            problem = problems[idx - 1]

            start_stamp = int(raw_item.head(1).time.values[0])
            end_stamp = int(raw_item.tail(1).time.values[0])

            for start in range(start_stamp, end_stamp, 1000):
                stop = start + 1000
