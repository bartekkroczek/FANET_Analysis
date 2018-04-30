import pandas as pd

import os
from os.path import join
from enum import Enum, auto
import yaml
import numpy as np
import time
import random
from tqdm import tqdm

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("VAR")
args = parser.parse_args()

item_by_item = pd.read_csv('item_wise.csv')
os.chdir(
    join('..', '..', '..', 'Dropbox', 'Data', 'FAN_ET', 'Badanie P', '2017-05-06_Badanie_P', 'BadanieP_FAN_ET',
         'Scripts'))

# Localisations of region of interest in data.
ROIS = {
    'P1': [(- 15, 280), (235, 30)],
    'P2': [(- 15, 680), (235, 430)],
    'P3': [(- 15, 1080), (235, 830)],
    'A': [(+ 465, 520), (715, 270)],
    'B': [(+ 955, 520), (1205, 270)],
    'C': [(+1445, 520), (1695, 270)],
    'D': [(+ 465, 870), (715, 630)],
    'E': [(+ 955, 870), (1205, 630)],
    'F': [(+1445, 870), (1695, 630)]
}

ROIS_ORDER = ['P1', 'P2', 'P3', 'A', 'B', 'C', 'D', 'E', 'F']


class CONDITIONS(Enum):
    LOW_WMC = auto()
    MED_WMC = auto()
    HIGH_WMC = auto()
    FULL = auto()
    CORR = auto()
    ERR = auto()
    TIME_SHORT = auto()
    TIME_MED = auto()
    TIME_LONG = auto()
    LEV_EASY = auto()
    LEV_MED = auto()
    LEV_HARD = auto()


DEBUG = False
VAR_CORR = True
CONDITION = {'WMC_LOW': CONDITIONS.LOW_WMC,
             'WMC_MED': CONDITIONS.MED_WMC,
             'WMC_HIGH': CONDITIONS.HIGH_WMC,
             'FULL': CONDITIONS.FULL,
             'CORR': CONDITIONS.CORR,
             'ERR': CONDITIONS.ERR,
             'TIME_SHORT': CONDITIONS.TIME_SHORT,
             'TIME_MED': CONDITIONS.TIME_MED,
             'TIME_LONG': CONDITIONS.TIME_LONG,
             'LEV_EASY': CONDITIONS.LEV_EASY,
             'LEV_MED': CONDITIONS.LEV_MED,
             'LEV_HARD': CONDITIONS.LEV_HARD
}[args.VAR]


def where_in_list(where, what):
    """
    Func that determine position of ROI in raw data yaml list.
    :param where: List of dicts, from experiment description yaml
    :param what: Name of ROI to find, eg. A, B, C, P1
    :return: Pos in list
    """
    bound = len(where)
    tmp = map(lambda x: x['name'] == what, where)
    return [idx for idx, _ in zip(range(bound), tmp) if _]


def in_roi(df, roi):
    """Assume that RECTANGLED roi is defined as list [a, b] where:
        a - coordinate of upper left corner (x1, y1)
        b - coordinate of lower right corner (x2, y2)
        x, y - point in space
    """
    x, y = df.axp, df.ayp
    [(x1, y1), (x2, y2)] = roi
    return (x > x1) & (x < x2) & (y < y1) & (y > y2)


LAB_TO_LEV = {'TRAINING': '[1, 2, 3, 4, 5, 6]',
              'EASY': '[1, 4, 4, 5, 5, 6]',
              'MEDIUM': '[1, 3, 3, 4, 4, 6]',
              'HARD': '[1, 2, 2, 3, 3, 6]'}

LEV_TO_LAB = {v: k for k, v in LAB_TO_LEV.items()}

RESULTS = list()
TRENING_TRIALS = [1, 2, 3]

# LOAD DATA
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

# split participants on order of Working memory capacity category.

low_wmc_band = ID_GF_WMC.WMC.quantile(1 / 3)
high_wmc_band = ID_GF_WMC.WMC.quantile(2 / 3)

low_wmc = ID_GF_WMC[(ID_GF_WMC.WMC.between(-20, low_wmc_band))].PART_ID.tolist()
med_wmc = ID_GF_WMC[(ID_GF_WMC.WMC.between(low_wmc_band, high_wmc_band))].PART_ID.tolist()
high_wmc = ID_GF_WMC[(ID_GF_WMC.WMC.between(high_wmc_band, 20))].PART_ID.tolist()

Lmin = 0
Lmax = 120

Kx = list()

res = [list() for _ in range(Lmin, Lmax)]
FOx = [list() for _ in range(Lmin, Lmax)]
RMx = [list() for _ in range(Lmin, Lmax)]
no_fix_in_sec = 0

if DEBUG:
    # sacc_files = [x for x in sacc_files if '25F' in x]
    sacc_files = [random.choice(sacc_files)]

if VAR_CORR:
    import pickle
    fo_corr = pickle.load(open('FO45.pickle', 'rb'))
    rm_corr = pickle.load(open('RM45.pickle', 'rb'))

with tqdm(total=len(sacc_files)) as pbar:
    for part_id in sacc_files:  # for each participant
        pbar.set_postfix(file=part_id)
        pbar.update(1)

        # Load Data
        part_id = part_id.split('_')[0]

        sacc_data = pd.read_csv(os.path.join(SACC_FOLDER, part_id + '_sacc.csv')).drop('Unnamed: 0', 1)
        sacc_idx = sacc_data.block.unique()

        beh_data = pd.read_csv(os.path.join(BEH_FOLDER, part_id + '_beh.csv'))
        beh_data.set_index(beh_data.index + 1, inplace=True)
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

        if CONDITION == CONDITIONS.LOW_WMC:
            if int(part_id) not in low_wmc:
                continue
        if CONDITION == CONDITIONS.MED_WMC:
            if int(part_id) not in med_wmc:
                continue
        if CONDITION == CONDITIONS.HIGH_WMC:
            if int(part_id) not in high_wmc:
                continue
        # remove broken trials
        index = set(sacc_idx).intersection(fix_idx).intersection(raw_idx)
        # remove training
        index.discard(1)
        index.discard(2)
        index.discard(3)
        index = sorted(list(index))

        # Get categories
        # full_index = beh_data[(beh_data['rt'] > 10.0) & (beh_data['ans_accept'])].index
        # corr_index = beh_data[beh_data['corr_and_accept']].index
        # err_index = beh_data[(((~ beh_data['corr']) | (~beh_data['ans_accept'])) & (beh_data['rt'] > 10.0))].index
        #
        # lev_easy = beh_data[(beh_data.answers.map(LEV_TO_LAB) == 'EASY') & (beh_data['rt'] > 10.0)].index
        # lev_med = beh_data[(beh_data.answers.map(LEV_TO_LAB) == 'MEDIUM') & (beh_data['rt'] > 10.0)].index
        # lev_hard = beh_data[(beh_data.answers.map(LEV_TO_LAB) == 'HARD') & (beh_data['rt'] > 10.0)].index
        #
        # time_short = beh_data[beh_data.rt.between(10.0, 40.0)].index
        # time_med = beh_data[beh_data.rt.between(40.0, 80.0)].index
        # time_long = beh_data[beh_data.rt.between(80.0, 121.0)].index

        # if int(part_id) not in low_wmc:
        #     continue
        for idx in index:  # iterate only over correct trials

            # Due to problems at the level of data acquisition, some indexes ar shifted.
            beh_idx = idx
            # Full shift
            if part_id in ['172', '14', '130', '86', '165', '62', '22', '68', '144']:
                beh_idx = idx - 1

            if part_id in ['142', '83', '150']:
                if idx >= 26:
                    beh_idx = idx - 1

            if part_id in ['50', '144']:
                if idx >= 26:
                    beh_idx = idx - 2

            if part_id in ['52']:
                if idx in range(13, 20):
                    beh_idx = idx - 1
                if idx >= 20:
                    beh_idx = idx - 2

            if idx > 45:
                continue

            raw_item = raw_data[raw_data.block == idx]
            beh_item = beh_data.ix[beh_idx]

            start_stamp = int(raw_item.head(1).time.values[0])
            end_stamp = int(raw_item.tail(1).time.values[0])

            a = (len(range(start_stamp, end_stamp, 1000)))
            if abs(beh_item.rt - a) > 1:
            #     print('ID: {} IDX: {} BEH_IDX: {} RT: {} real: {}'.format(part_id, idx, beh_idx, beh_item.rt, a))
                continue

            sacc_item = sacc_data[sacc_data.block == idx]
            fix_item = fix_data[fix_data.block == idx]
            problem = problems[idx - 1]

            # trials to remove
            if not (10.0 < beh_item.rt < 120.0):
                continue

            if ((end_stamp - start_stamp) / 1000.0) > 120.0:
                print('stamp: {}'.format((end_stamp - start_stamp) / 1000.0))
                continue

            # Select condition
            if CONDITION == CONDITIONS.FULL:
                if not beh_item.ans_accept:
                    continue
            if CONDITION == CONDITIONS.CORR:
                if not (beh_item['corr'] and beh_item['ans_accept']):
                    continue
            if CONDITION == CONDITIONS.ERR:
                if (not beh_item['corr']) or (not beh_item['ans_accept']):
                    continue
            if CONDITION == CONDITIONS.LEV_EASY:
                if LEV_TO_LAB[beh_item.answers] != 'EASY':
                    continue
            if CONDITION == CONDITIONS.LEV_MED:
                if LEV_TO_LAB[beh_item.answers] != 'MEDIUM':
                    continue
            if CONDITION == CONDITIONS.LEV_HARD:
                if LEV_TO_LAB[beh_item.answers] != 'HARD':
                    continue
            if CONDITION == CONDITIONS.TIME_SHORT:
                if not (10.0 < beh_item.rt < 40.0):
                    continue
            if CONDITION == CONDITIONS.TIME_MED:
                if not (40.0 < beh_item.rt < 80.0):
                    continue
            if CONDITION == CONDITIONS.TIME_LONG:
                if not (80.0 < beh_item.rt < 120.0):
                    continue

            Kx.append(beh_item.rt)

            for idx, start in enumerate(range(start_stamp, end_stamp, 1000)):  # iterate over seconds
                stop = start + 1000
                sec = set(range(start, stop))
                fix_in_sec = list()
                cur_fix = fix_item[((start - 15000) < fix_item['stime']) & (fix_item['stime'] < (start + 15000))]

                for fix in cur_fix.iterrows():
                    if set(range(int(fix[1]['stime']), int(fix[1]['etime']))).intersection(sec):
                        fix_in_sec.append(fix)
                if fix_in_sec:
                    longest_fix_in_sec = sorted(fix_in_sec, key=lambda x: -x[1].dur)[0][1]
                    res[idx].append(longest_fix_in_sec)

                    fix_in_pr = any([in_roi(longest_fix_in_sec[['axp', 'ayp']], ROIS[x]) for x in ['P1', 'P2', 'P3']])
                    fix_in_op = any(
                        [in_roi(longest_fix_in_sec[['axp', 'ayp']], ROIS[x]) for x in ['A', 'B', 'C', 'D', 'E', 'F']])

                    if fix_in_pr:
                        if VAR_CORR:
                            FOx[idx].append(0 - fo_corr[int(part_id)][idx])
                            RMx[idx].append(-1 - rm_corr[int(part_id)][idx])
                        else:
                            FOx[idx].append(0)
                            RMx[idx].append(-1)
                    if fix_in_op:
                        if VAR_CORR:
                            FOx[idx].append(1 - fo_corr[int(part_id)][idx])
                        else:
                            FOx[idx].append(1)

                        prob = problem['matrix_info']

                        which_option = np.where(
                            [in_roi(longest_fix_in_sec[['axp', 'ayp']], ROIS[x]) for x in
                             ['A', 'B', 'C', 'D', 'E', 'F']])[
                            0][0]
                        which_option = [x['name'] for x in prob][3 + which_option]

                        denom = np.sum([len(x['elements_changed']) for x in prob[1]['parameters']])
                        counter = [x for x in prob if x['name'] == which_option][0]['parameters']
                        counter = np.sum([len(x['elements_changed']) for x in counter])

                        if which_option == 'D2':  # some magic
                            rs = ((counter - 1) / denom) + 0.02
                        else:
                            rs = counter / denom
                        if VAR_CORR:
                            RMx[idx].append(rs - rm_corr[int(part_id)][idx])
                        else:
                            RMx[idx].append(rs)
                else:
                    no_fix_in_sec += 1

K = list()
Kx = pd.Series(Kx)
for l_bound in range(Lmin, Lmax):
    K.append((Kx >= l_bound).sum())

df = pd.DataFrame()
df['Kx'] = K
df['FOx'] = [sum(x) for x in FOx]
df['FOx_STD'] = [np.std(x) for x in FOx]
df['RMx'] = [sum([a for a in x if a >= 0.0]) for x in RMx]
df['RMx_STD'] = [np.std([a for a in x if a >= 0.0]) for x in RMx]
df['RMk'] = [sum([1 for a in x if a >= 0.0]) for x in RMx]
df['PROP_FOx'] = df.FOx / df.Kx
df['AVG_RMx'] = df.RMx / df.RMk

print('No fix in sec:{}'.format(no_fix_in_sec))

dat = time.localtime()
filename = '{}_{}_{}_{}:{}'.format(dat.tm_year, dat.tm_mon, dat.tm_mday, dat.tm_hour, dat.tm_min)

cond = {
    CONDITIONS.FULL: 'full',
    CONDITIONS.CORR: 'corr',
    CONDITIONS.ERR: 'err',
    CONDITIONS.LOW_WMC: 'wmc_low',
    CONDITIONS.MED_WMC: 'wmc_med',
    CONDITIONS.HIGH_WMC: 'wmc_high',
    CONDITIONS.TIME_SHORT: 'time_short',
    CONDITIONS.TIME_MED: 'time_med',
    CONDITIONS.TIME_LONG: 'time_long',
    CONDITIONS.LEV_EASY: 'lev_easy',
    CONDITIONS.LEV_MED: 'lev_med',
    CONDITIONS.LEV_HARD: 'lev_hard'
}
corr = 'VAR_CORR' if VAR_CORR else 'NO_CORR'

df.to_csv(join('results', 'dynamics_45_' + cond[CONDITION] + '_' + corr + '_' + filename + '.csv'))
