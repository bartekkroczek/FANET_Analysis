import argparse
import os
import random
import time
from enum import Enum, auto
from os.path import join

import numpy as np
import pandas as pd
import yaml
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("VAR")
parser.add_argument('INT')
args = parser.parse_args()
os.chdir(
    join('..', '..', '..', '..', 'Dropbox', 'Data', 'FAN_ET', 'Badanie P', '2017-05-06_Badanie_P', 'BadanieP_FAN_ET',
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

ANS_TYPE_DICT = {'EASY': {'D6': 'CON', 'D5': 'BE', 'D4': 'SE', 'D1': 'CORR'},
                 'MEDIUM': {'D6': 'CON', 'D4': 'BE', 'D3': 'SE', 'D1': 'CORR'},
                 'HARD': {'D6': 'CON', 'D3': 'BE', 'D2': 'SE', 'D1': 'CORR'}}


class CONDITIONS(Enum):
    LOW_WMC_FULL = auto()
    MED_WMC_FULL = auto()
    HIGH_WMC_FULL = auto()
    LOW_WMC_CORR = auto()
    LOW_WMC_ERR = auto()
    MED_WMC_CORR = auto()
    MED_WMC_ERR = auto()
    HIGH_WMC_CORR = auto()
    HIGH_WMC_ERR = auto()

    FULL = auto()
    CORR = auto()
    ERR = auto()
    LEV_EASY_FULL = auto()
    LEV_MED_FULL = auto()
    LEV_HARD_FULL = auto()

    LEV_EASY_CORR = auto()
    LEV_MED_CORR = auto()
    LEV_HARD_CORR = auto()
    LEV_EASY_ERR = auto()
    LEV_MED_ERR = auto()
    LEV_HARD_ERR = auto()
    WMC_LOW_LEV_EASY = auto()
    WMC_LOW_LEV_MED = auto()
    WMC_LOW_LEV_HARD = auto()
    WMC_MED_LEV_EASY = auto()
    WMC_MED_LEV_MED = auto()
    WMC_MED_LEV_HARD = auto()
    WMC_HIGH_LEV_EASY = auto()
    WMC_HIGH_LEV_MED = auto()
    WMC_HIGH_LEV_HARD = auto()


class INTERVAL(object):
    I0 = [0, 120]
    I1 = [40, 50]
    I2 = [50, 60]
    I3 = [60, 70]
    I4 = [70, 80]
    I5 = [80, 90]
    I6 = [90, 100]
    I7 = [100, 110]


inter = {
    'I0': INTERVAL.I0,
    'I1': INTERVAL.I1,
    'I2': INTERVAL.I2,
    'I3': INTERVAL.I3,
    'I4': INTERVAL.I4,
    'I5': INTERVAL.I5,
    'I6': INTERVAL.I6,
    'I7': INTERVAL.I7
}[args.INT]

DEBUG = False
CONDITION = {'WMC_LOW_FULL': CONDITIONS.LOW_WMC_FULL,
             'WMC_MED_FULL': CONDITIONS.MED_WMC_FULL,
             'WMC_HIGH_FULL': CONDITIONS.HIGH_WMC_FULL,
             'WMC_LOW_CORR': CONDITIONS.LOW_WMC_CORR,
             'WMC_LOW_ERR': CONDITIONS.LOW_WMC_ERR,
             'WMC_MED_CORR': CONDITIONS.MED_WMC_CORR,
             'WMC_MED_ERR': CONDITIONS.MED_WMC_ERR,
             'WMC_HIGH_CORR': CONDITIONS.HIGH_WMC_CORR,
             'WMC_HIGH_ERR': CONDITIONS.HIGH_WMC_ERR,

             'FULL': CONDITIONS.FULL,
             'CORR': CONDITIONS.CORR,
             'ERR': CONDITIONS.ERR,

             'LEV_EASY_FULL': CONDITIONS.LEV_EASY_FULL,
             'LEV_MED_FULL': CONDITIONS.LEV_MED_FULL,
             'LEV_HARD_FULL': CONDITIONS.LEV_HARD_FULL,
             'LEV_EASY_CORR': CONDITIONS.LEV_EASY_CORR,
             'LEV_MED_CORR': CONDITIONS.LEV_MED_CORR,
             'LEV_HARD_CORR': CONDITIONS.LEV_HARD_CORR,
             "LEV_EASY_ERR": CONDITIONS.LEV_EASY_ERR,
             'LEV_MED_ERR': CONDITIONS.LEV_MED_ERR,
             'LEV_HARD_ERR': CONDITIONS.LEV_HARD_ERR,

             'WMC_LOW_LEV_EASY': CONDITIONS.WMC_LOW_LEV_EASY,
             'WMC_LOW_LEV_MED': CONDITIONS.WMC_LOW_LEV_MED,
             'WMC_LOW_LEV_HARD': CONDITIONS.WMC_LOW_LEV_HARD,
             'WMC_MED_LEV_EASY': CONDITIONS.WMC_MED_LEV_EASY,
             'WMC_MED_LEV_MED': CONDITIONS.WMC_MED_LEV_MED,
             'WMC_MED_LEV_HARD': CONDITIONS.WMC_MED_LEV_HARD,
             'WMC_HIGH_LEV_EASY': CONDITIONS.WMC_HIGH_LEV_EASY,
             'WMC_HIGH_LEV_MED': CONDITIONS.WMC_HIGH_LEV_MED,
             'WMC_HIGH_LEV_HARD': CONDITIONS.WMC_HIGH_LEV_HARD
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
WINDOW_TIME = 5

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
Lmax = 120 // WINDOW_TIME

Kx = list()

res = [list() for _ in range(Lmin, Lmax)]
FOx = [list() for _ in range(Lmin, Lmax)]
RMx = [list() for _ in range(Lmin, Lmax)]
FOx_aggregated = [list() for _ in range(Lmin, Lmax)]
ANS_TYPE_RES = {'CORR': [list() for _ in range(Lmin, Lmax)], 'BE': [list() for _ in range(Lmin, Lmax)],
                'SE': [list() for _ in range(Lmin, Lmax)], 'CON': [list() for _ in range(Lmin, Lmax)]}

PUP_SIZE = [list() for _ in range(Lmin, Lmax)]
no_fix_in_sec = 0

if DEBUG:
    # sacc_files = [x for x in sacc_files if '25F' in x]
    sacc_files = [random.choice(sacc_files)]

with tqdm(total=len(sacc_files)) as pbar:
    for part_id in sacc_files:  # for each participant
        pbar.set_postfix(file=part_id)
        pbar.update(1)
        curr_fix_in_op = [list() for _ in range(Lmin, Lmax)]

        # Load Data
        part_id = part_id.split('_')[0]

        sacc_data = pd.read_csv(os.path.join(SACC_FOLDER, part_id + '_sacc.csv')).drop('Unnamed: 0', 1)
        sacc_idx = sacc_data.block.unique()

        beh_data = pd.read_csv(os.path.join(BEH_FOLDER, part_id + '_beh.csv'))
        beh_data.set_index(beh_data.index + 1, inplace=True)

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

        if CONDITION in [CONDITIONS.LOW_WMC_FULL, CONDITIONS.WMC_LOW_LEV_EASY, CONDITIONS.WMC_LOW_LEV_MED,
                         CONDITIONS.WMC_LOW_LEV_HARD, CONDITIONS.LOW_WMC_CORR, CONDITIONS.LOW_WMC_ERR]:
            if int(part_id) not in low_wmc:
                continue
        if CONDITION in [CONDITIONS.MED_WMC_FULL, CONDITIONS.WMC_MED_LEV_EASY, CONDITIONS.WMC_MED_LEV_MED,
                         CONDITIONS.WMC_MED_LEV_HARD, CONDITIONS.MED_WMC_CORR, CONDITIONS.MED_WMC_ERR]:
            if int(part_id) not in med_wmc:
                continue
        if CONDITION in [CONDITIONS.HIGH_WMC_FULL, CONDITIONS.WMC_HIGH_LEV_EASY, CONDITIONS.WMC_HIGH_LEV_MED,
                         CONDITIONS.WMC_HIGH_LEV_HARD, CONDITIONS.HIGH_WMC_CORR, CONDITIONS.HIGH_WMC_ERR]:
            if int(part_id) not in high_wmc:
                continue

        if int(part_id) in low_wmc:
            print('LOW WMC')
        if int(part_id) in med_wmc:
            print('MED_WMC')
        if int(part_id) in high_wmc:
            print('HIGH_WMC')

        # remove broken trials
        index = set(sacc_idx).intersection(fix_idx).intersection(raw_idx)
        # remove training
        index.discard(1)
        index.discard(2)
        index.discard(3)
        index = sorted(list(index))

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
            if abs(beh_item.rt - a) > 1:  # big difference between beh and real time => broken trial
                #     print('ID: {} IDX: {} BEH_IDX: {} RT: {} real: {}'.format(part_id, idx, beh_idx, beh_item.rt, a))
                continue

            sacc_item = sacc_data[sacc_data.block == idx]
            fix_item = fix_data[fix_data.block == idx]
            problem = problems[idx - 1]

            # trials to remove
            if not (10.0 < beh_item.rt < 120.0):
                print('beh rt: {}'.format(beh_item.rt))
                continue

            if not (10.0 < ((end_stamp - start_stamp) / 1000.0) < 120.0):
                print('stamp: {}'.format((end_stamp - start_stamp) / 1000.0))
                continue

            if not (inter[0] < beh_item.rt < inter[1]):
                continue
            # Select condition
            if CONDITION in [CONDITIONS.FULL, CONDITIONS.LOW_WMC_FULL, CONDITIONS.MED_WMC_FULL,
                             CONDITIONS.HIGH_WMC_FULL]:
                if not beh_item.ans_accept:
                    continue
            if CONDITION in [CONDITIONS.CORR, CONDITIONS.LOW_WMC_CORR, CONDITIONS.MED_WMC_CORR,
                             CONDITIONS.HIGH_WMC_CORR]:
                if (not beh_item['corr']) or (not beh_item['ans_accept']):
                    continue
            if CONDITION in [CONDITIONS.ERR, CONDITIONS.LOW_WMC_ERR, CONDITIONS.MED_WMC_ERR, CONDITIONS.HIGH_WMC_ERR]:
                if beh_item['corr'] and beh_item['ans_accept']:
                    continue
            if CONDITION in [CONDITIONS.LEV_EASY_FULL, CONDITIONS.WMC_LOW_LEV_EASY, CONDITIONS.WMC_MED_LEV_EASY,
                             CONDITIONS.WMC_HIGH_LEV_EASY]:
                if LEV_TO_LAB[beh_item.answers] != 'EASY':
                    continue
            if CONDITION in [CONDITIONS.LEV_MED_FULL, CONDITIONS.WMC_LOW_LEV_MED, CONDITIONS.WMC_MED_LEV_MED,
                             CONDITIONS.WMC_HIGH_LEV_MED]:
                if LEV_TO_LAB[beh_item.answers] != 'MEDIUM':
                    continue
            if CONDITION in [CONDITIONS.LEV_HARD_FULL, CONDITIONS.WMC_LOW_LEV_HARD, CONDITIONS.WMC_MED_LEV_HARD,
                             CONDITIONS.WMC_HIGH_LEV_HARD]:
                if LEV_TO_LAB[beh_item.answers] != 'HARD':
                    continue

            if CONDITION == CONDITIONS.LEV_EASY_CORR:
                if (not beh_item['corr']) or (not beh_item['ans_accept']):
                    continue
                if LEV_TO_LAB[beh_item.answers] != 'EASY':
                    continue

            if CONDITION == CONDITIONS.LEV_MED_CORR:
                if (not beh_item['corr']) or (not beh_item['ans_accept']):
                    continue
                if LEV_TO_LAB[beh_item.answers] != 'MEDIUM':
                    continue

            if CONDITION == CONDITIONS.LEV_HARD_CORR:
                if (not beh_item['corr']) or (not beh_item['ans_accept']):
                    continue
                if LEV_TO_LAB[beh_item.answers] != 'HARD':
                    continue

            if CONDITION == CONDITIONS.LEV_EASY_ERR:
                if beh_item['corr'] and beh_item['ans_accept']:
                    continue
                if LEV_TO_LAB[beh_item.answers] != 'EASY':
                    continue

            if CONDITION == CONDITIONS.LEV_MED_ERR:
                if beh_item['corr'] and beh_item['ans_accept']:
                    continue
                if LEV_TO_LAB[beh_item.answers] != 'MEDIUM':
                    continue

            if CONDITION == CONDITIONS.LEV_HARD_ERR:
                if beh_item['corr'] and beh_item['ans_accept']:
                    continue
                if LEV_TO_LAB[beh_item.answers] != 'HARD':
                    continue

            print("CORR: {} ACCEPTED: {} LEVEL: {} ".format(beh_item['corr'], beh_item['ans_accept'],
                                                            LEV_TO_LAB[beh_item.answers]))
            Kx.append(beh_item.rt)

            for idx, start in enumerate(range(start_stamp, end_stamp, WINDOW_TIME * 1000)):  # iterate over windows
                stop = start + (WINDOW_TIME * 1000)
                window = set(range(start, stop))
                fix_in_window = list()
                cur_fix = fix_item[((start - 15000) < fix_item['stime']) & (fix_item['stime'] < (start + 15000))]

                for fix in cur_fix.iterrows():
                    i = (set(range(int(fix[1]['stime']), int(fix[1]['etime']))).intersection(window))
                    if i:
                        # r = (set(range(int(fix[1]['stime']), int(fix[1]['etime']))))
                        # w = (window)
                        dur = len(i) + 1
                        # print('R: {} W:{} I:{} FIX DUR:{}'.format(len(r), len(w), len(i), fix[1].dur))
                        fix_in_window.append((fix, dur))

                RS_avg_counter = 0
                RS_avg_denom = 0
                curr_fix_in_op = list()
                for (fix, dur) in fix_in_window:
                    # fix_in_pr = any([in_roi(fix[['axp', 'ayp']], ROIS[x]) for x in ['P1', 'P2', 'P3']])
                    fix_in_op = any([in_roi(fix[1][['axp', 'ayp']], ROIS[x]) for x in ['A', 'B', 'C', 'D', 'E', 'F']])

                    # if fix_in_pr:
                    #     FOx[idx].append(0)
                    #     RMx[idx].append(-1)
                    if fix_in_op:
                        fix_dur = dur / 1000.0
                        pup_size = fix[1].aps

                        FOx[idx].append(fix_dur)
                        curr_fix_in_op.append(fix_dur)
                        PUP_SIZE[idx].append(pup_size)
                        prob = problem['matrix_info']

                        which_option = np.where([in_roi(fix[1][['axp', 'ayp']], ROIS[x]) for x in ['A', 'B', 'C', 'D', 'E', 'F']])[0][0]
                        which_option = [x['name'] for x in prob][3 + which_option]
                        denom = np.sum([len(x['elements_changed']) for x in prob[1]['parameters']])
                        counter = [x for x in prob if x['name'] == which_option][0]['parameters']
                        counter = np.sum([len(x['elements_changed']) for x in counter])
                        op_type = ANS_TYPE_DICT.get(LEV_TO_LAB[beh_item.answers], dict()).get(which_option, None)
                        if op_type:
                            ANS_TYPE_RES[op_type][idx].append(fix_dur)
                        if which_option == 'D2':  # some magic
                            rs = ((counter - 1) / denom) + 0.02
                        else:
                            rs = counter / denom
                        RS_avg_counter += (fix_dur * rs)
                        RS_avg_denom += fix_dur
                if RS_avg_denom:
                    RMx[idx].append(RS_avg_counter / RS_avg_denom)
                FOx_aggregated[idx].append(sum(curr_fix_in_op) / (WINDOW_TIME))
K = list()
Kx = pd.Series(Kx)
for l_bound in range(Lmin, Lmax):
    K.append((Kx >= (WINDOW_TIME * l_bound)).sum())

df = pd.DataFrame()
df['Kx'] = K
df['FOx'] = [sum(x) for x in FOx]
df['FOX_aggregated'] = [np.mean([a for a in x if a > 0.0001]) for x in FOx_aggregated]
df['FOx_mean'] = [np.mean(x) for x in FOx]
df['FOx_STD'] = [np.std(x) for x in FOx]
df['PSOx'] = [sum(x) for x in PUP_SIZE]
df['PSOx_STD'] = [np.std(x) for x in PUP_SIZE]
df['RMx'] = [sum([a for a in x if a >= 0.0]) for x in RMx]
df['RMx_STD'] = [np.std([a for a in x if a >= 0.0]) for x in RMx]
df['RMk'] = [sum([1 for a in x if a >= 0.0]) for x in RMx]
df['PROP_FOx'] = df.FOx / df.Kx
df['AVG_RMx'] = df.RMx / df.RMk
df['PROP_PSOx'] = df.PSOx / df.Kx

dat = time.localtime()
filename = '{}_{}_{}_{}:{}'.format(dat.tm_year, dat.tm_mon, dat.tm_mday, dat.tm_hour, dat.tm_min)
df.to_csv('mean_FOx.csv')
df.to_csv(join('results', 'dynamics_window_' + str(inter[0]) + '_' + str(inter[1]) + '_' + str(
    WINDOW_TIME) + '_' + args.VAR + '_' + filename + '.csv'))
