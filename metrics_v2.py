# coding: utf-8

import pandas as pd
import os
from os.path import join
import yaml
from collections import OrderedDict, defaultdict
import numpy as np
from tqdm import tqdm
import random
import cgitb

cgitb.enable(format='text')


class LEVEL(object):
    TRAINING = '[1, 2, 3, 4, 5, 6]'
    EASY = '[1, 4, 4, 5, 5, 6]'
    MEDIUM = '[1, 3, 3, 4, 4, 6]'
    HARD = '[1, 2, 2, 3, 3, 6]'


LAB_TO_LEV = {'TRAINING': '[1, 2, 3, 4, 5, 6]',
              'EASY': '[1, 4, 4, 5, 5, 6]',
              'MEDIUM': '[1, 3, 3, 4, 4, 6]',
              'HARD': '[1, 2, 2, 3, 3, 6]'}

LEV_TO_LAB = {v: k for k, v in LAB_TO_LEV.items()}

# ROI's definitions

ROIS = {
    'P1': [(-15, 280), (235, 30)],
    'P2': [(-15, 680), (235, 430)],
    'P3': [(-15, 1080), (235, 830)],
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


trial_dict = OrderedDict([
    ('1_E', 'EASY_CORR'),
    ('1_M', 'MEDIUM_CORR'),
    ('1_H', 'HARD_CORR'),
    ('4_E', 'EASY_SMALL_ERROR'),
    ('3_M', 'MEDIUM_SMALL_ERROR'),
    ('2_H', 'HARD_SMALL_ERROR'),
    ('5_E', 'EASY_BIG_ERROR'),
    ('4_M', 'MEDIUM_BIG_ERROR'),
    ('3_H', 'HARD_BIG_ERROR'),
    ('6_E', 'EASY_CONTROL'),
    ('6_M', 'MEDIUM_CONTROL'),
    ('6_H', 'HARD_CONTROL')])


def where_in_list(where, what):
    bound = len(where)
    tmp = map(lambda x: x['name'] == what, where)
    return [idx for idx, _ in zip(range(bound), tmp) if _]


ROIS_ORDER = ['P1', 'P2', 'P3', 'A', 'B', 'C', 'D', 'E', 'F']

RESULTS = list()

# Load data

SACC_FOLDER = join('..', 'Dane trackingowe', 'sacc')
BEH_FOLDER = join('..', 'results', 'beh')
FIX_FOLDER = join('..', 'Dane trackingowe', 'fix')
RAW_FOLDER = join('..', 'Dane trackingowe', 'raw')
YAML_FOLDER = join('..', 'results', 'yaml')

sacc_files = os.listdir(SACC_FOLDER)
sacc_files = [x for x in sacc_files if x.endswith('.csv')]
beh_files = os.listdir(BEH_FOLDER)
fix_files = os.listdir(FIX_FOLDER)
raw_files = os.listdir(RAW_FOLDER)
yaml_files = os.listdir(YAML_FOLDER)

sacc_files = [random.choice(sacc_files)]

with tqdm(total=len(sacc_files)) as pbar:
    for part_id in sacc_files:
        pbar.set_postfix(file=part_id)
        pbar.update(1)
        part_result = OrderedDict()
        part_id = part_id.split('_')[0]

        # Load Data
        sacc_data = pd.read_csv(join(SACC_FOLDER, part_id + '_sacc.csv')).drop('Unnamed: 0', 1)
        beh_data = pd.read_csv(join(BEH_FOLDER, part_id + '_beh.csv'))
        fix_data = pd.read_csv(join(FIX_FOLDER, part_id + '_fix.csv')).drop('Unnamed: 0', 1)

        if part_id in ['12MALE21', '14FEMALE19', '62FEMALE39', '83MALE27', '130MALE18', '142FEMALE29',
                       '165FEMALE20']:  # no Unnamed column
            raw_data = pd.read_csv(join(RAW_FOLDER, part_id + '_raw.csv'))
        else:
            raw_data = pd.read_csv(join(RAW_FOLDER, part_id + '_raw.csv')).drop('Unnamed: 0', 1)

        yaml_data = yaml.load(open(join(YAML_FOLDER, part_id + '.yaml'), 'r'))

        # problems loaded without training! training in block 0
        problems = yaml_data['list_of_blocks'][1]['experiment_elements'][1:]
        problems += yaml_data['list_of_blocks'][2]['experiment_elements'][1:]

        if 'FEMALE' in part_id:
            part_result['PART_ID'] = part_id.split('F')[0]
            part_result['SEX'] = 'FEMALE'
            part_result['AGE'] = int(part_id.split('FEMALE')[1])
        else:
            part_result['PART_ID'] = part_id.split('M')[0]
            part_result['SEX'] = 'MALE'
            part_result['AGE'] = int(part_id.split('MALE')[1])

        beh_data = beh_data[beh_data.exp == 'experiment']  # remove training
        beh_data['corr'] = beh_data['corr'].astype(int)

        # mean correctness
        part_result['ACC'] = beh_data['corr'].mean()
        w = beh_data.groupby('answers').mean()
        part_result["ACC_EASY"] = w['corr'][LEVEL.EASY]
        part_result["ACC_MEDIUM"] = w['corr'][LEVEL.MEDIUM]
        part_result["ACC_HARD"] = w['corr'][LEVEL.HARD]

        medium_err = beh_data[(beh_data.answers == LEVEL.MEDIUM) & (beh_data['corr'] == 0)]
        part_result['PERC_SE_MED'] = (medium_err.choosed_option == 'D3').mean()
        part_result['PERC_BE_MED'] = (medium_err.choosed_option == 'D4').mean()
        part_result['PERC_CON_MED'] = (medium_err.choosed_option == 'D6').mean()

        # Relational metric, now called TRS
        beh_data = beh_data[beh_data['choosed_option'] != '-1']  # removed unchoosed trials
        beh_data = beh_data[beh_data['rt'] > 10.0]  # remove reactions faster than 10 secs

        corr_beh = beh_data[beh_data['corr'] == 1]
        err_beh = beh_data[beh_data['corr'] == 0]

        trs = list()  # total
        ers = list()  # error
        for idx in beh_data.index:  # iterate over index, because some items are missed, due to choosed_option == -1
            choosed_option = beh_data['choosed_option'][idx]
            problem = problems[idx - 3]['matrix_info']
            err = not beh_data['corr'][idx]

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

        part_result['TRS'] = np.mean(trs)
        part_result['ERS'] = np.mean(ers)

        part_result['LAT_P'] = corr_beh['rt'].mean()
        part_result['LAT_N'] = err_beh['rt'].mean()

        lat_p = corr_beh.groupby('answers').rt.mean()
        lat_n = err_beh.groupby('answers').rt.mean()

        part_result["LAT_EASY_P"] = lat_p[LEVEL.EASY]
        part_result["LAT_MED_P"] = lat_p[LEVEL.MEDIUM]
        part_result["LAT_HARD_P"] = lat_p[LEVEL.HARD]

        # LAT_ERR_EASY – latency for incorrect, LAT_ERR_MED, LAT_ERR_HARD
        part_result["LAT_EASY_N"] = lat_n[LEVEL.EASY]
        part_result["LAT_MED_N"] = lat_n[LEVEL.MEDIUM]
        part_result["LAT_HARD_N"] = lat_n[LEVEL.HARD]

        # %%  Pupil size
        avg_pupil_size = raw_data.groupby('block').ps.mean()

        missing_blocks = set(range(1, 46)) - set(avg_pupil_size.index)
        for i in list(missing_blocks):
            avg_pupil_size.loc[i] = 0
        avg_pupil_size.sort_index(inplace=True)
        avg_pupil_size = avg_pupil_size.reset_index()

        beh_data['avg_pupil_size'] = avg_pupil_size['ps']
        pup_size = beh_data.groupby('answers').mean()['avg_pupil_size']
        part_result['PUP_SIZE_EASY'] = pup_size[LEVEL.EASY]
        part_result['PUP_SIZE_MED'] = pup_size[LEVEL.MEDIUM]
        part_result['PUP_SIZE_HARD'] = pup_size[LEVEL.HARD]

        pup_size = beh_data.groupby(['answers', 'corr']).mean()['avg_pupil_size']
        P, N = 1, 0
        part_result['PUP_SIZE_EASY_P'] = pup_size[LEVEL.EASY][P]
        part_result['PUP_SIZE_MED_P'] = pup_size[LEVEL.MEDIUM][P]
        part_result['PUP_SIZE_HARD_P'] = pup_size[LEVEL.HARD][P]

        part_result['PUP_SIZE_EASY_N'] = pup_size[LEVEL.EASY][N]
        part_result['PUP_SIZE_MED_N'] = pup_size[LEVEL.MEDIUM][N]
        part_result['PUP_SIZE_HARD_N'] = pup_size[LEVEL.HARD][N]
        # %% toggle rate (NT)

        sacc_ends_in_op = pd.concat([in_roi(sacc_data[['exp', 'eyp']], ROIS['A']),
                                     in_roi(sacc_data[['exp', 'eyp']], ROIS['B']),
                                     in_roi(sacc_data[['exp', 'eyp']], ROIS['C']),
                                     in_roi(sacc_data[['exp', 'eyp']], ROIS['D']),
                                     in_roi(sacc_data[['exp', 'eyp']], ROIS['E']),
                                     in_roi(sacc_data[['exp', 'eyp']], ROIS['F'])], axis=1).any(axis=1)

        sacc_ends_in_pr = pd.concat([in_roi(sacc_data[['exp', 'eyp']], ROIS['P1']),
                                     in_roi(sacc_data[['exp', 'eyp']], ROIS['P2']),
                                     in_roi(sacc_data[['exp', 'eyp']], ROIS['P3'])], axis=1).any(axis=1)

        toggled_sacc = sacc_data[pd.concat([sacc_ends_in_pr, sacc_ends_in_op], axis=1).any(axis=1)]

        # stime used just for counting how many events occurs in any particular block
        ts = toggled_sacc.groupby('block').count()['stime']
        del toggled_sacc
        missing_blocks = set(range(1, 46)) - set(ts.index)

        for i in list(missing_blocks):
            ts.loc[i] = 0
        ts.sort_index(inplace=True)
        ts = ts.reset_index()
        beh_data['nt'] = ts['stime']
        del ts

        toggles = beh_data.groupby('answers').sum()

        part_result["NT_EASY"] = toggles['nt'][LEVEL.EASY]
        part_result["NT_MEDIUM"] = toggles['nt'][LEVEL.MEDIUM]
        part_result["NT_HARD"] = toggles['nt'][LEVEL.HARD]

        toggles = beh_data.groupby(['answers', 'corr']).sum()

        part_result["NT_EASY_P"] = toggles['nt'][LEVEL.EASY][P]
        part_result["NT_MED_P"] = toggles['nt'][LEVEL.MEDIUM][P]
        part_result["NT_HARD_P"] = toggles['nt'][LEVEL.HARD][P]

        part_result["NT_EASY_N"] = toggles['nt'][LEVEL.EASY][N]
        part_result["NT_MED_N"] = toggles['nt'][LEVEL.MEDIUM][N]
        part_result["NT_HARD_N"] = toggles['nt'][LEVEL.HARD][N]

        # NT_PR
        toggled_sacc = sacc_data[sacc_ends_in_pr]

        ts = toggled_sacc.groupby('block').count()['stime']
        del toggled_sacc
        missing_blocks = set(range(1, 46)) - set(ts.index)

        for i in list(missing_blocks):
            ts.loc[i] = 0
        ts.sort_index(inplace=True)
        ts = ts.reset_index()
        beh_data['nt'] = ts['stime']
        del ts

        toggles = beh_data.groupby('answers').sum()

        part_result["NT_PR_EASY"] = toggles['nt'][LEVEL.EASY]
        part_result["NT_PR_MEDIUM"] = toggles['nt'][LEVEL.MEDIUM]
        part_result["NT_PR_HARD"] = toggles['nt'][LEVEL.HARD]

        toggles = beh_data.groupby(['answers', 'corr']).sum()

        part_result["NT_PR_EASY_P"] = toggles['nt'][LEVEL.EASY][P]
        part_result["NT_PR_MEDIUM_P"] = toggles['nt'][LEVEL.MEDIUM][P]
        part_result["NT_PR_HARD_P"] = toggles['nt'][LEVEL.HARD][P]

        part_result["NT_PR_EASY_N"] = toggles['nt'][LEVEL.EASY][N]
        part_result["NT_PR_MED_N"] = toggles['nt'][LEVEL.MEDIUM][N]
        part_result["NT_PR_HARD_N"] = toggles['nt'][LEVEL.HARD][N]

        # NT_OP
        toggled_sacc = sacc_data[sacc_ends_in_op]

        ts = toggled_sacc.groupby('block').count()['stime']
        del toggled_sacc
        missing_blocks = set(range(1, 46)) - set(ts.index)

        for i in list(missing_blocks):
            ts.loc[i] = 0
        ts.sort_index(inplace=True)
        ts = ts.reset_index()
        beh_data['nt'] = ts['stime']
        del ts

        toggles = beh_data.groupby('answers').sum()

        part_result["NT_OP_EASY"] = toggles['nt'][LEVEL.EASY]
        part_result["NT_OP_MEDIUM"] = toggles['nt'][LEVEL.MEDIUM]
        part_result["NT_OP_HARD"] = toggles['nt'][LEVEL.HARD]

        toggles = beh_data.groupby(['answers', 'corr']).sum()

        part_result["NT_OP_EASY_P"] = toggles['nt'][LEVEL.EASY][P]
        part_result["NT_OP_MEDIUM_P"] = toggles['nt'][LEVEL.MEDIUM][P]
        part_result["NT_OP_HARD_P"] = toggles['nt'][LEVEL.HARD][P]

        part_result["NT_OP_EASY_N"] = toggles['nt'][LEVEL.EASY][N]
        part_result["NT_OP_MED_N"] = toggles['nt'][LEVEL.MEDIUM][N]
        part_result["NT_OP_HARD_N"] = toggles['nt'][LEVEL.HARD][N]

        # NT_COR_EASY – no of toggles on correct option NT_COR_MED  NT_COR_HARD
        # NT_ERR_EASY – no of toggles on incorrect options NT_ERR_MED, NT_ERR_HARD
        nt_cor = {'EASY': 0, 'MEDIUM': 0, 'HARD': 0}
        nt_cor_p = {'EASY': 0, 'MEDIUM': 0, 'HARD': 0}
        nt_cor_n = {'EASY': 0, 'MEDIUM': 0, 'HARD': 0}
        nt_err = {'EASY': 0, 'MEDIUM': 0, 'HARD': 0}
        nt_err_p = {'EASY': 0, 'MEDIUM': 0, 'HARD': 0}
        nt_err_n = {'EASY': 0, 'MEDIUM': 0, 'HARD': 0}
        nt_med = {'SE': 0, 'BE': 0, 'CON': 0}

        for idx in beh_data.index:
            problem = problems[idx - 3]
            sacc_item = sacc_data[sacc_data.block == idx]

            # CORR == D1
            cor_roi = ROIS_ORDER[where_in_list(problem['matrix_info'], 'D1')[0]]
            err_roi = [where_in_list(problem['matrix_info'], x) for x in ['D2', 'D3', 'D4', 'D5', 'D6']]
            err_roi = [ROIS_ORDER[item] for sublist in err_roi for item in sublist]

            sacc_ends_in_corr = in_roi(sacc_item[['exp', 'eyp']], ROIS[cor_roi])
            sacc_ends_in_err = pd.concat([in_roi(sacc_item[['exp', 'eyp']], ROIS[x]) for x in err_roi], axis=1).any(
                axis=1)

            level = LEV_TO_LAB[str(problem['answers'])]
            nt_cor[level] += sacc_ends_in_corr.sum()
            nt_err[level] += sacc_ends_in_err.sum()

            if level == 'MEDIUM':
                se_roi = [ROIS_ORDER[x] for x in where_in_list(problem['matrix_info'], 'D3')]
                sacc_ends_in_se = pd.concat([in_roi(sacc_item[['exp', 'eyp']], ROIS[x]) for x in se_roi], axis=1).any(
                    axis=1)

                be_roi = [ROIS_ORDER[x] for x in where_in_list(problem['matrix_info'], 'D4')]
                sacc_ends_in_be = pd.concat([in_roi(sacc_item[['exp', 'eyp']], ROIS[x]) for x in be_roi], axis=1).any(
                    axis=1)

                con_roi = ROIS_ORDER[where_in_list(problem['matrix_info'], 'D6')[0]]
                sacc_ends_in_con = in_roi(sacc_item[['exp', 'eyp']], ROIS[con_roi])

                nt_med['SE'] += sacc_ends_in_se.sum()
                nt_med['BE'] += sacc_ends_in_be.sum()
                nt_med['CON'] += sacc_ends_in_con.sum()

        part_result['NT_COR_EASY'] = nt_cor['EASY']
        part_result['NT_COR_MED'] = nt_cor['MEDIUM']
        part_result['NT_COR_HARD'] = nt_cor['HARD']

        part_result['NT_ERR_EASY'] = nt_err['EASY']
        part_result['NT_ERR_MED'] = nt_err['MEDIUM']
        part_result['NT_ERR_HARD'] = nt_err['HARD']

        part_result['NT_SE_MED'] = nt_med['SE']
        part_result['NT_BE_MED'] = nt_med['BE']
        part_result['NT_CON_MED'] = nt_med['CON']

        for idx in corr_beh.index:
            problem = problems[idx - 3]
            sacc_item = sacc_data[sacc_data.block == idx]

            # CORR == D1
            cor_roi = ROIS_ORDER[where_in_list(problem['matrix_info'], 'D1')[0]]
            err_roi = [where_in_list(problem['matrix_info'], x) for x in ['D2', 'D3', 'D4', 'D5', 'D6']]
            err_roi = [ROIS_ORDER[item] for sublist in err_roi for item in sublist]

            sacc_ends_in_corr = in_roi(sacc_item[['exp', 'eyp']], ROIS[cor_roi])
            sacc_ends_in_err = pd.concat([in_roi(sacc_item[['exp', 'eyp']], ROIS[x]) for x in err_roi], axis=1).any(
                axis=1)

            level = LEV_TO_LAB[str(problem['answers'])]
            nt_cor_p[level] += sacc_ends_in_corr.sum()
            nt_err_p[level] += sacc_ends_in_err.sum()

        part_result['NT_COR_EASY_P'] = nt_cor_p['EASY']
        part_result['NT_COR_MED_P'] = nt_cor_p['MEDIUM']
        part_result['NT_COR_HARD_P'] = nt_cor_p['HARD']

        part_result['NT_ERR_EASY_P'] = nt_err_p['EASY']
        part_result['NT_ERR_MED_P'] = nt_err_p['MEDIUM']
        part_result['NT_ERR_HARD_P'] = nt_err_p['HARD']

        for idx in err_beh.index:
            problem = problems[idx - 3]
            sacc_item = sacc_data[sacc_data.block == idx]

            # CORR == D1
            cor_roi = ROIS_ORDER[where_in_list(problem['matrix_info'], 'D1')[0]]
            err_roi = [where_in_list(problem['matrix_info'], x) for x in ['D2', 'D3', 'D4', 'D5', 'D6']]
            err_roi = [ROIS_ORDER[item] for sublist in err_roi for item in sublist]

            sacc_ends_in_corr = in_roi(sacc_item[['exp', 'eyp']], ROIS[cor_roi])
            sacc_ends_in_err = pd.concat([in_roi(sacc_item[['exp', 'eyp']], ROIS[x]) for x in err_roi], axis=1).any(
                axis=1)

            level = LEV_TO_LAB[str(problem['answers'])]
            nt_cor_n[level] += sacc_ends_in_corr.sum()
            nt_err_n[level] += sacc_ends_in_err.sum()

        part_result['NT_COR_EASY_N'] = nt_cor_n['EASY']
        part_result['NT_COR_MED_N'] = nt_cor_n['MEDIUM']
        part_result['NT_COR_HARD_N'] = nt_cor_n['HARD']

        part_result['NT_ERR_EASY_N'] = nt_err_n['EASY']
        part_result['NT_ERR_MED_N'] = nt_err_n['MEDIUM']
        part_result['NT_ERR_HARD_N'] = nt_err_n['HARD']

        # relative time (RT)

        fix_in_pr = pd.concat([in_roi(fix_data[['axp', 'ayp']], ROIS[x]) for x in ['P1', 'P2', 'P3']], axis=1).any(
            axis=1)

        fix_in_pr_dur = fix_data[fix_in_pr].groupby('block').sum()['dur']

        missing_blocks = set(range(1, 46)) - set(fix_in_pr_dur.index)
        for i in list(missing_blocks):
            fix_in_pr_dur.loc[i] = 0
        fix_in_pr_dur.sort_index(inplace=True)
        fix_in_pr_dur = fix_in_pr_dur.reset_index()

        beh_data['fix_in_pr_dur'] = fix_in_pr_dur['dur']

        gb = beh_data.groupby('answers').sum()
        rt_pr = (gb['fix_in_pr_dur'] / (gb['rt'] * 1000.0))

        part_result["RT_PR_EASY"] = rt_pr[LEVEL.EASY]
        part_result["RT_PR_MEDIUM"] = rt_pr[LEVEL.MEDIUM]
        part_result["RT_PR_HARD"] = rt_pr[LEVEL.HARD]

        # RT_SE/BE/CON_MED
        rt_med = {'SE': 0, 'BE': 0, 'CON': 0}
        for idx in beh_data.index:
            beh_item = beh_data.ix[idx]
            problem = problems[idx - 3]
            fix_item = fix_data[fix_data.block == idx]
            level = LEV_TO_LAB[str(problem['answers'])]
            if level != 'MEDIUM':
                continue

            se_roi = [ROIS_ORDER[x] for x in where_in_list(problem['matrix_info'], 'D3')]
            fix_in_se = pd.concat([in_roi(fix_item[['axp', 'ayp']], ROIS[x]) for x in se_roi], axis=1).any(axis=1)
            rt_med['SE'] += fix_item[fix_in_se]['dur'].sum() / (beh_item['rt'] * 1000.0)

            be_roi = [ROIS_ORDER[x] for x in where_in_list(problem['matrix_info'], 'D4')]
            fix_in_be = pd.concat([in_roi(fix_item[['axp', 'ayp']], ROIS[x]) for x in be_roi], axis=1).any(axis=1)
            rt_med['BE'] += fix_item[fix_in_be]['dur'].sum() / (beh_item['rt'] * 1000.0)

            con_roi = ROIS_ORDER[where_in_list(problem['matrix_info'], 'D6')[0]]
            fix_in_con = in_roi(fix_item[['axp', 'ayp']], ROIS[con_roi])
            rt_med['CON'] += fix_item[fix_in_con]['dur'].sum() / (beh_item['rt'] * 1000.0)

        part_result["RT_SE_MED"] = rt_med['SE']
        part_result["RT_BE_MED"] = rt_med['BE']
        part_result["RT_CON_MED"] = rt_med['CON']

        # relative time (RT_PR_P/N)

        gb = beh_data.groupby(['answers', 'corr']).sum()
        rt_pr = (gb['fix_in_pr_dur'] / (gb['rt'] * 1000.0))

        part_result["RT_PR_EASY_P"] = rt_pr[LEVEL.EASY][P]
        part_result["RT_PR_MED_P"] = rt_pr[LEVEL.MEDIUM][P]
        part_result["RT_PR_HARD_P"] = rt_pr[LEVEL.HARD][P]

        part_result["RT_PR_EASY_N"] = rt_pr[LEVEL.EASY][N]
        part_result["RT_PR_MED_N"] = rt_pr[LEVEL.MEDIUM][N]
        part_result["RT_PR_HARD_N"] = rt_pr[LEVEL.HARD][N]


        # relative time (RT_OP)

        fix_in_op = pd.concat([in_roi(fix_data[['axp', 'ayp']], ROIS[x]) for x in 'ABCDEF'], axis=1).any(axis=1)

        fix_in_op_dur = fix_data[fix_in_op].groupby('block').sum()['dur']

        missing_blocks = set(range(1, 46)) - set(fix_in_op_dur.index)
        for i in list(missing_blocks):
            fix_in_op_dur.loc[i] = 0
        fix_in_op_dur.sort_index(inplace=True)
        fix_in_op_dur = fix_in_op_dur.reset_index()

        beh_data['fix_in_op_dur'] = fix_in_op_dur['dur']

        gb = beh_data.groupby('answers').sum()
        rt_op = (gb['fix_in_op_dur'] / (gb['rt'] * 1000.0))

        part_result["RT_OP_EASY"] = rt_op[LEVEL.EASY]
        part_result["RT_OP_MEDIUM"] = rt_op[LEVEL.MEDIUM]
        part_result["RT_OP_HARD"] = rt_op[LEVEL.HARD]

        # relative time (RT_OP_P/N)

        gb = beh_data.groupby(['answers', 'corr']).sum()
        rt_op = (gb['fix_in_op_dur'] / (gb['rt'] * 1000.0))

        part_result["RT_OP_EASY_P"] = rt_op[LEVEL.EASY][P]
        part_result["RT_OP_MED_P"] = rt_op[LEVEL.MEDIUM][P]
        part_result["RT_OP_HARD_P"] = rt_op[LEVEL.HARD][P]

        part_result["RT_OP_EASY_N"] = rt_op[LEVEL.EASY][N]
        part_result["RT_OP_MED_N"] = rt_op[LEVEL.MEDIUM][N]
        part_result["RT_OP_HARD_N"] = rt_op[LEVEL.HARD][N]

        # relative time (RT_COR_EASY)
        # TUTAJ SKOŃCZYŁEM PRACĘ
        rt_cor = {'SE': 0, 'BE': 0, 'CON': 0}
        for idx in beh_data.index:
            beh_item = beh_data.ix[idx]
            problem = problems[idx - 3]
            fix_item = fix_data[fix_data.block == idx]
            level = LEV_TO_LAB[str(problem['answers'])]

            cor_roi = [ROIS_ORDER[x] for x in where_in_list(problem['matrix_info'], 'D1')][0]

            fix_in_cor = pd.concat([in_roi(fix_item[['axp', 'ayp']], ROIS[x]) for x in se_roi], axis=1).any(axis=1)
            rt_med['SE'] += fix_item[fix_in_se]['dur'].sum() / (beh_item['rt'] * 1000.0)

            be_roi = [ROIS_ORDER[x] for x in where_in_list(problem['matrix_info'], 'D4')]
            fix_in_be = pd.concat([in_roi(fix_item[['axp', 'ayp']], ROIS[x]) for x in be_roi], axis=1).any(axis=1)
            rt_med['BE'] += fix_item[fix_in_be]['dur'].sum() / (beh_item['rt'] * 1000.0)

            con_roi = ROIS_ORDER[where_in_list(problem['matrix_info'], 'D6')[0]]
            fix_in_con = in_roi(fix_item[['axp', 'ayp']], ROIS[con_roi])
            rt_med['CON'] += fix_item[fix_in_con]['dur'].sum() / (beh_item['rt'] * 1000.0)

        part_result["RT_SE_MED"] = rt_med['SE']
        part_result["RT_BE_MED"] = rt_med['BE']
        part_result["RT_CON_MED"] = rt_med['CON']




#         # # 3. relative first response fixation (RFRF)
#         # timestamp of the first fixation within the response area (time before first response fixation) divided by total response time.
#
#         fix_in_problem_area = pd.concat([in_roi(fix_data[['axp', 'ayp']], ROIS['P1']),
#                                          in_roi(fix_data[['axp', 'ayp']], ROIS['P2']),
#                                          in_roi(fix_data[['axp', 'ayp']], ROIS['P3'])], axis=1).any(axis=1)
#         first_fix_in_problem_area = np.where(fix_in_problem_area == True)[0][0]
#         fix_data_for_RF = fix_data[first_fix_in_problem_area:]
#
#         fix_in_matrix_area = pd.concat([in_roi(fix_data_for_RF[['axp', 'ayp']], ROIS['A']),
#                                         in_roi(fix_data_for_RF[['axp', 'ayp']], ROIS['B']),
#                                         in_roi(fix_data_for_RF[['axp', 'ayp']], ROIS['C']),
#                                         in_roi(fix_data_for_RF[['axp', 'ayp']], ROIS['D']),
#                                         in_roi(fix_data_for_RF[['axp', 'ayp']], ROIS['E']),
#                                         in_roi(fix_data_for_RF[['axp', 'ayp']], ROIS['F'])], axis=1).any(axis=1)
#         first_fix_in_matrix_ds = fix_data_for_RF[fix_in_matrix_area].groupby('block').first()['stime']
#         blocks_start_ds = raw_data.groupby('block').first()['time']
#         beh_data['time_of_first_fix_on_matrix_area'] = \
#             (first_fix_in_matrix_ds - blocks_start_ds).fillna(0).reset_index()[0]
#
#         gb = beh_data.groupby('answers').sum()
#         rfrf = (gb['time_of_first_fix_on_matrix_area'] / (gb['rt'] * 1000.0))
#         part_result["RFRF_EASY"] = rfrf[LEVEL.EASY]
#         part_result["RFRF_MEDIUM"] = rfrf[LEVEL.MEDIUM]
#         part_result["RFRF_HARD"] = rfrf[LEVEL.HARD]
#         # # number of responses visited (RV)
#         # counted all response alternatives that were fixated at least once during the response time (following Bethell-Fox et al., 1984)
#         #
#         # Modyfikacja! Teraz: liczba fiksacji + łączny czas trwania fiksacji na każdej z odpowiedzi. [time spent on each response alternative]
#         # %%
#         problems = yaml_data['list_of_blocks'][1]['experiment_elements'][1:]
#         problems += yaml_data['list_of_blocks'][2]['experiment_elements'][1:]
#         names = list()
#         N = list()
#         for problem in problems:
#             N.append(problem['rel'])
#             tmp = []
#             for matrix in problem['matrix_info']:
#                 tmp.append(matrix['name'])
#             names.append(tmp)
#         names = [x[3:] for x in names]  # remove question area (A B C)
#
#         # %%
#         new_names = []
#         res = []
#         for x in names:
#             new_names.append([int(w[-1]) for w in x])
#         assert len(new_names) == len(beh_data['answers']) == len(N), 'Corrupted behavioral data files'
#         # %%
#         for name, lab, n in zip(new_names, beh_data['answers'].map({
#             '[1, 2, 2, 3, 3, 6]': 'H',
#             '[1, 3, 3, 4, 4, 6]': 'M',
#             '[1, 4, 4, 5, 5, 6]': 'E'}), N):
#             new_name = []
#             for item in name:
#                 new_name.append(str(item) + '_' + str(lab))
#             res.append(new_name)
#
#         new_names = res
#
#         # %%
#         trial_time = raw_data.groupby('block')
#         trial_time = pd.DataFrame([trial_time.first()['time'], trial_time.last()['time']])
#
#         trial_time = trial_time.quantile(q=[0.25, 0.5, 0.75]).transpose()
#         # %%
#         quarter = list()
#         for fixation in fix_data.iterrows():
#             q25, q50, q75 = trial_time.ix[fixation[1]['block']]
#             stime = fixation[1]['stime']
#             if stime <= q25:
#                 quarter.append(1)
#             elif stime >= q25 and stime < q50:
#                 quarter.append(2)
#             elif stime >= q50 and stime < q75:
#                 quarter.append(3)
#             else:  # stime > q75
#                 quarter.append(4)
#
#         assert fix_data.shape[0] == len(quarter), 'Quarters wrongly calculated'
#         fix_data['quarter'] = quarter
#         # %%
#         index = pd.MultiIndex.from_product([range(1, 46), [0]])
#
#         fix_in_A = fix_data[in_roi(fix_data[['axp', 'ayp']], ROIS['A'])].groupby(['block', 'quarter']).sum()['dur']
#         fix_in_A_2 = pd.Series(pd.Series(fix_in_A.sum(level=0), index=range(1, 46)).values, index=index)
#         fix_in_A = pd.concat([fix_in_A, fix_in_A_2], axis=1)
#         fix_in_A = fix_in_A['dur'].fillna(fix_in_A[0])
#
#         fix_in_B = fix_data[in_roi(fix_data[['axp', 'ayp']], ROIS['B'])].groupby(['block', 'quarter']).sum()['dur']
#         fix_in_B_2 = pd.Series(pd.Series(fix_in_B.sum(level=0), index=range(1, 46)).values, index=index)
#         fix_in_B = pd.concat([fix_in_B, fix_in_B_2], axis=1)
#         fix_in_B = fix_in_B['dur'].fillna(fix_in_B[0])
#
#         fix_in_C = fix_data[in_roi(fix_data[['axp', 'ayp']], ROIS['C'])].groupby(['block', 'quarter']).sum()['dur']
#         fix_in_C_2 = pd.Series(pd.Series(fix_in_C.sum(level=0), index=range(1, 46)).values, index=index)
#         fix_in_C = pd.concat([fix_in_C, fix_in_C_2], axis=1)
#         fix_in_C = fix_in_C['dur'].fillna(fix_in_C[0])
#
#         fix_in_D = fix_data[in_roi(fix_data[['axp', 'ayp']], ROIS['D'])].groupby(['block', 'quarter']).sum()['dur']
#         fix_in_D_2 = pd.Series(pd.Series(fix_in_D.sum(level=0), index=range(1, 46)).values, index=index)
#         fix_in_D = pd.concat([fix_in_D, fix_in_D_2], axis=1)
#         fix_in_D = fix_in_D['dur'].fillna(fix_in_D[0])
#
#         fix_in_E = fix_data[in_roi(fix_data[['axp', 'ayp']], ROIS['E'])].groupby(['block', 'quarter']).sum()['dur']
#         fix_in_E_2 = pd.Series(pd.Series(fix_in_E.sum(level=0), index=range(1, 46)).values, index=index)
#         fix_in_E = pd.concat([fix_in_E, fix_in_E_2], axis=1)
#         fix_in_E = fix_in_E['dur'].fillna(fix_in_E[0])
#
#         fix_in_F = fix_data[in_roi(fix_data[['axp', 'ayp']], ROIS['F'])].groupby(['block', 'quarter']).sum()['dur']
#         fix_in_F_2 = pd.Series(pd.Series(fix_in_F.sum(level=0), index=range(1, 46)).values, index=index)
#         fix_in_F = pd.concat([fix_in_F, fix_in_F_2], axis=1)
#         fix_in_F = fix_in_F['dur'].fillna(fix_in_F[0])
#
#         # %% creating mock in order to fill all absent values in df
#         high_index = sorted(list(range(1, 46)) + list(range(1, 46)) + list(range(1, 46)) + list(range(1, 46)))
#         low_index = list(range(1, 5)) * 45
#         assert len(high_index) == len(low_index), "Error in mock index geration"
#         tuples = list(zip(high_index, low_index))
#         index = pd.MultiIndex.from_tuples(tuples, names=['block', 'quarter'])
#         mock = pd.Series(np.NaN, index=index)
#         fix_dur = pd.concat([fix_in_A, fix_in_B, fix_in_C, fix_in_D, fix_in_E, fix_in_F, mock], axis=1).drop('blk', 1)
#         fix_dur.columns = ['dur_in_A', 'dur_in_B', 'dur_in_C', 'dur_in_D', 'dur_in_E', 'dur_in_F', 'mock']
#         fix_dur = fix_dur.fillna(0).drop('mock', 1)
#
#         # %%
#         d = defaultdict(list)
#         for l, row in zip(new_names, fix_dur.xs(0, level=1)[3:].iterrows()):
#             if (row[0]) in [1, 2, 3]:  # trening
#                 continue
#             for k, v in zip(l, row[1][2:]):
#                 d[k].append(v)
#         # %%
#         for k in trial_dict.keys():
#             part_result['RV_AVG_DUR_' + trial_dict[k]] = np.nanmean(d[k])
#
#             # %%
#         fix_in_A = fix_data[in_roi(fix_data[['axp', 'ayp']], ROIS['A'])].groupby('block').count()['dur']
#         fix_in_B = fix_data[in_roi(fix_data[['axp', 'ayp']], ROIS['B'])].groupby('block').count()['dur']
#         fix_in_C = fix_data[in_roi(fix_data[['axp', 'ayp']], ROIS['C'])].groupby('block').count()['dur']
#         fix_in_D = fix_data[in_roi(fix_data[['axp', 'ayp']], ROIS['D'])].groupby('block').count()['dur']
#         fix_in_E = fix_data[in_roi(fix_data[['axp', 'ayp']], ROIS['E'])].groupby('block').count()['dur']
#         fix_in_F = fix_data[in_roi(fix_data[['axp', 'ayp']], ROIS['F'])].groupby('block').count()['dur']
#
#         missing_blocks = set(range(1, 46)) - set(fix_in_A.index)
#         for i in list(missing_blocks):
#             fix_in_A.loc[i] = 0
#         fix_in_A.sort_index(inplace=True)
#         fix_in_A = fix_in_A.reset_index()
#
#         missing_blocks = set(range(1, 46)) - set(fix_in_B.index)
#         for i in list(missing_blocks):
#             fix_in_B.loc[i] = 0
#         fix_in_B.sort_index(inplace=True)
#         fix_in_B = fix_in_B.reset_index()
#
#         missing_blocks = set(range(1, 46)) - set(fix_in_C.index)
#         for i in list(missing_blocks):
#             fix_in_C.loc[i] = 0
#         fix_in_C.sort_index(inplace=True)
#         fix_in_C = fix_in_C.reset_index()
#
#         missing_blocks = set(range(1, 46)) - set(fix_in_D.index)
#         for i in list(missing_blocks):
#             fix_in_D.loc[i] = 0
#         fix_in_D.sort_index(inplace=True)
#         fix_in_D = fix_in_D.reset_index()
#
#         missing_blocks = set(range(1, 46)) - set(fix_in_E.index)
#         for i in list(missing_blocks):
#             fix_in_E.loc[i] = 0
#         fix_in_E.sort_index(inplace=True)
#         fix_in_E = fix_in_E.reset_index()
#
#         missing_blocks = set(range(1, 46)) - set(fix_in_F.index)
#         for i in list(missing_blocks):
#             fix_in_F.loc[i] = 0
#         fix_in_F.sort_index(inplace=True)
#         fix_in_F = fix_in_F.reset_index()
#
#         # print fix_in_A.columns
#         fix_in_A.columns = ['block', 'no_in_A']
#         fix_in_B.columns = ['blk', 'no_in_B']
#         fix_in_C.columns = ['blk', 'no_in_C']
#         fix_in_D.columns = ['blk', 'no_in_D']
#         fix_in_E.columns = ['blk', 'no_in_E']
#         fix_in_F.columns = ['blk', 'no_in_F']
#
#         fix_dur = pd.concat([fix_in_A, fix_in_B, fix_in_C, fix_in_D, fix_in_E, fix_in_F], axis=1).drop('blk', 1)
#         fix_dur = fix_dur[fix_dur.block != 1]
#         fix_dur = fix_dur[fix_dur.block != 2]
#         fix_dur = fix_dur[fix_dur.block != 3]
#         fix_dur.reset_index(inplace=True)
#
#         d = defaultdict(list)
#         for l, row in zip(new_names, fix_dur.iterrows()):
#             for k, v in zip(l, row[1][2:]):
#                 d[k].append(v)
#         # BIG_ERROR i SMALL_ERROR must be divised by two
#         for k in trial_dict.keys():
#             if 'ERROR' in trial_dict[k]:
#                 part_result['RV_SUM_FIX_' + trial_dict[k]] = np.nansum(d[k]) / 2.0
#             else:
#                 part_result['RV_SUM_FIX_' + trial_dict[k]] = np.nansum(d[k])
#
#                 # # Pupil size
#         # %%
#         avg_pupil_size = raw_data.groupby('block').mean()['ps']
#
#         missing_blocks = set(range(1, 46)) - set(avg_pupil_size.index)
#         for i in list(missing_blocks):
#             avg_pupil_size.loc[i] = 0
#         avg_pupil_size.sort_index(inplace=True)
#         avg_pupil_size = avg_pupil_size.reset_index()
#
#         beh_data['avg_pupil_size'] = avg_pupil_size['ps']
#         w = beh_data.groupby('answers').mean()['avg_pupil_size']
#         part_result["AVG_PUP_SIZE_EASY"] = w[LEVEL.EASY]
#         part_result["AVG_PUP_SIZE_MEDIUM"] = w[LEVEL.MEDIUM]
#         part_result["AVG_PUP_SIZE_HARD"] = w[LEVEL.HARD]
#
#         part_result["MEAN_RT_EASY"] = w['rt'][LEVEL.EASY]
#         part_result["MEAN_RT_MEDIUM"] = w['rt'][LEVEL.MEDIUM]
#         part_result["MEAN_RT_HARD"] = w['rt'][LEVEL.HARD]
#
#         # %% percentage of individual options in given answers
#
#         beh_data = beh_data[beh_data['choosed_option'] != '-1']  # removed unchoosed trials
#         easy = beh_data[beh_data['answers'] == LEVEL.EASY]
#         medium = beh_data[beh_data['answers'] == LEVEL.MEDIUM]
#         hard = beh_data[beh_data['answers'] == LEVEL.HARD]
#
#         easy = easy.groupby('choosed_option').count()['ans_accept'] / easy.shape[0]
#         medium = medium.groupby('choosed_option').count()['ans_accept'] / medium.shape[0]
#         hard = hard.groupby('choosed_option').count()['ans_accept'] / hard.shape[0]
#
#         part_result['PERC_CORR_EASY'] = easy.get('D1', 0.0)
#         part_result['PERC_CORR_MEDIUM'] = medium.get('D1', 0.0)
#         part_result['PERC_CORR_HARD'] = hard.get('D1', 0.0)
#
#         part_result['PERC_SMALL_ERROR_EASY'] = easy.get('D4', 0.0)
#         part_result['PERC_SMALL_ERROR_MEDIUM'] = medium.get('D3', 0.0)
#         part_result['PERC_SMALL_ERROR_HARD'] = hard.get('D2', 0.0)
#
#         part_result['PERC_BIG_ERROR_EASY'] = easy.get('D5', 0.0)
#         part_result['PERC_BIG_ERROR_MEDIUM'] = medium.get('D4', 0.0)
#         part_result['PERC_BIG_ERROR_HARD'] = hard.get('D3', 0.0)
#
#         part_result['PERC_CONTROL_EASY'] = easy.get('D6', 0.0)
#         part_result['PERC_CONTROL_MEDIUM'] = medium.get('D6', 0.0)
#         part_result['PERC_CONTROL_HARD'] = hard.get('D6', 0.0)
#
#         # %%
#         RESULTS.append(part_result)
#
# # %%Save results
# res = pd.DataFrame(RESULTS)
# pd.DataFrame(RESULTS).to_csv('new_result.csv')
# pd.DataFrame(RESULTS).to_excel('new_result.xlsx')
