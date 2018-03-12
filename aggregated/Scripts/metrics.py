# coding: utf-8

import pandas as pd
import os
from os.path import join
import yaml
from collections import OrderedDict
import numpy as np
from tqdm import tqdm
import cgitb

os.chdir(join( '..', '..', 'Dropbox', 'Data', 'FAN_ET', 'Badanie P', '2017-05-06_Badanie_P', 'BadanieP_FAN_ET', 'Scripts'))

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

# sacc_files = [random.choice(sacc_files)]

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

        beh_data = beh_data[beh_data['choosed_option'] != '-1']  # removed unchoosed trials

        beh_medium = beh_data[beh_data.answers == LEVEL.MEDIUM]
        part_result['PERC_SE_MED'] = (beh_medium.choosed_option == 'D3').mean()
        part_result['PERC_BE_MED'] = (beh_medium.choosed_option == 'D4').mean()
        part_result['PERC_CON_MED'] = (beh_medium.choosed_option == 'D6').mean()

        # Relational metric, now called TRS
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
        try:
            part_result["LAT_HARD_P"] = lat_p[LEVEL.HARD]
        except KeyError:
            part_result["LAT_HARD_P"] = np.nan

            # LAT_ERR_EASY – latency for incorrect, LAT_ERR_MED, LAT_ERR_HARD
        part_result["LAT_EASY_N"] = lat_n.get(LEVEL.EASY, np.nan)
        part_result["LAT_MED_N"] = lat_n.get(LEVEL.MEDIUM, np.nan)
        part_result["LAT_HARD_N"] = lat_n.get(LEVEL.HARD, )

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
        try:
            part_result['PUP_SIZE_HARD_P'] = pup_size[LEVEL.HARD][P]
        except:
            part_result['PUP_SIZE_HARD_P'] = np.nan

        part_result['PUP_SIZE_EASY_N'] = pup_size[LEVEL.EASY].get(N, np.nan)
        part_result['PUP_SIZE_MED_N'] = pup_size[LEVEL.MEDIUM].get(N, np.nan)
        part_result['PUP_SIZE_HARD_N'] = pup_size[LEVEL.HARD].get(N, np.nan)
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
        try:
            part_result["NT_HARD_P"] = toggles['nt'][LEVEL.HARD][P]
        except KeyError:
            part_result["NT_HARD_P"] = np.nan

        part_result["NT_EASY_N"] = toggles['nt'][LEVEL.EASY].get(N, np.nan)
        part_result["NT_MED_N"] = toggles['nt'][LEVEL.MEDIUM].get(N, np.nan)
        part_result["NT_HARD_N"] = toggles['nt'][LEVEL.HARD].get(N, np.nan)

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

        try:
            part_result["NT_PR_EASY_P"] = toggles['nt'][LEVEL.EASY][P]
        except KeyError:
            part_result["NT_PR_EASY_P"] = np.nan
        try:
            part_result["NT_PR_MEDIUM_P"] = toggles['nt'][LEVEL.MEDIUM][P]
        except KeyError:
            part_result["NT_PR_MEDIUM_P"] = np.nan
        try:
            part_result["NT_PR_HARD_P"] = toggles['nt'][LEVEL.HARD][P]
        except KeyError:
            part_result["NT_PR_HARD_P"] = np.nan

        part_result["NT_PR_EASY_N"] = toggles['nt'][LEVEL.EASY].get(N, np.nan)
        part_result["NT_PR_MED_N"] = toggles['nt'][LEVEL.MEDIUM].get(N, np.nan)
        part_result["NT_PR_HARD_N"] = toggles['nt'][LEVEL.HARD].get(N, np.nan)

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

        try:
            part_result["NT_OP_EASY_P"] = toggles['nt'][LEVEL.EASY][P]
        except KeyError:
            part_result["NT_OP_EASY_P"] = np.nan
        try:
            part_result["NT_OP_MEDIUM_P"] = toggles['nt'][LEVEL.MEDIUM][P]
        except KeyError:
            part_result["NT_OP_MEDIUM_P"] = np.nan
        try:
            part_result["NT_OP_HARD_P"] = toggles['nt'][LEVEL.HARD][P]
        except KeyError:
            part_result["NT_OP_HARD_P"] = np.nan

        part_result["NT_OP_EASY_N"] = toggles['nt'][LEVEL.EASY].get(N, np.nan)
        part_result["NT_OP_MED_N"] = toggles['nt'][LEVEL.MEDIUM].get(N, np.nan)
        part_result["NT_OP_HARD_N"] = toggles['nt'][LEVEL.HARD].get(N, np.nan)

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

        try:
            part_result["RT_PR_EASY_P"] = rt_pr[LEVEL.EASY][P]
        except KeyError:
            part_result["RT_PR_EASY_P"] = np.nan
        try:
            part_result["RT_PR_MED_P"] = rt_pr[LEVEL.MEDIUM][P]
        except KeyError:
            part_result["RT_PR_MED_P"] = np.nan
        try:
            part_result["RT_PR_HARD_P"] = rt_pr[LEVEL.HARD][P]
        except KeyError:
            part_result["RT_PR_HARD_P"] = np.nan

        part_result["RT_PR_EASY_N"] = rt_pr[LEVEL.EASY].get(N, np.nan)
        part_result["RT_PR_MED_N"] = rt_pr[LEVEL.MEDIUM].get(N, np.nan)
        part_result["RT_PR_HARD_N"] = rt_pr[LEVEL.HARD].get(N, np.nan)

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

        try:
            part_result["RT_OP_EASY_P"] = rt_op[LEVEL.EASY][P]
        except KeyError:
            part_result["RT_OP_EASY_P"] = np.nan
        try:
            part_result["RT_OP_MED_P"] = rt_op[LEVEL.MEDIUM][P]
        except KeyError:
            part_result["RT_OP_MED_P"] = np.nan
        try:
            part_result["RT_OP_HARD_P"] = rt_op[LEVEL.HARD][P]
        except:
            part_result["RT_OP_HARD_P"] = np.nan

        part_result["RT_OP_EASY_N"] = rt_op[LEVEL.EASY].get(N, np.nan)
        part_result["RT_OP_MED_N"] = rt_op[LEVEL.MEDIUM].get(N, np.nan)
        part_result["RT_OP_HARD_N"] = rt_op[LEVEL.HARD].get(N, np.nan)

        # relative time (RT_COR_EASY)
        rt_cor = {'EASY': 0, 'MEDIUM': 0, 'HARD': 0}
        rt_cor_p = {'EASY': 0, 'MEDIUM': 0, 'HARD': 0}
        rt_cor_n = {'EASY': 0, 'MEDIUM': 0, 'HARD': 0}
        for idx in beh_data.index:
            beh_item = beh_data.ix[idx]
            problem = problems[idx - 3]
            fix_item = fix_data[fix_data.block == idx]
            level = LEV_TO_LAB[str(problem['answers'])]

            cor_roi = [ROIS_ORDER[x] for x in where_in_list(problem['matrix_info'], 'D1')][0]
            fix_in_cor = in_roi(fix_item[['axp', 'ayp']], ROIS[cor_roi])
            item_dur_sum = fix_item[fix_in_cor]['dur'].sum()
            rt_cor[level] += item_dur_sum

            if idx in corr_beh.index:
                rt_cor_p[level] += item_dur_sum
            elif idx in err_beh.index:
                rt_cor_n[level] += item_dur_sum

        gb_full = beh_data.groupby('answers').sum()['rt'] * 1000.0
        gb_p = corr_beh.groupby('answers').sum()['rt'] * 1000.0
        gb_n = err_beh.groupby('answers').sum()['rt'] * 1000.0

        part_result["RT_COR_EASY"] = rt_cor['EASY'] / gb_full[LEVEL.EASY]
        part_result["RT_COR_MED"] = rt_cor['MEDIUM'] / gb_full[LEVEL.MEDIUM]
        part_result["RT_COR_HARD"] = rt_cor['HARD'] / gb_full[LEVEL.HARD]

        try:
            part_result["RT_COR_EASY_P"] = rt_cor_p['EASY'] / gb_p[LEVEL.EASY]
        except KeyError:
            part_result["RT_COR_EASY_P"] = np.nan
        try:
            part_result["RT_COR_MED_P"] = rt_cor_p['MEDIUM'] / gb_p[LEVEL.MEDIUM]
        except KeyError:
            part_result["RT_COR_MED_P"] = np.nan
        try:
            part_result["RT_COR_HARD_P"] = rt_cor_p['HARD'] / gb_p[LEVEL.HARD]
        except:
            part_result["RT_COR_HARD_P"] = np.nan

        try:
            part_result["RT_COR_EASY_N"] = rt_cor_n['EASY'] / gb_n.get(LEVEL.EASY, np.nan)
        except ZeroDivisionError:
            part_result['RT_COR_EASY_N'] = np.nan
        try:
            part_result["RT_COR_MED_N"] = rt_cor_n['MEDIUM'] / gb_n.get(LEVEL.MEDIUM, np.nan)
        except ZeroDivisionError:
            part_result["RT_COR_MED_N"] = np.nan
        try:
            part_result["RT_COR_HARD_N"] = rt_cor_n['HARD'] / gb_n.get(LEVEL.HARD, np.nan)
        except ZeroDivisionError:
            part_result["RT_COR_HARD_N"] = np.nan

        # relative time (RT_ERR_EASY)

        rt_err = {'EASY': 0, 'MEDIUM': 0, 'HARD': 0}
        rt_err_p = {'EASY': 0, 'MEDIUM': 0, 'HARD': 0}
        rt_err_n = {'EASY': 0, 'MEDIUM': 0, 'HARD': 0}
        for idx in beh_data.index:
            beh_item = beh_data.ix[idx]
            problem = problems[idx - 3]
            fix_item = fix_data[fix_data.block == idx]
            level = LEV_TO_LAB[str(problem['answers'])]

            err_roi = [where_in_list(problem['matrix_info'], x) for x in ['D2', 'D3', 'D4', 'D5', 'D6']]
            err_roi = [ROIS_ORDER[item] for sublist in err_roi for item in sublist]
            fix_in_err = pd.concat([in_roi(fix_item[['axp', 'ayp']], ROIS[x]) for x in err_roi], axis=1).any(axis=1)
            item_dur_sum = fix_item[fix_in_err]['dur'].sum()
            rt_err[level] += item_dur_sum

            if idx in corr_beh.index:
                rt_err_p[level] += item_dur_sum
            elif idx in err_beh.index:
                rt_err_n[level] += item_dur_sum

        gb_full = beh_data.groupby('answers').sum()['rt'] * 1000.0
        gb_p = corr_beh.groupby('answers').sum()['rt'] * 1000.0
        gb_n = err_beh.groupby('answers').sum()['rt'] * 1000.0

        part_result["RT_ERR_EASY"] = rt_err['EASY'] / gb_full[LEVEL.EASY]
        part_result["RT_ERR_MED"] = rt_err['MEDIUM'] / gb_full[LEVEL.MEDIUM]
        part_result["RT_ERR_HARD"] = rt_err['HARD'] / gb_full[LEVEL.HARD]

        try:
            part_result["RT_ERR_EASY_P"] = rt_err_p['EASY'] / gb_p[LEVEL.EASY]
        except KeyError:
            part_result["RT_ERR_EASY_P"] = np.nan
        try:
            part_result["RT_ERR_MED_P"] = rt_err_p['MEDIUM'] / gb_p[LEVEL.MEDIUM]
        except KeyError:
            part_result["RT_ERR_MED_P"] = np.nan
        try:
            part_result["RT_ERR_HARD_P"] = rt_err_p['HARD'] / gb_p[LEVEL.HARD]
        except KeyError:
            part_result["RT_ERR_HARD_P"] = np.nan

        try:
            part_result["RT_ERR_EASY_N"] = rt_err_n['EASY'] / gb_n.get(LEVEL.EASY, np.nan)
        except ZeroDivisionError:
            part_result["RT_ERR_EASY_N"] = np.nan
        try:
            part_result["RT_ERR_MED_N"] = rt_err_n['MEDIUM'] / gb_n.get(LEVEL.MEDIUM, np.nan)
        except ZeroDivisionError:
            part_result["RT_ERR_MED_N"] = np.nan
        try:
            part_result["RT_ERR_HARD_N"] = rt_err_n['HARD'] / gb_n.get(LEVEL.HARD, np.nan)
        except ZeroDivisionError:
            part_result["RT_ERR_HARD_N"] = np.nan

            # mean time (DUR)

        fix_in_pr = pd.concat([in_roi(fix_data[['axp', 'ayp']], ROIS[x]) for x in ['P1', 'P2', 'P3']], axis=1).any(
            axis=1)
        fix_in_pr_dur = fix_data[fix_in_pr].groupby('block')['dur'].apply(list)

        missing_blocks = set(range(1, 46)) - set(fix_in_pr_dur.index)
        for i in list(missing_blocks):
            fix_in_pr_dur.loc[i] = []
        fix_in_pr_dur.sort_index(inplace=True)
        fix_in_pr_dur = fix_in_pr_dur.reset_index()

        beh_data['fix_in_pr_dur'] = fix_in_pr_dur['dur']

        rt_pr = beh_data.groupby('answers')['fix_in_pr_dur'].apply(list)
        part_result["DUR_PR_EASY"] = np.mean([item for sublist in rt_pr[LEVEL.EASY] for item in sublist])
        part_result["DUR_PR_MEDIUM"] = np.mean([item for sublist in rt_pr[LEVEL.MEDIUM] for item in sublist])
        part_result["DUR_PR_HARD"] = np.mean([item for sublist in rt_pr[LEVEL.HARD] for item in sublist])

        # DUR_SE/BE/CON_MED
        dur_med = {'SE': [], 'BE': [], 'CON': []}
        for idx in beh_data.index:
            beh_item = beh_data.ix[idx]
            problem = problems[idx - 3]
            fix_item = fix_data[fix_data.block == idx]
            level = LEV_TO_LAB[str(problem['answers'])]
            if level != 'MEDIUM':
                continue

            se_roi = [ROIS_ORDER[x] for x in where_in_list(problem['matrix_info'], 'D3')]
            fix_in_se = pd.concat([in_roi(fix_item[['axp', 'ayp']], ROIS[x]) for x in se_roi], axis=1).any(axis=1)
            dur_med['SE'].extend(fix_item[fix_in_se]['dur'])

            be_roi = [ROIS_ORDER[x] for x in where_in_list(problem['matrix_info'], 'D4')]
            fix_in_be = pd.concat([in_roi(fix_item[['axp', 'ayp']], ROIS[x]) for x in be_roi], axis=1).any(axis=1)
            dur_med['BE'].extend(fix_item[fix_in_be]['dur'])

            con_roi = ROIS_ORDER[where_in_list(problem['matrix_info'], 'D6')[0]]
            fix_in_con = in_roi(fix_item[['axp', 'ayp']], ROIS[con_roi])
            dur_med['CON'].extend(fix_item[fix_in_con]['dur'])

        part_result["DUR_SE_MED"] = np.mean(dur_med['SE'])
        part_result["DUR_BE_MED"] = np.mean(dur_med['BE'])
        part_result["DUR_CON_MED"] = np.mean(dur_med['CON'])

        # dur time (DUR_PR_P/N)

        rt_pr = beh_data.groupby(['answers', 'corr'])['fix_in_pr_dur'].apply(list)

        try:
            part_result["DUR_PR_EASY_P"] = np.mean([item for sublist in rt_pr[LEVEL.EASY][P] for item in sublist])
        except KeyError:
            part_result["DUR_PR_EASY_P"] = np.nan
        try:
            part_result["DUR_PR_MED_P"] = np.mean([item for sublist in rt_pr[LEVEL.MEDIUM][P] for item in sublist])
        except KeyError:
            part_result["DUR_PR_MED_P"] = np.nan
        try:
            part_result["DUR_PR_HARD_P"] = np.mean([item for sublist in rt_pr[LEVEL.HARD][P] for item in sublist])
        except KeyError:
            part_result["DUR_PR_HARD_P"] = np.nan

        try:
            part_result["DUR_PR_EASY_N"] = np.mean([item for sublist in rt_pr[LEVEL.EASY][N] for item in sublist])
        except KeyError:
            part_result["DUR_PR_EASY_N"] = np.nan
        try:
            part_result["DUR_PR_MED_N"] = np.mean([item for sublist in rt_pr[LEVEL.MEDIUM][N] for item in sublist])
        except KeyError:
            part_result["DUR_PR_MED_N"] = np.nan
        try:
            part_result["DUR_PR_HARD_N"] = np.mean([item for sublist in rt_pr[LEVEL.HARD][N] for item in sublist])
        except KeyError:
            part_result["DUR_PR_HARD_N"] = np.nan
            # relative time (DUR_OP)
        fix_in_op = pd.concat([in_roi(fix_data[['axp', 'ayp']], ROIS[x]) for x in 'ABCDEF'], axis=1).any(axis=1)

        beh_data['fix_in_op_dur'] = fix_data[fix_in_op].groupby('block')['dur'].apply(list)
        gb = beh_data.groupby('answers')['fix_in_op_dur'].apply(list)

        part_result["DUR_OP_EASY"] = np.mean(
            [item for sublist in (map(lambda x: [] if x is np.nan else x, gb[LEVEL.EASY])) for item in sublist])
        part_result["DUR_OP_MEDIUM"] = np.mean(
            [item for sublist in (map(lambda x: [] if x is np.nan else x, gb[LEVEL.MEDIUM])) for item in sublist])
        part_result["DUR_OP_HARD"] = np.mean(
            [item for sublist in (map(lambda x: [] if x is np.nan else x, gb[LEVEL.HARD])) for item in sublist])

        # dur time (DUR_OP)

        gb = beh_data.groupby(['answers', 'corr'])['fix_in_op_dur'].apply(list)

        try:
            part_result["DUR_OP_EASY_P"] = np.mean(
                [item for sublist in (map(lambda x: [] if x is np.nan else x, gb[LEVEL.EASY][P])) for item in sublist])
        except KeyError:
            part_result["DUR_OP_EASY_P"] = np.nan
        try:
            part_result["DUR_OP_MED_P"] = np.mean(
                [item for sublist in (map(lambda x: [] if x is np.nan else x, gb[LEVEL.MEDIUM][P])) for item in
                 sublist])
        except KeyError:
            part_result["DUR_OP_MED_P"] = np.nan
        try:
            part_result["DUR_OP_HARD_P"] = np.mean(
                [item for sublist in (map(lambda x: [] if x is np.nan else x, gb[LEVEL.HARD][P])) for item in sublist])
        except KeyError:
            part_result["DUR_OP_HARD_P"] = np.nan

        try:
            part_result["DUR_OP_EASY_N"] = np.mean(
                [item for sublist in (map(lambda x: [] if x is np.nan else x, gb[LEVEL.EASY][N])) for item in sublist])
        except KeyError:
            part_result["DUR_OP_EASY_N"] = np.nan
        try:
            part_result["DUR_OP_MED_N"] = np.mean(
                [item for sublist in (map(lambda x: [] if x is np.nan else x, gb[LEVEL.MEDIUM][N])) for item in
                 sublist])
        except KeyError:
            part_result["DUR_OP_MED_N"] = np.nan
        try:
            part_result["DUR_OP_HARD_N"] = np.mean(
                [item for sublist in (map(lambda x: [] if x is np.nan else x, gb[LEVEL.HARD][N])) for item in sublist])
        except KeyError:
            part_result["DUR_OP_HARD_N"] = np.nan
        # dur time (DUR_COR_EASY)
        dur_cor = {'EASY': [], 'MEDIUM': [], 'HARD': []}
        dur_cor_p = {'EASY': [], 'MEDIUM': [], 'HARD': []}
        dur_cor_n = {'EASY': [], 'MEDIUM': [], 'HARD': []}
        for idx in beh_data.index:
            beh_item = beh_data.ix[idx]
            problem = problems[idx - 3]
            fix_item = fix_data[fix_data.block == idx]
            level = LEV_TO_LAB[str(problem['answers'])]

            cor_roi = [ROIS_ORDER[x] for x in where_in_list(problem['matrix_info'], 'D1')][0]
            fix_in_cor = in_roi(fix_item[['axp', 'ayp']], ROIS[cor_roi])
            item_dur_sum = fix_item[fix_in_cor]['dur']
            dur_cor[level].extend(item_dur_sum)

            if idx in corr_beh.index:
                dur_cor_p[level].extend(item_dur_sum)
            elif idx in err_beh.index:
                dur_cor_n[level].extend(item_dur_sum)

        part_result["DUR_COR_EASY"] = np.mean(dur_cor['EASY'])
        part_result["DUR_COR_MED"] = np.mean(dur_cor['MEDIUM'])
        part_result["DUR_COR_HARD"] = np.mean(dur_cor['HARD'])

        part_result["DUR_COR_EASY_P"] = np.mean(dur_cor_p['EASY'])
        part_result["DUR_COR_MED_P"] = np.mean(dur_cor_p['MEDIUM'])
        part_result["DUR_COR_HARD_P"] = np.mean(dur_cor_p['HARD'])

        part_result["DUR_COR_EASY_N"] = np.mean(dur_cor_n['EASY'])
        part_result["DUR_COR_MED_N"] = np.mean(dur_cor_n['MEDIUM'])
        part_result["DUR_COR_HARD_N"] = np.mean(dur_cor_n['HARD'])

        # relative time (DUR_ERR_EASY)

        dur_err = {'EASY': [], 'MEDIUM': [], 'HARD': []}
        dur_err_p = {'EASY': [], 'MEDIUM': [], 'HARD': []}
        dur_err_n = {'EASY': [], 'MEDIUM': [], 'HARD': []}
        for idx in beh_data.index:
            beh_item = beh_data.ix[idx]
            problem = problems[idx - 3]
            fix_item = fix_data[fix_data.block == idx]
            level = LEV_TO_LAB[str(problem['answers'])]

            err_roi = [where_in_list(problem['matrix_info'], x) for x in ['D2', 'D3', 'D4', 'D5', 'D6']]
            err_roi = [ROIS_ORDER[item] for sublist in err_roi for item in sublist]
            fix_in_err = pd.concat([in_roi(fix_item[['axp', 'ayp']], ROIS[x]) for x in err_roi], axis=1).any(axis=1)
            item_dur_sum = fix_item[fix_in_err]['dur']
            dur_err[level].extend(item_dur_sum)

            if idx in corr_beh.index:
                dur_err_p[level].extend(item_dur_sum)
            elif idx in err_beh.index:
                dur_err_n[level].extend(item_dur_sum)

        part_result["DUR_ERR_EASY"] = np.mean(dur_err['EASY'])
        part_result["DUR_ERR_MED"] = np.mean(dur_err['MEDIUM'])
        part_result["DUR_ERR_HARD"] = np.mean(dur_err['HARD'])

        part_result["DUR_ERR_EASY_P"] = np.mean(dur_err_p['EASY'])
        part_result["DUR_ERR_MED_P"] = np.mean(dur_err_p['MEDIUM'])
        part_result["DUR_ERR_HARD_P"] = np.mean(dur_err_p['HARD'])

        part_result["DUR_ERR_EASY_N"] = np.mean(dur_err_n['EASY'])
        part_result["DUR_ERR_MED_N"] = np.mean(dur_err_n['MEDIUM'])
        part_result["DUR_ERR_HARD_N"] = np.mean(dur_err_n['HARD'])

        # sum time (FIX)

        fix_in_pr = pd.concat([in_roi(fix_data[['axp', 'ayp']], ROIS[x]) for x in ['P1', 'P2', 'P3']], axis=1).any(
            axis=1)
        fix_in_pr_dur = fix_data[fix_in_pr].groupby('block')['dur'].apply(list)

        missing_blocks = set(range(1, 46)) - set(fix_in_pr_dur.index)
        for i in list(missing_blocks):
            fix_in_pr_dur.loc[i] = []
        fix_in_pr_dur.sort_index(inplace=True)
        fix_in_pr_dur = fix_in_pr_dur.reset_index()

        beh_data['fix_in_pr_dur'] = fix_in_pr_dur['dur']

        rt_pr = beh_data.groupby('answers')['fix_in_pr_dur'].apply(list)
        part_result["FIX_PR_EASY"] = np.sum([item for sublist in rt_pr[LEVEL.EASY] for item in sublist])
        part_result["FIX_PR_MEDIUM"] = np.sum([item for sublist in rt_pr[LEVEL.MEDIUM] for item in sublist])
        part_result["FIX_PR_HARD"] = np.sum([item for sublist in rt_pr[LEVEL.HARD] for item in sublist])

        # DUR_SE/BE/CON_MED
        dur_med = {'SE': [], 'BE': [], 'CON': []}
        for idx in beh_data.index:
            beh_item = beh_data.ix[idx]
            problem = problems[idx - 3]
            fix_item = fix_data[fix_data.block == idx]
            level = LEV_TO_LAB[str(problem['answers'])]
            if level != 'MEDIUM':
                continue

            se_roi = [ROIS_ORDER[x] for x in where_in_list(problem['matrix_info'], 'D3')]
            fix_in_se = pd.concat([in_roi(fix_item[['axp', 'ayp']], ROIS[x]) for x in se_roi], axis=1).any(axis=1)
            dur_med['SE'].extend(fix_item[fix_in_se]['dur'])

            be_roi = [ROIS_ORDER[x] for x in where_in_list(problem['matrix_info'], 'D4')]
            fix_in_be = pd.concat([in_roi(fix_item[['axp', 'ayp']], ROIS[x]) for x in be_roi], axis=1).any(axis=1)
            dur_med['BE'].extend(fix_item[fix_in_be]['dur'])

            con_roi = ROIS_ORDER[where_in_list(problem['matrix_info'], 'D6')[0]]
            fix_in_con = in_roi(fix_item[['axp', 'ayp']], ROIS[con_roi])
            dur_med['CON'].extend(fix_item[fix_in_con]['dur'])

        part_result["FIX_SE_MED"] = np.sum(dur_med['SE'])
        part_result["FIX_BE_MED"] = np.sum(dur_med['BE'])
        part_result["FIX_CON_MED"] = np.sum(dur_med['CON'])

        # dur time (DUR_PR_P/N)

        rt_pr = beh_data.groupby(['answers', 'corr'])['fix_in_pr_dur'].apply(list)

        try:
            part_result["FIX_PR_EASY_P"] = np.sum([item for sublist in rt_pr[LEVEL.EASY][P] for item in sublist])
        except KeyError:
            part_result["FIX_PR_EASY_P"] = np.nan
        try:
            part_result["FIX_PR_MED_P"] = np.sum([item for sublist in rt_pr[LEVEL.MEDIUM][P] for item in sublist])
        except KeyError:
            part_result["FIX_PR_MED_P"] = np.nan
        try:
            part_result["FIX_PR_HARD_P"] = np.sum([item for sublist in rt_pr[LEVEL.HARD][P] for item in sublist])
        except KeyError:
            part_result["FIX_PR_HARD_P"] = np.nan

        try:
            part_result["FIX_PR_EASY_N"] = np.sum([item for sublist in rt_pr[LEVEL.EASY][N] for item in sublist])
        except KeyError:
            part_result["FIX_PR_EASY_N"] = np.nan
        try:
            part_result["FIX_PR_MED_N"] = np.sum([item for sublist in rt_pr[LEVEL.MEDIUM][N] for item in sublist])
        except KeyError:
            part_result["FIX_PR_MED_N"] = np.nan
        try:
            part_result["FIX_PR_HARD_N"] = np.sum([item for sublist in rt_pr[LEVEL.HARD][N] for item in sublist])
        except KeyError:
            part_result["FIX_PR_HARD_N"] = np.nan
            # relative time (DUR_OP)
        fix_in_op = pd.concat([in_roi(fix_data[['axp', 'ayp']], ROIS[x]) for x in 'ABCDEF'], axis=1).any(axis=1)

        beh_data['fix_in_op_dur'] = fix_data[fix_in_op].groupby('block')['dur'].apply(list)

        gb = beh_data.groupby('answers')['fix_in_op_dur'].apply(list)

        try:
            part_result["FIX_OP_EASY"] = np.sum(
                [item for sublist in (map(lambda x: [] if x is np.nan else x, gb[LEVEL.EASY])) for item in sublist])
        except KeyError:
            part_result["FIX_OP_EASY"] = np.nan
        try:
            part_result["FIX_OP_MEDIUM"] = np.sum(
                [item for sublist in (map(lambda x: [] if x is np.nan else x, gb[LEVEL.MEDIUM])) for item in sublist])
        except KeyError:
            part_result["FIX_OP_MEDIUM"] = np.nan
        try:
            part_result["FIX_OP_HARD"] = np.sum(
                [item for sublist in (map(lambda x: [] if x is np.nan else x, gb[LEVEL.HARD])) for item in sublist])
        except KeyError:
            part_result["FIX_OP_HARD"] = np.nan
            # dur time (DUR_OP)

        gb = beh_data.groupby(['answers', 'corr'])['fix_in_op_dur'].apply(list)

        try:
            part_result["FIX_OP_EASY_P"] = np.sum(
                [item for sublist in (map(lambda x: [] if x is np.nan else x, gb[LEVEL.EASY][P])) for item in sublist])
        except KeyError:
            part_result["FIX_OP_EASY_P"] = np.nan
        try:
            part_result["FIX_OP_MED_P"] = np.sum(
                [item for sublist in (map(lambda x: [] if x is np.nan else x, gb[LEVEL.MEDIUM][P])) for item in
                 sublist])
        except KeyError:
            part_result["FIX_OP_MED_P"] = np.nan
        try:
            part_result["FIX_OP_HARD_P"] = np.sum(
                [item for sublist in (map(lambda x: [] if x is np.nan else x, gb[LEVEL.HARD][P])) for item in sublist])
        except KeyError:
            part_result["FIX_OP_HARD_P"] = np.nan

        try:
            part_result["FIX_OP_EASY_N"] = np.sum(
                [item for sublist in (map(lambda x: [] if x is np.nan else x, gb[LEVEL.EASY][N])) for item in sublist])
        except KeyError:
            part_result["FIX_OP_EASY_N"] = np.nan
        try:
            part_result["FIX_OP_MED_N"] = np.sum(
                [item for sublist in (map(lambda x: [] if x is np.nan else x, gb[LEVEL.MEDIUM][N])) for item in
                 sublist])
        except KeyError:
            part_result["FIX_OP_MED_N"] = np.nan
        try:
            part_result["FIX_OP_HARD_N"] = np.sum(
                [item for sublist in (map(lambda x: [] if x is np.nan else x, gb[LEVEL.HARD][N])) for item in sublist])
        except KeyError:
            part_result["FIX_OP_HARD_N"] = np.nan
            # dur time (DUR_COR_EASY)
        dur_cor = {'EASY': [], 'MEDIUM': [], 'HARD': []}
        dur_cor_p = {'EASY': [], 'MEDIUM': [], 'HARD': []}
        dur_cor_n = {'EASY': [], 'MEDIUM': [], 'HARD': []}
        for idx in beh_data.index:
            beh_item = beh_data.ix[idx]
            problem = problems[idx - 3]
            fix_item = fix_data[fix_data.block == idx]
            level = LEV_TO_LAB[str(problem['answers'])]

            cor_roi = [ROIS_ORDER[x] for x in where_in_list(problem['matrix_info'], 'D1')][0]
            fix_in_cor = in_roi(fix_item[['axp', 'ayp']], ROIS[cor_roi])
            item_dur_sum = fix_item[fix_in_cor]['dur']
            dur_cor[level].extend(item_dur_sum)

            if idx in corr_beh.index:
                dur_cor_p[level].extend(item_dur_sum)
            elif idx in err_beh.index:
                dur_cor_n[level].extend(item_dur_sum)

        part_result["FIX_COR_EASY"] = np.sum(dur_cor['EASY'])
        part_result["FIX_COR_MED"] = np.sum(dur_cor['MEDIUM'])
        part_result["FIX_COR_HARD"] = np.sum(dur_cor['HARD'])

        part_result["FIX_COR_EASY_P"] = np.sum(dur_cor_p['EASY'])
        part_result["FIX_COR_MED_P"] = np.sum(dur_cor_p['MEDIUM'])
        part_result["FIX_COR_HARD_P"] = np.sum(dur_cor_p['HARD'])

        part_result["FIX_COR_EASY_N"] = np.sum(dur_cor_n['EASY'])
        part_result["FIX_COR_MED_N"] = np.sum(dur_cor_n['MEDIUM'])
        part_result["FIX_COR_HARD_N"] = np.sum(dur_cor_n['HARD'])

        # relative time (DUR_ERR_EASY)

        dur_err = {'EASY': [], 'MEDIUM': [], 'HARD': []}
        dur_err_p = {'EASY': [], 'MEDIUM': [], 'HARD': []}
        dur_err_n = {'EASY': [], 'MEDIUM': [], 'HARD': []}
        for idx in beh_data.index:
            beh_item = beh_data.ix[idx]
            problem = problems[idx - 3]
            fix_item = fix_data[fix_data.block == idx]
            level = LEV_TO_LAB[str(problem['answers'])]

            err_roi = [where_in_list(problem['matrix_info'], x) for x in ['D2', 'D3', 'D4', 'D5', 'D6']]
            err_roi = [ROIS_ORDER[item] for sublist in err_roi for item in sublist]
            fix_in_err = pd.concat([in_roi(fix_item[['axp', 'ayp']], ROIS[x]) for x in err_roi], axis=1).any(axis=1)
            item_dur_sum = fix_item[fix_in_err]['dur']
            dur_err[level].extend(item_dur_sum)

            if idx in corr_beh.index:
                dur_err_p[level].extend(item_dur_sum)
            elif idx in err_beh.index:
                dur_err_n[level].extend(item_dur_sum)

        part_result["FIX_ERR_EASY"] = np.sum(dur_err['EASY'])
        part_result["FIX_ERR_MED"] = np.sum(dur_err['MEDIUM'])
        part_result["FIX_ERR_HARD"] = np.sum(dur_err['HARD'])

        part_result["FIX_ERR_EASY_P"] = np.sum(dur_err_p['EASY'])
        part_result["FIX_ERR_MED_P"] = np.sum(dur_err_p['MEDIUM'])
        part_result["FIX_ERR_HARD_P"] = np.sum(dur_err_p['HARD'])

        part_result["FIX_ERR_EASY_N"] = np.sum(dur_err_n['EASY'])
        part_result["FIX_ERR_MED_N"] = np.sum(dur_err_n['MEDIUM'])
        part_result["FIX_ERR_HARD_N"] = np.sum(dur_err_n['HARD'])

        RESULTS.append(part_result)

# Save results
res = pd.DataFrame(RESULTS)
pd.DataFrame(RESULTS).to_csv('metrics.csv', na_rep=np.nan)
