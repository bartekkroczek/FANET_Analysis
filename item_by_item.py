#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  4 14:09:38 2018

@author: bkroczek
"""

import pandas as pd
import os
import yaml
import collections
import numpy as np
from tqdm import tqdm

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

with tqdm(total=len(sacc_files)) as pbar:
    for part_id in sacc_files:
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
            part_result = collections.OrderedDict()
            sacc_item = sacc_data[sacc_data.block == idx]
            fix_item = fix_data[fix_data.block == idx]
            raw_item = raw_data[raw_data.block == idx]
            beh_item = beh_data.ix[idx - 1]
            problem = problems[idx - 1]

            if beh_item.rt == -1:  # no ans selected
                continue
            part_result['Part_id'] = part_id
            # correctness 
            part_result['AVG_CORR'] = beh_data[beh_data.exp == 'experiment']['corr'].mean()
            # WMC
            part_result['WMC'] = float(index_data.WMC.tolist()[0].replace(',', '.'))
            part_result['GF'] = index_data.gf.values[0]
            part_result['KOL'] = idx
            part_result['WAR'] = LEV_TO_LAB[beh_item.answers]
            # N
            part_result['ZLOZ'] = problem['rel']
            # WYBrana odpowiedź w danym itemie (cor, be, se, con)
            level = LEV_TO_LAB[beh_item.answers]
            choosed_option = beh_item.choosed_option
            if choosed_option == '-1':
                continue
            d = {'EASY': {'D1': 'corr', 'D4': 'se', 'D5': 'be', 'D6': 'con'},
                 'MEDIUM': {'D1': 'corr', 'D3': 'se', 'D4': 'be', 'D6': 'con'},
                 'HARD': {'D1': 'corr', 'D2': 'se', 'D3': 'be', 'D6': 'con'}}
            part_result['WYB'] = d[level][choosed_option]
            part_result['LAT'] = beh_item['rt']
            # NT

            sacc_start_in_question_area = pd.concat([in_roi(sacc_item[['sxp', 'syp']], ROIS['P1']),
                                                     in_roi(sacc_item[['sxp', 'syp']], ROIS['P2']),
                                                     in_roi(sacc_item[['sxp', 'syp']], ROIS['P3'])], axis=1).any(axis=1)

            sacc_ends_in_matrix_area = pd.concat([in_roi(sacc_item[['exp', 'eyp']], ROIS['A']),
                                                  in_roi(sacc_item[['exp', 'eyp']], ROIS['B']),
                                                  in_roi(sacc_item[['exp', 'eyp']], ROIS['C']),
                                                  in_roi(sacc_item[['exp', 'eyp']], ROIS['D']),
                                                  in_roi(sacc_item[['exp', 'eyp']], ROIS['E']),
                                                  in_roi(sacc_item[['exp', 'eyp']], ROIS['F'])], axis=1).any(axis=1)

            sacc_starts_in_question_area_and_ends_in_matrix_area = pd.concat(
                [sacc_start_in_question_area, sacc_ends_in_matrix_area], axis=1).all(axis=1)

            sacc_start_in_matrix_area = pd.concat([in_roi(sacc_item[['sxp', 'syp']], ROIS['A']),
                                                   in_roi(sacc_item[['sxp', 'syp']], ROIS['B']),
                                                   in_roi(sacc_item[['sxp', 'syp']], ROIS['C']),
                                                   in_roi(sacc_item[['sxp', 'syp']], ROIS['D']),
                                                   in_roi(sacc_item[['sxp', 'syp']], ROIS['E']),
                                                   in_roi(sacc_item[['sxp', 'syp']], ROIS['F'])], axis=1).any(axis=1)

            sacc_ends_in_question_area = pd.concat([in_roi(sacc_item[['exp', 'eyp']], ROIS['P1']),
                                                    in_roi(sacc_item[['exp', 'eyp']], ROIS['P2']),
                                                    in_roi(sacc_item[['exp', 'eyp']], ROIS['P3'])], axis=1).any(axis=1)

            sacc_starts_in_matrix_area_and_ends_in_question_area = pd.concat(
                [sacc_start_in_matrix_area, sacc_ends_in_question_area], axis=1).all(axis=1)

            toggled_sacc = pd.concat([sacc_starts_in_matrix_area_and_ends_in_question_area,
                                      sacc_starts_in_question_area_and_ends_in_matrix_area], axis=1).any(axis=1)
            part_result['NT'] = toggled_sacc.sum()
            # RTPR w danym itemie (dawne RTM, czyli relative time on PRoblem)
            # # relative time on matrix (RTM) 
            # summed duration of all fixations within the problem area (time on matrix) divided by total response time. 

            fix_in_matrix_area = pd.concat([in_roi(fix_item[['axp', 'ayp']], ROIS['P1']),
                                            in_roi(fix_item[['axp', 'ayp']], ROIS['P2']),
                                            in_roi(fix_item[['axp', 'ayp']], ROIS['P3'])], axis=1).any(axis=1)

            part_result['RTPR'] = fix_item[fix_in_matrix_area]['dur'].sum() / (beh_item['rt'] * 1000.0)

            fix_in_matrix_area = pd.concat([in_roi(fix_item[['axp', 'ayp']], ROIS['A']),
                                            in_roi(fix_item[['axp', 'ayp']], ROIS['B']),
                                            in_roi(fix_item[['axp', 'ayp']], ROIS['C']),
                                            in_roi(fix_item[['axp', 'ayp']], ROIS['D']),
                                            in_roi(fix_item[['axp', 'ayp']], ROIS['E']),
                                            in_roi(fix_item[['axp', 'ayp']], ROIS['F'])], axis=1).any(axis=1)

            part_result['RTOP'] = fix_item[fix_in_matrix_area]['dur'].sum() / (beh_item['rt'] * 1000.0)


            #  AVG_DUR_COR (czyli średni czas jednej fiksacji na opcji poprawnej)

            def where_in_list(where, what):
                bound = len(where)
                tmp = map(lambda x: x['name'] == what, where)
                return [idx for idx, _ in zip(range(bound), tmp) if _]


            ROIS_ORDER = ['P1', 'P2', 'P3', 'A', 'B', 'C', 'D', 'E', 'F']

            # CORR == D1  
            roi = ROIS_ORDER[where_in_list(problem['matrix_info'], 'D1')[0]]

            fix_cor = fix_item[in_roi(fix_item[['axp', 'ayp']], ROIS[roi])]['dur']

            # AVG_DUR_SE (czyli średni czas jednej fiksacji na opcjach se, skoro to średnia to nie trzeba dzielić przez 2)

            se = {'EASY': 'D4', 'MEDIUM': 'D3', 'HARD': 'D2'}[level]
            be = {'EASY': 'D5', 'MEDIUM': 'D4', 'HARD': 'D3'}[level]

            se = [ROIS_ORDER[x] for x in where_in_list(problem['matrix_info'], se)]
            be = [ROIS_ORDER[x] for x in where_in_list(problem['matrix_info'], be)]

            fix_in_se = pd.concat(
                [in_roi(fix_item[['axp', 'ayp']], ROIS[roi]) for roi in se],
                axis=1).any(axis=1)

            fix_in_be = pd.concat(
                [in_roi(fix_item[['axp', 'ayp']], ROIS[roi]) for roi in be],
                axis=1).any(axis=1)

            fix_se = fix_item[fix_in_se]['dur']
            fix_be = fix_item[fix_in_be]['dur']

            roi = ROIS_ORDER[where_in_list(problem['matrix_info'], 'D6')[0]]
            fix_con = fix_item[in_roi(fix_item[['axp', 'ayp']], ROIS[roi])]['dur']

            part_result['AVG_DUR_COR'] = fix_cor.mean()
            part_result['AVG_DUR_SE'] = fix_se.mean()
            part_result['AVG_DUR_BE'] = fix_be.mean()
            part_result['AVG_DUR_CON'] = fix_con.mean()

            part_result['SUM_FIX_COR'] = fix_cor.sum()
            part_result['SUM_FIX_SE'] = fix_se.sum() / 2.0
            part_result['SUM_FIX_BE'] = fix_be.sum() / 2.0
            part_result['SUM_FIX_CON'] = fix_con.sum()
            RESULTS.append(part_result)

res = pd.DataFrame(RESULTS)
res.to_csv('item_wise.csv')
res.to_excel('item_wise.xlsx')
