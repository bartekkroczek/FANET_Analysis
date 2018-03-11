# coding: utf-8

import pandas as pd
import os
from os.path import join
import yaml
from collections import OrderedDict, defaultdict
import numpy as np
from tqdm import tqdm

class LEVEL(object):
    TRAINING = '[1, 2, 3, 4, 5, 6]'
    EASY = '[1, 4, 4, 5, 5, 6]'
    MEDIUM = '[1, 3, 3, 4, 4, 6]'
    HARD = '[1, 2, 2, 3, 3, 6]'

# # Load data

SACC_FOLDER = join('..','Dane trackingowe', 'sacc')
BEH_FOLDER = join('..','results', 'beh')
FIX_FOLDER = join('..','Dane trackingowe', 'fix')
RAW_FOLDER = join('..', 'Dane trackingowe', 'raw')
YAML_FOLDER = join('..', 'results', 'yaml')

sacc_files = os.listdir(SACC_FOLDER)
sacc_files = [x for x in sacc_files if x.endswith('.csv')]
beh_files = os.listdir(BEH_FOLDER)
fix_files = os.listdir(FIX_FOLDER)
raw_files = os.listdir(RAW_FOLDER)
yaml_files = os.listdir(YAML_FOLDER)


RESULTS = list()

#%% ROI's definitions

ROIS = {
    'P1' : [(-  15,  280), ( 235,   30)],
    'P2' : [(-  15,  680), ( 235,  430)],
    'P3' : [(-  15, 1080), ( 235,  830)],
    'A'  : [(+ 465,  520), ( 715,  270)],
    'B'  : [(+ 955,  520), (1205,  270)],
    'C'  : [(+1445,  520), (1695,  270)],
    'D'  : [(+ 465,  870), ( 715,  630)],
    'E'  : [(+ 955,  870), (1205,  630)],
    'F'  : [(+1445,  870), (1695,  630)]
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
              

import random
sacc_files = [random.choice(sacc_files)]
#%%
    
with tqdm(total = len(sacc_files)) as pbar:
    for part_id in sacc_files:
        pbar.set_postfix(file = part_id)
        pbar.update(1)
        part_result = OrderedDict()
        part_id = part_id.split('_')[0]
        sacc_data = pd.read_csv(join(SACC_FOLDER, part_id + '_sacc.csv')).drop('Unnamed: 0', 1)
        beh_data = pd.read_csv(join(BEH_FOLDER, part_id + '_beh.csv'))
        fix_data = pd.read_csv(join(FIX_FOLDER, part_id + '_fix.csv')).drop('Unnamed: 0', 1)

        
        if part_id in ['12MALE21', '14FEMALE19', '62FEMALE39', '83MALE27', '130MALE18', '142FEMALE29', '165FEMALE20']: # no Unnamed column
            raw_data = pd.read_csv(join(RAW_FOLDER, part_id + '_raw.csv'))
        else:
            raw_data = pd.read_csv(join(RAW_FOLDER, part_id + '_raw.csv')).drop('Unnamed: 0', 1)
        yaml_data = yaml.load(open(join(YAML_FOLDER, part_id + '.yaml'), 'r'))

        if 'FEMALE' in part_id:
            part_result['Part_id'] = part_id.split('F')[0]
        else:
            part_result['Part_id'] = part_id.split('M')[0]
    
        # # toggle rate (TR) 
        # based on the number of saccades running either way between the matrix area and the response area (number of toggles) divided by total response time in seconds (i.e., toggles per second). 
    
        sacc_start_in_question_area = pd.concat([in_roi(sacc_data[['sxp', 'syp']], ROIS['P1']),
                                                 in_roi(sacc_data[['sxp', 'syp']], ROIS['P2']),
                                                 in_roi(sacc_data[['sxp', 'syp']], ROIS['P3'])], axis=1).any(axis = 1)
    
        sacc_ends_in_matrix_area = pd.concat([in_roi(sacc_data[['exp', 'eyp']], ROIS['A']),
                                              in_roi(sacc_data[['exp', 'eyp']], ROIS['B']),
                                              in_roi(sacc_data[['exp', 'eyp']], ROIS['C']),
                                              in_roi(sacc_data[['exp', 'eyp']], ROIS['D']),
                                              in_roi(sacc_data[['exp', 'eyp']], ROIS['E']),
                                              in_roi(sacc_data[['exp', 'eyp']], ROIS['F'])], axis=1).any(axis = 1)
    
        sacc_starts_in_question_area_and_ends_in_matrix_area = pd.concat([sacc_start_in_question_area, sacc_ends_in_matrix_area], axis = 1).all(axis = 1)
    
        sacc_start_in_matrix_area = pd.concat([in_roi(sacc_data[['sxp', 'syp']], ROIS['A']),
                                               in_roi(sacc_data[['sxp', 'syp']], ROIS['B']),
                                               in_roi(sacc_data[['sxp', 'syp']], ROIS['C']),
                                               in_roi(sacc_data[['sxp', 'syp']], ROIS['D']),
                                               in_roi(sacc_data[['sxp', 'syp']], ROIS['E']),
                                               in_roi(sacc_data[['sxp', 'syp']], ROIS['F'])], axis=1).any(axis = 1)
    
        sacc_ends_in_question_area = pd.concat([in_roi(sacc_data[['exp', 'eyp']], ROIS['P1']),
                                                in_roi(sacc_data[['exp', 'eyp']], ROIS['P2']),
                                                in_roi(sacc_data[['exp', 'eyp']], ROIS['P3'])], axis=1).any(axis = 1)
    
        sacc_starts_in_matrix_area_and_ends_in_question_area = pd.concat([sacc_start_in_matrix_area, sacc_ends_in_question_area], axis = 1).all(axis = 1)
    
    
        toggled_sacc = pd.concat([sacc_starts_in_matrix_area_and_ends_in_question_area, sacc_starts_in_question_area_and_ends_in_matrix_area], axis = 1).any(axis = 1)
        toggled_sacc = sacc_data[toggled_sacc]
    
        
        # stime used just for counting how many events occurs in any particular block
        ts = toggled_sacc.groupby('block').count()['stime']
        del toggled_sacc
        missing_blocks = set(range(1, 46)) - set(ts.index)
        
        for i in list(missing_blocks):
            ts.loc[i] = 0 
        ts.sort_index(inplace=True)
        ts = ts.reset_index()
        beh_data['no_toggles'] = ts['stime']
        del ts
        toggles = beh_data.groupby('answers').sum()
        
        #%%
        part_result["NT_EASY"] = toggles['no_toggles'][LEVEL.EASY]
        part_result["NT_MEDIUM"] = toggles['no_toggles'][LEVEL.MEDIUM]
        part_result["NT_HARD"] = toggles['no_toggles'][LEVEL.HARD]
        
        toggle_rate = toggles['no_toggles'] / toggles['rt']
        
        part_result["TR_EASY"] = toggle_rate[LEVEL.EASY]
        part_result["TR_MEDIUM"] = toggle_rate[LEVEL.MEDIUM]
        part_result["TR_HARD"] = toggle_rate[LEVEL.HARD]
        del toggles
        
        # # relative time on matrix (RTM) 
        # summed duration of all fixations within the matrix area (time on matrix) divided by total response time. 
    
        fix_in_matrix_area = pd.concat([in_roi(fix_data[['axp', 'ayp']], ROIS['P1']),
                                        in_roi(fix_data[['axp', 'ayp']], ROIS['P2']),
                                        in_roi(fix_data[['axp', 'ayp']], ROIS['P3'])], axis = 1).any(axis = 1)
    
    
        fix_in_matrix_area_on_block_dur = fix_data[fix_in_matrix_area].groupby('block').sum()['dur']
    
        missing_blocks = set(range(1, 46)) - set(fix_in_matrix_area_on_block_dur.index)
        for i in list(missing_blocks):
             fix_in_matrix_area_on_block_dur.loc[i] = 0 
        fix_in_matrix_area_on_block_dur.sort_index(inplace=True)
        fix_in_matrix_area_on_block_dur = fix_in_matrix_area_on_block_dur.reset_index()
    
        beh_data['fix_in_matrix_area_dur'] = fix_in_matrix_area_on_block_dur['dur']
    
        gb = beh_data.groupby('answers').sum()
        rtm = (gb['fix_in_matrix_area_dur'] / (gb['rt'] * 1000.0))
        
        part_result["RTM_EASY"] = rtm[LEVEL.EASY]
        part_result["RTM_MEDIUM"] = rtm[LEVEL.MEDIUM]
        part_result["RTM_HARD"] = rtm[LEVEL.HARD]
        
        # # 3. relative first response fixation (RFRF) 
        # timestamp of the first fixation within the response area (time before first response fixation) divided by total response time.
    
    
        fix_in_problem_area = pd.concat([in_roi(fix_data[['axp', 'ayp']], ROIS['P1']),
                                         in_roi(fix_data[['axp', 'ayp']], ROIS['P2']),
                                         in_roi(fix_data[['axp', 'ayp']], ROIS['P3'])], axis=1).any(axis = 1)
        first_fix_in_problem_area = np.where(fix_in_problem_area == True)[0][0]
        fix_data_for_RF = fix_data[first_fix_in_problem_area:]
    
        fix_in_matrix_area = pd.concat([in_roi(fix_data_for_RF[['axp', 'ayp']], ROIS['A']),
                                        in_roi(fix_data_for_RF[['axp', 'ayp']], ROIS['B']),
                                        in_roi(fix_data_for_RF[['axp', 'ayp']], ROIS['C']),
                                        in_roi(fix_data_for_RF[['axp', 'ayp']], ROIS['D']),
                                        in_roi(fix_data_for_RF[['axp', 'ayp']], ROIS['E']),
                                        in_roi(fix_data_for_RF[['axp', 'ayp']], ROIS['F'])], axis = 1).any(axis = 1)
        first_fix_in_matrix_ds = fix_data_for_RF[fix_in_matrix_area].groupby('block').first()['stime']
        blocks_start_ds  = raw_data.groupby('block').first()['time']
        beh_data['time_of_first_fix_on_matrix_area'] = (first_fix_in_matrix_ds - blocks_start_ds).fillna(0).reset_index()[0]
    
    
        gb = beh_data.groupby('answers').sum() 
        rfrf = (gb['time_of_first_fix_on_matrix_area'] / (gb['rt'] * 1000.0))
        part_result["RFRF_EASY"] = rfrf[LEVEL.EASY]
        part_result["RFRF_MEDIUM"] = rfrf[LEVEL.MEDIUM]
        part_result["RFRF_HARD"] = rfrf[LEVEL.HARD]
        # # number of responses visited (RV) 
        # counted all response alternatives that were fixated at least once during the response time (following Bethell-Fox et al., 1984)
        # 
        # Modyfikacja! Teraz: liczba fiksacji + łączny czas trwania fiksacji na każdej z odpowiedzi. [time spent on each response alternative]
    #%%
        problems = yaml_data['list_of_blocks'][1]['experiment_elements'][1:]
        problems += yaml_data['list_of_blocks'][2]['experiment_elements'][1:]
        names = list()
        N = list()
        for problem in problems:
            N.append(problem['rel'])
            tmp = []
            for matrix in problem['matrix_info']:
                tmp.append(matrix['name'])
            names.append(tmp)
        names = [x[3:] for x in names] # remove question area (A B C)
    
        #%% couse we based on yaml wth problems, but we don't use block with training
        beh_data = beh_data[beh_data['answers'] != LEVEL.TRAINING]
        beh_data = beh_data.reset_index()
        #%%
        new_names = []
        res = []
        for x in names:
            new_names.append([int(w[-1]) for w in x])
        assert len(new_names) == len(beh_data['answers']) == len(N), 'Corrupted behavioral data files'
        #%%
        for name, lab, n in zip(new_names, beh_data['answers'].map({
                    '[1, 2, 2, 3, 3, 6]': 'H', 
                    '[1, 3, 3, 4, 4, 6]': 'M',
                    '[1, 4, 4, 5, 5, 6]': 'E'}), N):
            new_name = []
            for item in name:
                new_name.append(str(item) + '_' + str(lab))
            res.append(new_name)
            
        new_names = res
        
        #%% 
        trial_time = raw_data.groupby('block')
        trial_time = pd.DataFrame([trial_time.first()['time'], trial_time.last()['time']])
        
        trial_time = trial_time.quantile(q=[0.25, 0.5, 0.75]).transpose()
        #%%
        quarter = list()
        for fixation in fix_data.iterrows():
            q25, q50, q75 = trial_time.ix[fixation[1]['block']]
            stime = fixation[1]['stime']
            if stime <= q25:
                quarter.append(1)
            elif stime >= q25 and stime < q50:
                quarter.append(2)
            elif stime >= q50 and stime < q75:
                quarter.append(3)
            else: #stime > q75
                quarter.append(4)
                
        assert fix_data.shape[0] == len(quarter), 'Quarters wrongly calculated'
        fix_data['quarter'] = quarter
        #%%
        index = pd.MultiIndex.from_product([range(1, 46), [0]])
        
        fix_in_A = fix_data[in_roi(fix_data[['axp', 'ayp']], ROIS['A'])].groupby(['block', 'quarter']).sum()['dur']    
        fix_in_A_2 = pd.Series(pd.Series(fix_in_A.sum(level=0), index=range(1,46)).values, index=index)
        fix_in_A = pd.concat([fix_in_A, fix_in_A_2], axis=1)
        fix_in_A = fix_in_A['dur'].fillna(fix_in_A[0])    
        
        fix_in_B = fix_data[in_roi(fix_data[['axp', 'ayp']], ROIS['B'])].groupby(['block', 'quarter']).sum()['dur']    
        fix_in_B_2 = pd.Series(pd.Series(fix_in_B.sum(level=0), index=range(1,46)).values, index=index)
        fix_in_B = pd.concat([fix_in_B, fix_in_B_2], axis=1)
        fix_in_B = fix_in_B['dur'].fillna(fix_in_B[0])
        
        fix_in_C = fix_data[in_roi(fix_data[['axp', 'ayp']], ROIS['C'])].groupby(['block', 'quarter']).sum()['dur']
        fix_in_C_2 = pd.Series(pd.Series(fix_in_C.sum(level=0), index=range(1,46)).values, index=index)
        fix_in_C = pd.concat([fix_in_C, fix_in_C_2], axis=1)
        fix_in_C = fix_in_C['dur'].fillna(fix_in_C[0])
        
        fix_in_D = fix_data[in_roi(fix_data[['axp', 'ayp']], ROIS['D'])].groupby(['block', 'quarter']).sum()['dur']
        fix_in_D_2 = pd.Series(pd.Series(fix_in_D.sum(level=0), index=range(1,46)).values, index=index)
        fix_in_D = pd.concat([fix_in_D, fix_in_D_2], axis=1)
        fix_in_D = fix_in_D['dur'].fillna(fix_in_D[0])        
        
        fix_in_E = fix_data[in_roi(fix_data[['axp', 'ayp']], ROIS['E'])].groupby(['block', 'quarter']).sum()['dur']
        fix_in_E_2 = pd.Series(pd.Series(fix_in_E.sum(level=0), index=range(1,46)).values, index=index)
        fix_in_E = pd.concat([fix_in_E, fix_in_E_2], axis=1)
        fix_in_E = fix_in_E['dur'].fillna(fix_in_E[0])         
        
        fix_in_F = fix_data[in_roi(fix_data[['axp', 'ayp']], ROIS['F'])].groupby(['block', 'quarter']).sum()['dur']
        fix_in_F_2 = pd.Series(pd.Series(fix_in_F.sum(level=0), index=range(1,46)).values, index=index)
        fix_in_F = pd.concat([fix_in_F, fix_in_F_2], axis=1)
        fix_in_F = fix_in_F['dur'].fillna(fix_in_F[0])   
    
        #%% creating mock in order to fill all absent values in df 
        high_index = sorted(list(range(1,46)) + list(range(1,46))+list(range(1,46))+list(range(1,46)))
        low_index = list(range(1,5)) * 45
        assert len(high_index) == len(low_index), "Error in mock index geration"
        tuples = list(zip(high_index, low_index))
        index = pd.MultiIndex.from_tuples(tuples, names=['block', 'quarter'])
        mock = pd.Series(np.NaN, index=index)
        fix_dur = pd.concat([fix_in_A, fix_in_B, fix_in_C, fix_in_D,fix_in_E,fix_in_F, mock], axis = 1).drop('blk', 1)
        fix_dur.columns = ['dur_in_A', 'dur_in_B','dur_in_C', 'dur_in_D', 'dur_in_E', 'dur_in_F', 'mock']
        fix_dur = fix_dur.fillna(0).drop('mock', 1)
        
    #%%      
        d = defaultdict(list)
        for l, row in zip(new_names, fix_dur.xs(0, level=1)[3:].iterrows()):
            if (row[0]) in [1, 2, 3]: #trening
                continue
            for k, v in  zip(l, row[1][2:]):
                d[k].append(v)
        #%%
        for k in trial_dict.keys():
            part_result['RV_AVG_DUR_' + trial_dict[k]] = np.nanmean(d[k]) 
            
        #%%
        fix_in_A = fix_data[in_roi(fix_data[['axp', 'ayp']], ROIS['A'])].groupby('block').count()['dur']
        fix_in_B = fix_data[in_roi(fix_data[['axp', 'ayp']], ROIS['B'])].groupby('block').count()['dur']
        fix_in_C = fix_data[in_roi(fix_data[['axp', 'ayp']], ROIS['C'])].groupby('block').count()['dur']
        fix_in_D = fix_data[in_roi(fix_data[['axp', 'ayp']], ROIS['D'])].groupby('block').count()['dur']
        fix_in_E = fix_data[in_roi(fix_data[['axp', 'ayp']], ROIS['E'])].groupby('block').count()['dur']
        fix_in_F = fix_data[in_roi(fix_data[['axp', 'ayp']], ROIS['F'])].groupby('block').count()['dur']
    
    
    
        missing_blocks = set(range(1, 46)) - set(fix_in_A.index)
        for i in list(missing_blocks):
             fix_in_A.loc[i] = 0 
        fix_in_A.sort_index(inplace=True)
        fix_in_A = fix_in_A.reset_index()
    
        missing_blocks = set(range(1, 46)) - set(fix_in_B.index)
        for i in list(missing_blocks):
             fix_in_B.loc[i] = 0 
        fix_in_B.sort_index(inplace=True)
        fix_in_B = fix_in_B.reset_index()
    
        missing_blocks = set(range(1, 46)) - set(fix_in_C.index)
        for i in list(missing_blocks):
             fix_in_C.loc[i] = 0 
        fix_in_C.sort_index(inplace=True)
        fix_in_C = fix_in_C.reset_index()
    
        missing_blocks = set(range(1, 46)) - set(fix_in_D.index)
        for i in list(missing_blocks):
             fix_in_D.loc[i] = 0 
        fix_in_D.sort_index(inplace=True)
        fix_in_D = fix_in_D.reset_index()
    
        missing_blocks = set(range(1, 46)) - set(fix_in_E.index)
        for i in list(missing_blocks):
             fix_in_E.loc[i] = 0 
        fix_in_E.sort_index(inplace=True)
        fix_in_E = fix_in_E.reset_index()
    
        missing_blocks = set(range(1, 46)) - set(fix_in_F.index)
        for i in list(missing_blocks):
             fix_in_F.loc[i] = 0 
        fix_in_F.sort_index(inplace=True)
        fix_in_F = fix_in_F.reset_index()
    
        # print fix_in_A.columns
        fix_in_A.columns = ['block', 'no_in_A']
        fix_in_B.columns = ['blk', 'no_in_B']
        fix_in_C.columns = ['blk', 'no_in_C']
        fix_in_D.columns = ['blk', 'no_in_D']
        fix_in_E.columns = ['blk', 'no_in_E']
        fix_in_F.columns = ['blk', 'no_in_F']
    
    
    
        fix_dur = pd.concat([fix_in_A, fix_in_B, fix_in_C, fix_in_D,fix_in_E,fix_in_F], axis = 1).drop('blk', 1)
        fix_dur = fix_dur[fix_dur.block != 1]
        fix_dur = fix_dur[fix_dur.block != 2]
        fix_dur = fix_dur[fix_dur.block != 3]
        fix_dur.reset_index(inplace=True)
    
    
        d = defaultdict(list)
        for l, row in zip(new_names, fix_dur.iterrows()):
            for k, v in  zip(l, row[1][2:]):
                d[k].append(v)
        # BIG_ERROR i SMALL_ERROR must be divised by two
        for k in trial_dict.keys():
            if 'ERROR' in trial_dict[k]:
                part_result['RV_SUM_FIX_' + trial_dict[k]] = np.nansum(d[k]) / 2.0
            else:
                part_result['RV_SUM_FIX_' + trial_dict[k]] = np.nansum(d[k]) 
    
        # # Pupil size
  #%%  
        avg_pupil_size = raw_data.groupby('block').mean()['ps']
    
        missing_blocks = set(range(1, 46)) - set(avg_pupil_size.index)
        for i in list(missing_blocks):
             avg_pupil_size.loc[i] = 0 
        avg_pupil_size.sort_index(inplace=True)
        avg_pupil_size = avg_pupil_size.reset_index()
    
        beh_data['avg_pupil_size'] = avg_pupil_size['ps']
        w = beh_data.groupby('answers').mean()['avg_pupil_size']
        part_result["AVG_PUP_SIZE_EASY"] = w[LEVEL.EASY]
        part_result["AVG_PUP_SIZE_MEDIUM"] = w[LEVEL.MEDIUM]
        part_result["AVG_PUP_SIZE_HARD"] = w[LEVEL.HARD]
        
        # mean correctness
        beh_data['corr'] = beh_data['corr'].astype(int)
        w = beh_data.groupby( 'answers').mean()
        part_result["MEAN_CORR_EASY"] = w['corr'][LEVEL.EASY]
        part_result["MEAN_CORR_MEDIUM"] = w['corr'][LEVEL.MEDIUM]
        part_result["MEAN_CORR_HARD"] = w['corr'][LEVEL.HARD]
        
        part_result["MEAN_RT_EASY"] = w['rt'][LEVEL.EASY]
        part_result["MEAN_RT_MEDIUM"] = w['rt'][LEVEL.MEDIUM]
        part_result["MEAN_RT_HARD"] = w['rt'][LEVEL.HARD]
        
        #%% percentage of individual options in given answers
        
        beh_data = beh_data[beh_data['choosed_option'] != '-1'] # removed unchoosed trials
        easy = beh_data[beh_data['answers'] == LEVEL.EASY]
        medium = beh_data[beh_data['answers'] == LEVEL.MEDIUM]
        hard = beh_data[beh_data['answers'] == LEVEL.HARD]
        
        easy = easy.groupby('choosed_option').count()['ans_accept'] / easy.shape[0]
        medium = medium.groupby('choosed_option').count()['ans_accept'] / medium.shape[0]
        hard = hard.groupby('choosed_option').count()['ans_accept'] / hard.shape[0]
        
        part_result['PERC_CORR_EASY'] = easy.get('D1', 0.0)
        part_result['PERC_CORR_MEDIUM'] = medium.get('D1', 0.0)
        part_result['PERC_CORR_HARD'] = hard.get('D1', 0.0)
        
        part_result['PERC_SMALL_ERROR_EASY'] = easy.get('D4', 0.0)
        part_result['PERC_SMALL_ERROR_MEDIUM'] = medium.get('D3', 0.0)   
        part_result['PERC_SMALL_ERROR_HARD'] = hard.get('D2', 0.0)
        
        part_result['PERC_BIG_ERROR_EASY'] = easy.get('D5', 0.0)
        part_result['PERC_BIG_ERROR_MEDIUM'] = medium.get('D4', 0.0)
        part_result['PERC_BIG_ERROR_HARD'] = hard.get('D3', 0.0) 
        
        part_result['PERC_CONTROL_EASY'] = easy.get('D6', 0.0)
        part_result['PERC_CONTROL_MEDIUM'] = medium.get('D6', 0.0)
        part_result['PERC_CONTROL_HARD'] = hard.get('D6', 0.0)
        #%% Relatinal metric
        
        rm = list()
        for idx in beh_data.index: # iterate over index, couse some items are mised, due to choosed_option == -1
            choosed_option = beh_data['choosed_option'][idx]
            problem = problems[idx]['matrix_info']
            
            denom = np.sum([len(x['elements_changed']) for x in problem[1]['parameters']])
            counter = [x for x in problem if x['name'] == choosed_option][0]['parameters']
            counter = np.sum([len(x['elements_changed']) for x in counter])
            if choosed_option == 'D2':# some magic
                rm.append(((counter - 1)/denom) + 0.02)
            else:
                rm.append(counter/denom)
        part_result['RM'] = np.mean(rm)
        #%%
        RESULTS.append(part_result)


 # %%Save results
res = pd.DataFrame(RESULTS)
pd.DataFrame(RESULTS).to_csv('new_result.csv')
pd.DataFrame(RESULTS).to_excel('new_result.xlsx')
