#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 20 15:50:41 2017

@author: bkroczek
"""
import pandas as pd 
from tqdm import tqdm
def RepresentsInt(s):
    try: 
        int(s)
        return True
    except ValueError:
        return False
    
def RepresentsFloat(s):
    try: 
        float(s)
        return True
    except ValueError:
        return False

files = ['12MALE21.asc', '14FEMALE19.asc', '62FEMALE39.asc', '83MALE27.asc', '130MALE18.asc','142FEMALE29.asc', '165FEMALE20.asc']
# /home/bkroczek/Dropbox/Data/FAN_ET/Badanie P/2017-05-06_Badanie_P/BadanieP_FAN_ET/Dane trackingowe/asc_to_fix
#%%


with tqdm(total = len(files)) as pbar:
    for f_name in files:  
        blk_no = 0
        pbar.set_postfix(file = f_name)
        pbar.update(1)
        f = open(f_name, 'r').readlines()
        res = [['time', "ps", "block"]]
        for line in f:
           if line.startswith('START'):
               blk_no += 1
           line = line.split()
           if line:
               if RepresentsInt(line[0]) and blk_no != 0 and RepresentsFloat(line[-2]):
                   res.append([int(line[0]), float(line[-2]), blk_no])
        pd.DataFrame(res).to_csv(f_name.split('.')[0] + '_raw.csv', index=False)
        