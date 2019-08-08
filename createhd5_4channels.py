# -*- coding: utf-8 -*-

import sys
import numpy as np
import h5py
import os
from scipy.stats import norm


IMAGE_SIZE = (6, 18)

postable={1:[0, 0], 2:[0, 17],
                3: [1, 0], 4: [1, 1], 5: [1, 12], 6: [1, 13], 7: [1, 14],8:[1,15], 9:[1,16],10:[1,17],
                11: [2,0], 12: [2,1], 13: [2,12], 14: [2,13], 15: [2,14],16:[2,15],17: [2,16],18:[2,17],
                19: [3, 0], 20: [3, 1], 21: [3, 2], 22: [3, 3], 23: [3, 4], 24: [3, 5], 25: [3, 6], 26: [3, 7], 27: [3, 8], 28: [3, 9], 29: [3, 10], 30: [3, 11],31: [3,12], 32: [3,13], 33: [3,14],34:[3,15],35: [3,16],36:[3,17],
                37: [4, 0], 38: [4, 1], 39: [4, 2], 40: [4, 3], 41: [4, 4], 42: [4, 5], 43: [4, 6], 44: [4, 7], 45: [4, 8], 46: [4, 9], 47: [4, 10], 48: [4, 11], 49: [4,12], 50: [4,13], 51: [4,14],52:[4,15],53: [4,16],54:[4,17],
                55: [5, 0], 56: [5, 1], 72: [5, 3], 73: [5, 4], 74: [5, 5], 75: [5, 6], 76: [5, 7], 77: [5, 8], 78: [5, 9], 79: [5, 10], 80: [5, 11], 81: [5,12], 82: [5,13], 83: [5,14],84:[5,15]}

tconst=np.zeros(IMAGE_SIZE, dtype=np.float32)
for key in postable.iterkeys():
    posy=postable[key][0]
    posx=postable[key][1]
    tconst[posy,posx]=-1

def gettrsta_low(filename):
    if '_ts' in filename:
        filename=filename.replace('_ts','_tr')
    with open(filename, 'r') as f:
        lines = f.readlines()

    tvec1=[]    
    for i, line in enumerate(lines):
        trdata = line.strip().split('>')
        tvec1.append(float(trdata[-1]))        
    
    tvec1=np.array(tvec1)
    return tvec1.mean(),tvec1.std(), tvec1

def get1channel(val):
    tablenum=len(postable)
    pos, num = val
    tmpdata21 = tconst.copy()
    posy=postable[pos][0]
    posx=postable[pos][1]    
    tmpdata21 *= num / (tablenum - 1.0)
    tmpdata21[posy,posx]= num
    tmpdata3 = tmpdata21*40
    return tmpdata3

def makefea_low(line):
    trdata = line.strip().split('>')
    gt_val = float(trdata[-1])

    tmpdata2 = np.reshape([int(x) for x in trdata[0].split()], [-1,2])
    tmp_rep=[]
    for val in tmpdata2:
        tmps=get1channel(val)
        tmp_rep.append(tmps)
    gt_rep=np.stack(tmp_rep, axis=2)    
    gt_rep=np.transpose(gt_rep, [2, 0, 1])
    return gt_rep, gt_val

def gettrdat_low(filename):
    if not os.path.exists(filename):
        return False

    setname=filename.split('.txt')[0]
    with open(filename, 'r') as f:
        lines = f.readlines()

    np.random.shuffle(lines)

    sample_size = len(lines)
    imgs = np.zeros((sample_size, 4,) + IMAGE_SIZE, dtype=np.float32)
    scores = np.zeros((sample_size,1), dtype=np.float32)
    h5_filename = '{}_4c_W_low.h5'.format(setname)
    index = 0

    tmean1,tstd1,_=gettrsta_low(filename)
    with h5py.File(h5_filename, 'w') as h:
        for i, line in enumerate(lines):
            # print line
            gt_rep, gt_val=makefea_low(line)
            imgs[index] = gt_rep
            scores[index] = (gt_val - tmean1)/tstd1
            index += 1
            if (i+1) % 1000 == 0:
                print('processed {} data!'.format(i+1))    
        h.create_dataset('data', data=imgs[:index])
        h.create_dataset('score', data=scores[:index])
    
    h5_filename = '{}_4c_W_low.h5'.format(setname.split('/')[-1])
    with open('{}_4c_W_low_h5.txt'.format(setname), 'w') as f:
        f.write(h5_filename)
    print('the total number is {}, the real number is {}'.format(sample_size, index))
    return True


def gettrsta_high(filename):
    if '_ts' in filename:
        filename=filename.replace('_ts','_tr')
    with open(filename, 'r') as f:
        lines = f.readlines()

    tvec1=[]    
    for i, line in enumerate(lines):
        trdata = line.strip().split('>')
        gt_val1, pred1 = map(float,trdata[-1].split(' '))
        tvec1.append(gt_val1-pred1)
    
    tvec1=np.array(tvec1)
    return tvec1.mean(),tvec1.std(), tvec1



def makefea_high(line):
    trdata = line.strip().split('>')
    gt_val1, pred1 = map(float,trdata[-1].split(' '))

    tmpdata2 = np.reshape([int(x) for x in trdata[0].split()], [-1,2])
    tmp_rep=[]
    for val in tmpdata2:
        tmps=get1channel(val)
        tmp_rep.append(tmps)
    gt_rep=np.stack(tmp_rep, axis=2)    
    gt_rep=np.transpose(gt_rep, [2, 0, 1])

    return gt_rep, gt_val1, pred1

def gettrdat_high(filename):
    if not os.path.exists(filename):
        return False

    setname=filename.split('.txt')[0]
    with open(filename, 'r') as f:
        lines = f.readlines()

    np.random.shuffle(lines)

    sample_size = len(lines)
    imgs = np.zeros((sample_size, 4,) + IMAGE_SIZE, dtype=np.float32)
    scores = np.zeros((sample_size,1), dtype=np.float32)
    h5_filename = '{}_4c_W_high.h5'.format(setname)
    index = 0
    tmean1,tstd1,_=gettrsta_high(filename)
    with h5py.File(h5_filename, 'w') as h:
        for i, line in enumerate(lines):
            gt_rep, gt_val, pred_val=makefea_high(line)
            imgs[index] = gt_rep      
            gt_val -= pred_val
            scores[index] =(gt_val-tmean1)/tstd1
            index += 1
            if (i+1) % 1000 == 0:
                print('processed {} data!'.format(i+1))    
        h.create_dataset('data', data=imgs[:index])
        h.create_dataset('score', data=scores[:index])
    
    partname='/'.join(setname.split('/')[2:])
    h5_filename = './{}_4c_W_high.h5'.format(partname)
    with open('{}_4c_W_high_h5.txt'.format(setname), 'w') as f:
        f.write(h5_filename)
    print('the total number is {}, the real number is {}'.format(sample_size, index))
    return True

def getalldat_low():
    fldlist=os.listdir('./')
    for fld in fldlist:
        fldpath='./{}'.format(fld)
        if os.path.isdir(fldpath) and fld.isdigit():        
            fldnum=int(fld) 
            if fldnum < 1000 or fldnum > 10000:
                continue       
            clspath=os.path.join(fldpath, 'cls_4c_W_low')      
            if not os.path.exists(clspath):
                os.mkdir(clspath)     
            bsav = gettrdat_low(os.path.join(fldpath,'abc2d6_tr.txt'))
            bsav = gettrdat_low(os.path.join(fldpath,'abc2d6_ts.txt'))
            if bsav:
                print('save {}'.format(fld))    
            else:
                print('{} not save'.format(fld))

def getalldat_high():
    fldlist=os.listdir('./')
    # fldlist=['10000']
    for fld in fldlist:
        fldpath='./{}'.format(fld)
        if os.path.isdir(fldpath) and fld.isdigit(): 
            fldnum=int(fld) 
            if fldnum < 1000 or fldnum > 10000:
                continue
            clspath=os.path.join(fldpath, 'cls_4c_W_high')      
            if not os.path.exists(clspath):
                os.mkdir(clspath)
            bsav = gettrdat_high(os.path.join(fldpath,'res/abc2d6_tr_4c_W_low_res.txt'))
            bsav = gettrdat_high(os.path.join(fldpath,'res/abc2d6_ts_4c_W_low_res.txt'))
            if bsav:
                print('save {}'.format(fld))    
            else:
                print('{} not save'.format(fld))

if __name__=='__main__':
    if sys.argv[-1] == 'low':
        getalldat_low()
    else:
        getalldat_high()
