# -*- coding:utf-8 -*-

import sys
import numpy as np
import h5py
import os
import re

periodictable_symbols=['H','He','Li','Be','B','C','N','O','F','Ne','Na','Mg','Al','Si','P','S','Cl','Ar','K','Ca','Sc','Ti','V','Cr','Mn','Fe','Co',
                       'Ni','Cu','Zn','Ga','Ge','As','Se','Br','Kr','Rb','Sr','Y','Zr','Nb','Mo','Tc','Ru','Rh','Pd','Ag','Cd','In','Sn','Sb','Te',
                       'I','Xe','Cs','Ba','La','Ce','Pr','Nd','Pm','Sm','Eu','Gd','Tb','Dy','Ho','Er','Tm','Yb','Lu','Hf','Ta','W','Re','Os','Ir',
                       'Pt','Au','Hg','Tl','Pb','Bi','Po','At','Rn','Fr','Ra','Ac','Th','Pa','U','Np','Pu','Am','Cm','Bk','Cf','Es','Fm','Md','No',
                       'Lr','Rf','Db','Sg','Bh','Hs','Mt','Ds','Rg','Cn','Uut','Uuq','Uup','Uuh','Uus','Uuo']


periodictable_dict=dict(zip(periodictable_symbols, range(1,1+len(periodictable_symbols))))



def make_strline(line):
    tele=line.strip().split(' ')
    tmptele=np.array(tele[:-1])
    ele=[periodictable_dict[ttele] for ttele in tmptele[[0,1,2,4]]]
    eng=float(tele[-1])

    tstr=map(str, ele)        
    tstr=zip(tstr,['1','1','2','6'])
    tstr=' '.join([' '.join(tstr[i]) for i in range(len(tstr))])
    tstr=' '.join([tstr, '>'+ str(eng)])    
    return tstr

def make_sublist(keepn):
    np.random.seed(10)

    with open('triangt.dat') as fid:
        lines=fid.readlines()

    indexpos=range(len(lines))

    trpath='./'+keepn+'/abc2d6_tr.txt'
    tspath='./'+keepn+'/abc2d6_ts.txt'
    clspath='./'+keepn+'/cls_4c_W_low'    
    if not os.path.exists(clspath):
        os.makedirs(clspath)
    clspath='./'+keepn+'/cls_4c_W_high'
    if not os.path.exists(clspath):
        os.makedirs(clspath)
    logpath='./'+keepn+'/log'
    if not os.path.exists(logpath):
        os.makedirs(logpath)        
    np.random.shuffle(indexpos)
    keepm=int(keepn)    

    with open(trpath,'w') as trfid:
        trnum=0
        cnt=0
        for index in indexpos[:keepm]:
            trnum+=1
            line=lines[index]
            tstr=make_strline(line)
            trfid.write(tstr+'\n')
            cnt+=1
            if cnt % 100 == 0:
                print('processed {} images!'.format(cnt+1))    
    
    with open(tspath,'w') as tsfid:
        cnt=0    
        tsnum=0
        for index in indexpos[keepm:]:
            tsnum+=1
            line=lines[index]
            tstr=make_strline(line)
            tsfid.write(tstr+'\n')
            cnt+=1
            if cnt % 100 == 0:
                print('processed {} images!'.format(cnt+1))    
    
    print('the train number is {}, the test number is {}'.format(trnum,tsnum))


def make_allsub():
    dirlist=['01000','02000','03000','04000','05000','06000','07000','08000','09000','10000']
    for subdir in dirlist:
        subdirpath=os.path.join('.', subdir)
        if not os.path.exists(subdirpath):
            os.mkdir(subdirpath)
    for keepn in dirlist:
        fldpath=os.path.join('.', keepn)
        if keepn[0].isdigit() and os.path.isdir(fldpath):
            make_sublist(keepn)

if __name__ == '__main__':
    make_allsub()

