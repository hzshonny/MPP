# -*- coding:utf-8 -*-
import sys
import numpy
import os
import shutil
import glob

def cpfile2():
    cpsrcf1=glob.glob('./*.prototxt')
    cpsrcf2=glob.glob('./*.sh')
    cpsrcf=cpsrcf1+cpsrcf2
    fldlist=os.listdir('./')
    for fld in fldlist:
        fldpath=os.path.join('./', fld)
        if fld[0].isdigit() and os.path.isdir(fldpath):
            for cpf in cpsrcf:
                cpname=os.path.basename(cpf)
                if 'cmd' in cpname:
                    continue                
                desfile=os.path.join(fldpath, cpname)
                shutil.copy(cpf, desfile)
                print('copy {}'.format(desfile))



if __name__ == '__main__':    
    cpfile2()
