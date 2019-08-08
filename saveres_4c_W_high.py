import numpy as np
import caffe
import os
import time
import createhd5_4channels



def testsubdat_high(filename):

    pos=filename.rfind('/')
    fldname=filename[:pos]
    WEIGHTS_FILE = os.path.join(fldname, 'cls_4c_W_high/reg_iter_100000.caffemodel')
    WEIGHTS_FILE=WEIGHTS_FILE.replace('res/','')
    if not os.path.exists(filename):
        return False
    if not os.path.exists(WEIGHTS_FILE):
        return False
    DEPLOY_FILE = os.path.join(fldname,'deploy_4c_W_high.prototxt')
    DEPLOY_FILE=DEPLOY_FILE.replace('res/','')
    IMAGE_SIZE = createhd5_4channels.IMAGE_SIZE


    caffe.set_device(0)
    caffe.set_mode_gpu()

    net1 = caffe.Net(DEPLOY_FILE, WEIGHTS_FILE, caffe.TEST)
    net1.blobs['data'].reshape(1, 4, *IMAGE_SIZE)

    with open(filename, 'r') as f:
        lines = f.readlines()

    sample_size = len(lines)
    tscores = np.zeros((sample_size), dtype=np.float32)
    index=0
    step1s=0
    
    tmean1,tstd1,_=createhd5_4channels.gettrsta_high(filename)

    setname=os.path.basename(filename).split('.')[0]
    partname=filename.split('/')[1]
    if not os.path.exists('./res_4channels'):
        os.makedirs('./res_4channels')
    resfile='./res_4channels/{}_{}_4c_W_high_res.txt'.format(partname, setname)
    alltimeused=0
    with open(resfile,'w') as fid:
        for i, line in enumerate(lines):
            gt_rep, gt_val, pred_val1=createhd5_4channels.makefea_high(line)
            net1.blobs['data'].data[...] = gt_rep
            t1=time.time()
            output1 = net1.forward()   
            alltimeused += (time.time()-t1) 
            x1=output1['pred'][0][0]            
            y1=x1*tstd1+tmean1
            y2=pred_val1+y1
            step1s+=abs(gt_val-y2)/10
            index+=1       
            fid.write('{} {} {}\n'.format(line.strip(),y1, y2)) 
            if (i+1) % 1000 == 0:
                print('processed {} data!, the engery error is {}'.format(i+1, step1s/index))
    print('the final engery error is {}'.format(step1s/index))
    if '_ts' in filename:
        with open('./res_4channels/allres_4c_W_high_ts.txt','a') as fid:
            strline='train:{},test:{}\nengery:{}, time:{}\n'.format(10590-index,index,step1s/index, alltimeused/index)
            fid.write(strline)
    else:
        with open('./res_4channels/allres_4c_W_high_tr.txt','a') as fid:
            strline='train:{},test:{}\n engery:{}, time:{}\n'.format(index,10590-index,step1s/index, alltimeused/index)
            fid.write(strline)  
    return True  

def testalldat():

    fldlist=os.listdir('./')    
    for fld in fldlist:
        fldpath='./{}'.format(fld)
        if fld.isdigit() and os.path.isdir(fldpath):
            fldnum=int(fld) 
            if fldnum < 1000  or fldnum > 10000:
                continue
            print('train {}'.format(fld))    
            bsav = testsubdat_high(os.path.join(fldpath,'res/abc2d6_tr_4c_W_low_res.txt'))
            if not bsav:
                print('{} non train'.format(fld))
            print('test {}'.format(fld)) 
            bsav = testsubdat_high(os.path.join(fldpath,'res/abc2d6_ts_4c_W_low_res.txt'))
            if not bsav:
                print('{} non test'.format(fld))


if __name__=='__main__':
    testalldat()