# -*- coding: utf-8 -*-
# Algorithm 1: CNN-based activity segmentation algorithm
# Input: Trained state inference model, CSI data, window size w, length for calculating the mode m
# Output: Start point t_{start} and end point t_{end} of the activity

import os
import time

import numpy as np
import sys
import math
import h5py
import queue
#import simple_usual1

import scipy.io as scio
import hdf5storage

import argparse

#parser is used to accept parameters from commandlines,such as seting epoch=10:python train_CSI.py --epoch 10 
parser = argparse.ArgumentParser(description='')
#parser.add_argument('--epoch', dest='epoch', type=int, default=200, help='# of epoch')
parser.add_argument('--dataDir',   dest='dataDir', default='data', help='directory of data')
parser.add_argument('--StateFile', dest='StateFile', default='StateLabel_DiscretizeCsi_only6/', help='directory of data')

args = parser.parse_args()

lenAction = round(4000/20) #动作大小
whichCol = 0 #whichCol = args.whichCol

def longQueueModeRatePrevious(currentQ,predictResult,i):
    selectLong1 = 24
    selectLong2 = 24
    qBig   = currentQ.copy()
    qBig.clear()
    for kk in range(i-7- selectLong1 - 1, i - 7):
        qBig.append(predictResult[kk,whichCol])
    countsBig1 = np.bincount(qBig)
    modePrev1 = np.argmax(countsBig1)
    
    qBig.clear()
    for kk in range(i-7- selectLong2 - 1 -selectLong1, i - 7 - selectLong1):
        qBig.append(predictResult[kk,whichCol])
    countsBig2 = np.bincount(qBig)
    modePrev2 = np.argmax(countsBig2)    
    '''
    print('countsBig            :', countsBig)
    print('qBig                :', qBig)      
    print('i+120  ---Start--- :', i+120)  
    print('countsBig[modeBig]/float(20)                :', countsBig[modeBig]/float(selectLong )) 
    '''
    return (modePrev1, modePrev2, countsBig1[modePrev1]/float(selectLong1))

def longQueueModeRateNext(currentQ,predictResult,i):
    selectLong1 = 24
    selectLong2 = 24
    qBig   = currentQ.copy()
    qBig.clear()
    for kk in range(i -3,i -3 + selectLong1  + 1):
        qBig.append(predictResult[kk,whichCol])
    countsBig1 = np.bincount(qBig)
    modeNext1 = np.argmax(countsBig1)
        
    qBig.clear()
    for kk in range(i -3 + selectLong1 + 1,i -3 + selectLong1  + 1 + selectLong2):
        qBig.append(predictResult[kk,whichCol])
    countsBig2 = np.bincount(qBig)
    modeNext2 = np.argmax(countsBig2)    
    
    '''
    print('countsBig            :', countsBig)
    print('qBig                :', qBig)      
    print('i+120  ---End----- :', i+120)  
    print('countsBig[modeBig]/float(20)                :', countsBig[modeBig]/float(selectLong + 10)) 
    '''
    return (modeNext1, modeNext2, countsBig1[modeNext1]/float(selectLong1))
def actionStartEndPoints(predictDataDir):

    predictResult = np.loadtxt(predictDataDir) 
    print('predictResult.shape    ::', predictResult.shape) 
    #print(predictResult[4,1])
    #print('len(predictResult)-10    ::', len(predictResult)-10) 
    
    
    qDetectChange = queue.Queue(maxsize=10)
    for i in range(10):
        qDetectChange.put(predictResult[i,whichCol],block=False)
    #print('qSmall.qsize()   :', qSmall.qsize())
    #print('qSmall.queue     :', qSmall.queue)
    qModeRecord = queue.Queue(maxsize=12)
    for i in range(12):
        qModeRecord.put(0,block=False)
    bigModeRateThrehold = 0.49
    startEndMark = np.zeros([10,2])
    lastMode = 0
    currentMode = 0
    whichSample = 0
    oneSampleStartMark = False
    for i in range(10,len(predictResult)):
        
        currentQ =qDetectChange.queue
        #print(currentQ[1])
        
        counts = np.bincount(currentQ)
        #print('counts            :', counts)
        #print('np.argmax(counts) :', np.argmax(counts))
        currentMode = np.argmax(counts)
        qModeRecord.get(block=False)
        qModeRecord.put(currentMode,block=False)
        
        if(lastMode != currentMode and lastMode == 1): #----detect start points
            if(not oneSampleStartMark):
                (modePrev1, modePrev2, bigRatePrev) = longQueueModeRatePrevious(currentQ,predictResult,i)
                (modeNext1, modeNext2, bigRateNext) = longQueueModeRateNext(currentQ,predictResult,i)
                if((modePrev1 == 1 or modePrev2 == 1) and (modeNext1 == 2 or modeNext1 == 3) and \
                   (modeNext2 == 2 or modeNext2 == 3) and bigRatePrev > bigModeRateThrehold):
                    startEndMark[whichSample,0] = i-6+120 # for each file, it cuts first 120.
                    oneSampleStartMark = True
        if(lastMode != currentMode and currentMode == 1): #----detect end points
            if(oneSampleStartMark):
                (modePrev1, modePrev2, bigRatePrev) = longQueueModeRatePrevious(currentQ,predictResult,i)
                (modeNext1, modeNext2, bigRateNext) = longQueueModeRateNext(currentQ,predictResult,i)
                if((modePrev1 == 4 or modePrev2 == 4) and (modeNext1 == 1 and modeNext2==1)  and bigRatePrev > bigModeRateThrehold):
                    #if(i-7+120 - 120 - startEndMark[whichSample,0] > 30):
                        startEndMark[whichSample,1] = i-7+120 - 120
                        oneSampleStartMark = False
                        whichSample += 1
                        if(startEndMark[whichSample-1,1] - startEndMark[whichSample-1,0] < 20): #-----for--user2_iw_5_predict_ema
                            startEndMark[whichSample,0] = 0
                            startEndMark[whichSample,1] = 0
                            whichSample -= 1
                            
                        if(whichSample >= 13):               #-----for--------user5_ph_1
                            if((startEndMark[whichSample-1,1] - startEndMark[whichSample-1,0] \
                               + startEndMark[whichSample-2,1] - startEndMark[whichSample-2,0]) < 120):
                                startEndMark[whichSample-2,1] = startEndMark[whichSample-1,1]
                                startEndMark[whichSample,1] = 0
                                whichSample -= 2
                        
        if(oneSampleStartMark):      #----add only for user4_rp_1_predict_ema----------         
            if((i - startEndMark[whichSample,0]) > 400 and currentMode == 1):    
                startEndMark[whichSample,1] = i-7+120 - 120 -120
                oneSampleStartMark = False
                whichSample += 1
        #if(lastMode != currentMode and currentMode == 1 and lastMode == 4 and oneSampleStartMark):
        #    startEndMark[whichSample,1] = i-7+120 -60
        #    oneSampleStartMark = False
        #    whichSample += 1            
            
        qDetectChange.get(block=False)
        qDetectChange.put(predictResult[i,whichCol],block=False)
        
        lastMode = currentMode
        if(whichSample >= 10):
            break
    #if(predictDataDir.find('user2_iw_5_predict_ema') >= 1): #-----there are some problems for--user2_iw_5_predict_ema
    #    startEndMark = np.array([[390,585],[895,1086],[1316,1524],[1858,2060],[2381,2597],\
    #                    [2930,3185],[2593,3866],[4306,4591],[4975,5265],[5675,5595]] )  #--right one----
        
        #startEndMark = np.array([[390,785],[895,1286],[1316,1724],[1858,2060],[2381,2797],\
        #                [2930,3485],[2593,3866],[4306,4791],[4975,5565],[5675,5695]] )   #--wrong one----
         
    print('startEndMark:\n', str(startEndMark))
    return startEndMark

def OneActionSampleFile(originalCsi,startEndPoints,actionMark):
    """
    需要传入的参数有lowpass A fn(原始数据ln*3*30*样本数   文件分段   文件名iw,ph,rp,sd,wd
    """
    #lenAction = round(4000/20) #动作大小
    pick = np.zeros([3,30,lenAction,5])
    pick_label = np.zeros([5,1])
    handup = np.zeros([3,30,lenAction,5])
    handup_label = np.zeros([5,1])
    
    kk_1 = 0
    kk_2 = 0
    len_A=len(startEndPoints) #10个动作
    F =(np.zeros([10,2])).astype(np.int16)
    #下面是对分段规整成cnn的要求
    for i in range(0,len_A):
        if((startEndPoints[i,1] - startEndPoints[i,0]) <= lenAction):
            le = lenAction - (startEndPoints[i,1]-startEndPoints[i,0])
            le_r = round(le/2)
            le_l = le - le_r
            F[i,0] = int(startEndPoints[i,0] - le_l)
            F[i,1] = int(startEndPoints[i,1] + le_r)
        elif((startEndPoints[i,1] - startEndPoints[i,0]) > lenAction):
            le = (startEndPoints[i,1] - startEndPoints[i,0]) - lenAction
            le_r = round(le/2)
            le_l = le - le_r
            F[i,0] = int(startEndPoints[i,0] + le_l)
            F[i,1] = int(startEndPoints[i,1] - le_r)
    #下面组建data label
    for i in range(0,len_A):
        if(i < 5): #大动作的地方
            pick[:,:,:,kk_1] = originalCsi[:,:,F[i,0]:F[i,1]:1]
            if(actionMark == 'iw'):
                pick_label[kk_1,0] = 1;
            elif(actionMark == 'ph'):
                pick_label[kk_1,0] = 3;
            elif(actionMark == 'rp'):
                pick_label[kk_1,0] = 5;
            elif(actionMark == 'sd'):
                pick_label[kk_1,0] = 7;
            elif(actionMark == 'wd'):
                pick_label[kk_1,0] = 9;
            kk_1=kk_1+1;
        else:  #小动作的地方
            handup[:,:,:,kk_2]=originalCsi[:,:,F[i,0]:F[i,1]:1];
            if(actionMark == 'iw'):
                handup_label[kk_2,0] = 2;
            elif(actionMark == 'ph'):
                handup_label[kk_2,0] = 4;
            elif(actionMark == 'rp'):
                handup_label[kk_2,0] = 6;
            elif(actionMark == 'sd'):
                handup_label[kk_2,0] = 8;
            elif(actionMark == 'wd'):
                handup_label[kk_2,0] = 10;
            kk_2=kk_2+1;
    data = np.concatenate((pick,handup),axis = 3)                # all the action Csi
    label = np.concatenate((pick_label,handup_label),axis = 0)   # all the action Label
    #data = pick                      # only the big action Csi
    #label = np.ceil(pick_label/2)    # only the big action Label 
    #data = handup                    # only the small action Csi
    #label = np.ceil(handup_label/2)  # only the small action Label     
    return data,label

def actionSampleExtract(startEndPoints,originalFile,outFileName,segmentResultDir):
    # startEndPoints are 10 pairs of startPoint and endPoints of action samples. 
    mat=h5py.File(originalFile,'r') #   print(mat.keys())   
    fileName = list(mat.keys())[0] #print(list(mat.keys())[0])
    originalCsi=mat[fileName]
    print('originalCsi.shape     ::', originalCsi.shape)
    originalCsi = np.asarray(originalCsi)# transfer to numpy array
    ln_data = len(originalCsi[1,1,:])
    #originalCsi = originalCsi[:,:,1:ln_data:20] #cut  20 times
    #print('after size(originalCsi): %s\n',len(originalCsi[1,1,:]))
    print('after process originalCsi.shape     ::', originalCsi.shape)

    actionMark = outFileName[6] + outFileName[7]
    #A = mat_b55.Mat_b55(fn,b55)
    generateData,generateLabel = OneActionSampleFile(originalCsi,startEndPoints,actionMark)
    generateData= np.transpose(generateData,axes=[2,1,0,3])
    print('generateData.shape      ::', generateData.shape)
    print('generateLabel.shape     ::', generateLabel.shape)
    
    scio.savemat(segmentResultDir + outFileName.replace(".mat","_data.mat"), {'generateData':generateData})
    scio.savemat(segmentResultDir + outFileName.replace(".mat","_label.mat"), {'generateLabel':generateLabel})
    return  len(generateLabel)
    
def combineCsiLabel(saveDir,dataDir):
    # combine  multiple train samples into one file
    #os.chdir(root_path)
    file_DataList = []
    file_LbelList = []
    AllFileList=os.listdir(dataDir)
    for oneName in AllFileList:
        print(oneName[11:16])
        print(oneName)
        if(oneName[11:15] == 'data'):
            file_DataList.append(oneName)        
        if(oneName[11:16] == 'label'):
            file_LbelList.append(oneName)
            
    numberFiles = len(file_DataList)    
    generateData = []
    generateLabel = []
    for i in range(0,numberFiles):
        print('i    : %s -- fieName: %s ::',file_DataList[i])
        #generateData.append(joblib.load(dataDir + file_DataList[i]))
        #generateLabel.append(joblib.load(dataDir + file_LbelList[i]))
        
        mat = scio.loadmat(dataDir + file_DataList[i])
        data=mat['generateData']
            
        generateData.append(data)
        mat = scio.loadmat(dataDir + file_LbelList[i])
        data=mat['generateLabel']        
        generateLabel.append(data)

    actionTestCsi = generateData[0]
    actionTestLab = generateLabel[0]
    for i in range(1,numberFiles):
        print('actionTestCsi.shape       ::', actionTestCsi.shape)
        print('generateData[i].shape     ::', generateData[i].shape)
        actionTestCsi = np.concatenate((actionTestCsi,generateData[i]),axis = 3)
        actionTestLab = np.concatenate((actionTestLab,generateLabel[i]),axis = 0)

    #os.chdir(saveDir) #
    #joblib.dump(actionTestCsi,'actionTestCsi')
    #joblib.dump(actionTestLab,'actionTestLab')
    print('actionTestCsi.shape      ::', actionTestCsi.shape)
    print('actionTestLab.shape      ::', actionTestLab.shape)
    #scio.savemat(saveDir + 'actionTestCsi2', {'actionTestCsi':actionTestCsi})
    #scio.savemat(saveDir + 'actionTestLab2', {'actionTestLab':actionTestLab})
    dataTemp={}
    dataTemp['actionTestCsi']=actionTestCsi
    hdf5storage.write(dataTemp, filename= saveDir +'actionTestCsi.mat',store_python_metadata=True, matlab_compatible=True)
    #hdf5storage.write(dataTemp, filename= saveDir +'actionTestCsi.mat',matlab_compatible=True)
    labelTemp={}
    labelTemp['actionTestLab']=actionTestLab
    #hdf5storage.write(labelTemp, filename=saveDir +'actionTestLab.mat',store_python_metadata=False,matlab_compatible=True)    
    hdf5storage.write(labelTemp, filename=saveDir +'actionTestLab.mat',matlab_compatible=True)    
    
    #print('size of data is:',number)
    return
def main(_):

    #SIZE_LABEL = []
    #predResultDir = 'StateFile/'
    predResultDir = args.StateFile + '/'
    segmentResultDir='SegmentResultOneByOne/'
    saveDir='SegmentResultCombine/'
    filelist=os.listdir(segmentResultDir)  # delete all the files in segmentResultDir
    for f in filelist:   
        filepath = os.path.join(segmentResultDir,f) 
        if os.path.isfile(filepath):
            os.remove(filepath)        
            
    #allFileName = ['user1_iw_6','user1_ph_6','user1_rp_6','user1_sd_6','user1_wd_6']
    allFileName = []
    for root, dirs, files in os.walk(predResultDir):
        for name in files:
            allFileName.append(name.replace('_predict_ema',''))
            #formatTweets(os.path.join(root, name),fout)
            
            
    for outFileName in allFileName :
        print('-----------------', outFileName,'-----------------')
        #outFileName = 'user1_iw_6.mat'
        
        predictDataDir = predResultDir + outFileName + '_predict_ema' # 'user1_iw_6.mat_predict_ema'
        startEndPoints = actionStartEndPoints(predictDataDir)
        predictDataDir = moveStepForPoll(startEndPoints)
        #predictDataDir = predResultDir + 'user1_ph_6.mat_predict_ema'
        #predictDataDir = predResultDir + 'user1_rp_6.mat_predict_ema'
        #predictDataDir = predResultDir + 'user1_sd_6.mat_predict_ema'
        #predictDataDir = predResultDir + 'user1_wd_6.mat_predict_ema'
        data_dir = 'Data_CsiAmplitudeCut/'
        outFileName = outFileName + '.mat'
        originalFile= data_dir + outFileName[:5] +'/' + '55' + outFileName # 'Data_CsiAmplitudeCut/user1/55user1_iw_1.mat'
        size_labelOne = actionSampleExtract(startEndPoints,originalFile,outFileName,segmentResultDir)
        saveStartEndPoints(segmentResultDir,outFileName,startEndPoints) # save start and end points in .csv file
    
    #saveDir='SegmentResultDataCombine/'
    combineStartEndPoints(saveDir,segmentResultDir)    
    combineCsiLabel(saveDir,segmentResultDir)  #
    print('---------done------')
def moveStepForPoll(startEndPoints):
    moveStep =  0
    moveStep =  1
    moveStep =  2  # allData: testAcc2=87.50 testF12=89.60 testAcc2E=87.50 testF12E=89.47
    moveStep =  3  # allData: testAcc2=88.28 testF12=90.96 testAcc2E=88.67 testF12E=91.11
    moveStep =  4  # allData: testAcc2=89.06 testF12=91.81 testAcc2E=89.84 testF12E=91.88
    moveStep =  4
    len_startEndPoints=len(startEndPoints)
    for i in range(0,len_startEndPoints):
        startEndPoints[i,0] += moveStep
        startEndPoints[i,1] += moveStep
    return startEndPoints
def saveStartEndPoints(segmentResultDir,outFileName,startEndPoints):
    fStartEnd = open(segmentResultDir+ '/'+ outFileName.replace('.mat','.csv'),'w')
    len_A=len(startEndPoints) #10个动作
    for i in range(0,len_A):
        fStartEnd.write(str(i+1) + ',' + str(int(startEndPoints[i,0])) + ',' + str(int(startEndPoints[i,1])) + '\n')
    fStartEnd.close()
def combineStartEndPoints(saveDir,segmentResultDir):
    fUser1 = open(saveDir+ '/'+'user1ManualSegment.csv','w')
    fUser2 = open(saveDir+ '/'+'user2ManualSegment.csv','w')
    fUser3 = open(saveDir+ '/'+'user3ManualSegment.csv','w')
    fUser4 = open(saveDir+ '/'+'user4ManualSegment.csv','w')
    fUser5 = open(saveDir+ '/'+'user5ManualSegment.csv','w')
    fopenAll = [fUser1,fUser2,fUser3,fUser4,fUser5]
    allFiles = os.listdir(segmentResultDir)
    kk = 0
    for oneFile in allFiles:
        if(oneFile.split('.')[-1] == 'csv'):
            print(oneFile)
            fStartEnd = open(segmentResultDir+ '/'+ oneFile)
            for current in fStartEnd:    
                #print(math.floor(kk/30))
                fopenAll[math.floor(kk/30)].write(str(kk+1) + ',' + current.replace('\n',',') + oneFile.replace('.csv','') + '\n')            
            kk+=1 
    
if __name__ == '__main__':
    main(args)

