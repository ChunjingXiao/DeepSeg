import os
import time

import numpy as np
import tensorflow as tf
import sys
from sklearn import metrics
import math
import h5py

#from networkCNN import classifier


import argparse

#parser is used to accept parameters from commandlines,such as seting epoch=10:python train_CSI.py --epoch 10 
parser = argparse.ArgumentParser(description='')
#parser.add_argument('--epoch', dest='epoch', type=int, default=200, help='# of epoch')
parser.add_argument('--dataDir',   dest='dataDir', default='Data_DiscretizeCsi', help='directory of data')
parser.add_argument('--csiFile',   dest='csiFile', default='user1_wd_6.mat', help='CSI data File Name')
parser.add_argument('--labelFile', dest='labelFile', default='userSegmentLab12000.mat', help='Label data File Name')
# python 4test_5_inputParameter.py --csiFileName actionTestCsi.mat --labelFileName actionTestLab.mat
parser.add_argument('--modelDir', dest='modelDir', default='./saveModel/model-850', help='save model directory')
args = parser.parse_args()


flags = tf.app.flags
flags.DEFINE_integer('gpu', 0, 'gpu [0]')   # which GPU is used. If it is beyong the number of GPU, CPU will is used.
#flags.DEFINE_string('modelDir', './saveModel/model-850', 'save model directory')  #-------------modelDir-------------

#flags.DEFINE_integer('batch_size', 64, "batch size [25]")  
#flags.DEFINE_integer('seed', 10, 'seed numpy')
#flags.DEFINE_string('data_dir', './data/cifar-10-python/','data directory')
#flags.DEFINE_integer('category_number', 125, 'number of categories in the dataset [125]') #---categoryNum = 125----
#flags.DEFINE_integer('epoch', 1400, 'epochs [1400]')
#flags.DEFINE_integer('decay_start', 1200, 'start learning rate decay [1200]')
#flags.DEFINE_float('learning_rate', 0.01, 'learning_rate[0.0003]')

#flags.DEFINE_string('data_dir', './data/cifar-10-python/','data directory')

#flags.DEFINE_float('lbl_weight', 1.0, 'unlabeled weight [1.]')
#flags.DEFINE_float('ma_decay', 0.9999, 'exponential moving average for inference [0.9999]')
#flags.DEFINE_boolean('validation', False, 'validation [False]')  
#flags.DEFINE_boolean('clamp', False, 'validation [False]')
#flags.DEFINE_boolean('abs', False, 'validation [False]')
#flags.DEFINE_float('lmin', 1.0, 'unlabeled weight [1.]')
#flags.DEFINE_float('lmax', 1.0, 'unlabeled weight [1.]')
#flags.DEFINE_integer('nabla', 1, 'choose regularization [1]')  #-------------xiao-------------
#flags.DEFINE_float('gamma', 0.001, 'weight regularization')
#flags.DEFINE_float('alpha', 40., 'displacement along data manifold')  #-------------default epsilon is 20. ---------
#flags.DEFINE_float('eta', 1., 'perturbation latent code')
#flags.DEFINE_integer('freq_print', 10000, 'frequency image print tensorboard [10000]')
#flags.DEFINE_integer('step_print', 50, 'frequency scalar print tensorboard [50]')
#flags.DEFINE_integer('freq_test', 1, 'frequency test [500]')
#flags.DEFINE_integer('freq_save', 10, 'frequency saver epoch[50]')
FLAGS = flags.FLAGS


def scaler(x, feature_range=(-1, 1)):
    # scale to (0, 1)
    getMax = x.max()
    x = ((x - x.min())/(getMax - x.min()))
    # scale to feature_range
    min, max = feature_range
    x = x * (max - min) + min
    return x
def mat2Npy(data_dir, fileName,typeName):
    path= data_dir + fileName

    
    mat=h5py.File(path,'r') #   print(mat.keys())   
    fileName = list(mat.keys())[0] #print(list(mat.keys())[0])
    data=mat[fileName]
    #print(fileName)     #print(typeName)    #print(typeName == 'Csi')
    if(typeName == 'Csi'):
        dataReturn= np.transpose(data,axes=[0,3,2,1])
    elif(typeName == 'Label'):
        data= np.transpose(data,axes=[1,0])
        dataReturn=np.zeros(data.shape[0])
        for i in range(dataReturn.shape[0]):
            dataReturn[i]=data[i]-1    
    print(dataReturn.shape)
    return dataReturn
def loadDataTest(dataDir,csiFile,labelFile):
    #if not os.path.exists(FLAGS.logdir):
    #    os.makedirs(FLAGS.logdir)
    dataDir = dataDir + '/' #dataDir = 'data/'
    #testx= mat2Npy(dataDir,'actionTestCsi.mat','Csi')
    #testy= mat2Npy(dataDir,'actionTestLab.mat','Label') 
    testx= mat2Npy(dataDir,csiFile,'Csi')
    testy= mat2Npy('data/',labelFile,'Label') #testy= mat2Npy(dataDir,labelFile,'Label') 
    print('testx.shape      ::', testx.shape)
    print('testy.shape      ::', testy.shape) 
    #---------add for change scaler, which impacts results--------20191230-------begin--------
    # when the number of test data is very small, it needs this code. No needs for big testdata
    testAllForScalerX= mat2Npy('data/','segmentTestCsi.mat','Csi')
    testAllForScalerY= mat2Npy('data/','segmentTestLab.mat','Label') 
    testCsiCombine = np.concatenate((testx,testAllForScalerX),axis = 0)
    print('--testCsiCombine.shape      ::', testCsiCombine.shape)    
    testxTemp = scaler(testCsiCombine) 
    lenTextx = testx.shape[0]
    testx = testxTemp[0:lenTextx,:,:,:]
    print('testx.shape      ::', testx.shape)
    print('testy.shape      ::', testy.shape)   
    #---------add for change scaler, which impacts results--------20191230-------end----------
    
    #trainx = scaler(trainx)                         #-----------------xiao--------normalize data--------------
    #testx = scaler(testx)   
    
    #rng = np.random.RandomState(10)  # seed labels
    #rng_data = np.random.RandomState(rng.randint(0, 2**10))  # seed shuffling        
    #inds = rng_data.permutation(trainx.shape[0])   #-----------------xiao--------shuffling data-------------
    #trainx = trainx[inds]
    #trainy = trainy[inds]
    #inds = rng_data.permutation(testx.shape[0])
    #testx = testx[inds]
    #testy = testy[inds]   
    return (testx,testy) 

#the function to calculate entropy, you should use the probabilities as the parameters
def entropy(c):
    result=-1;
    if(len(c)>0):
        result=0;
    for x in c:
        if(x==0):
            x = 0.00000000001
        result += (-x)* math.log(x,2)
    return result;

def main(_):

    os.environ["CUDA_VISIBLE_DEVICES"] = str(FLAGS.gpu)  # ----------which GPU is used-----------
    
    
    #modelDir = "./saveModel/model-750"     # "./saveModel/model-750" https://blog.csdn.net/sjtuxx_lee/article/details/82663394 ----------------
    saver = tf.train.import_meta_graph(args.modelDir + ".meta")
    graph = tf.get_default_graph()
    #tensor_name_list = [tensor.name for tensor in graph.as_graph_def().node]    
    
    
    logits_lab = tf.get_collection('logits_lab_save')[0] #------------------add for save and load model-------20191127------xiao------------
    logits_ema = tf.get_collection('logits_ema_save')[0] #------------------add for save and load model-------20191127------xiao------------
    batch_size = tf.get_collection('batch_size_save')[0] #------------------add for save and load model-------20191127------xiao------------    
    
    
    
    
    
    
    #dataDir = args.dataDir  #   csiFile = args.csiFile   #  labelFile = args.labelFile
    (testx,testy) = loadDataTest(args.dataDir, args.csiFile,args.labelFile)
    
    nr_batches_test = int(testx.shape[0] / batch_size)
    #----if testx.shape[0] modulo  FLAGS.batch_size <> 0, there will be some samples to  be left, here fill some samples to testX------begin------
    #----for outputing full test samples------20191206-----------
    #print(testx.shape[0]/FLAGS.batch_size)
    modNumber = testx.shape[0] % batch_size
    if( modNumber != 0):
        testx2 = testx[0:batch_size-modNumber,:,:,:]
        testy2 = testy[0:batch_size-modNumber]
        #print(testx2.shape)
        testx = np.concatenate((testx,testx2), axis=0)
        testy = np.concatenate((testy,testy2), axis=0)
        #print(testx.shape)
        nr_batches_test += 1
    
    
    #----if testx.shape[0] modulo  batch_size <> 0, there will be some samples to  be left, here fill some samples to testX------end--------
    
    
    
    #print("Data:")
    print('test examples %d, batch %d' % (testx.shape[0], nr_batches_test))
    print('histogram test ', np.histogram(testy, bins=10)[0])
    #print("")
    
    

    
    inp = graph.get_operation_by_name('labeled_data_input_pl').outputs[0]#inp=gragh.get_tensor_by_name('labeled_data_input_pl')
    lbl = graph.get_operation_by_name('lbl_input_pl').outputs[0]
    is_training_pl = graph.get_operation_by_name('is_training_pl').outputs[0]
    

    
    correct_pred = tf.equal(tf.cast(tf.argmax(logits_lab, 1), tf.int32), tf.cast(lbl, tf.int32))
    accuracy_classifier = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    y_predict = tf.cast(tf.argmax(logits_lab, 1), tf.int32)
    
    correct_pred_ema = tf.equal(tf.cast(tf.argmax(logits_ema, 1), tf.int32), tf.cast(lbl, tf.int32))
    accuracy_ema = tf.reduce_mean(tf.cast(correct_pred_ema, tf.float32))
    y_predict_ema = tf.cast(tf.argmax(logits_ema, 1), tf.int32)    
    
    # --------added for computing entropy------xiao--------20191128-------begin------
    #fopen = open(args.csiFile.replace('.mat','') + "_entropy", "w")
    #fopen_ema = open(args.csiFile.replace('.mat','') + "_entropy_ema", "w")
    softmax_out = tf.nn.softmax(logits_lab, name='softmax_out')
    softmax_out_ema = tf.nn.softmax(logits_ema, name='softmax_out_ema')
    #fopen_predict_ema = open(args.csiFile.replace(".mat","") + "_predict_ema", "w")
    if(args.csiFile[8:10]=='_6'):
        stateLabelDir = 'StateLabel_DiscretizeCsi_only6/'
    else:
        stateLabelDir = 'StateLabel_DiscretizeCsi_1_5/'
    fopen_predict_ema = open(stateLabelDir + args.csiFile.replace(".mat","") + "_predict_ema", "w")
    # --------added for computing entropy------xiao--------20191128-------end--------
    acc_all=acc_ema_all=f1_all = f1_ema_all= precsion_all = recall_all = 0
    acc_all2=acc_ema_all2=f1_all2 = f1_ema_all2=0  #----for outputing full test samples------20191206-----------
    with tf.Session() as sess:
        begin = time.time()
        saver.restore(sess, args.modelDir) #saver.restore(sess, tf.train.latest_checkpoint("./saveModel"))
        print('finish loading model!')
        
        for t in range(nr_batches_test):
            ran_from = t * batch_size
            ran_to = (t + 1) * batch_size
            feed_dict = {inp: testx[ran_from:ran_to],
                         lbl: testy[ran_from:ran_to],
                         is_training_pl: False}
            
            #acc, acc_ema,y_pred,y_pred_ema = sess.run([accuracy_classifier, accuracy_ema,y_predict,y_predict_ema], feed_dict=feed_dict) # correct without entropy
            # --------added for computing entropy------xiao--------20191128-------begin------
            y_true = testy[ran_from:ran_to]
            
            acc, acc_ema,y_pred,y_pred_ema,softmax_entropy,softmax_entropy_ema = sess.run([accuracy_classifier, accuracy_ema,y_predict,y_predict_ema,softmax_out,softmax_out_ema], feed_dict=feed_dict) # correct without entropy
            #print("softmax_out_entropy:" + str(softmax_out_entropy))
            #----if testx.shape[0] modulo  batch_size <> 0, there will be some samples to  be left, here fill some samples to testX------begin------
            entropyShape = softmax_entropy.shape[0]
            entropyShapeEma = softmax_entropy_ema.shape[0]
            if(modNumber !=0 and t== nr_batches_test-1):
                entropyShape = modNumber
                entropyShapeEma = modNumber
                y_true = y_true[0:modNumber]
                y_pred = y_pred[0:modNumber]
                y_pred_ema = y_pred_ema[0:modNumber]
            #----if testx.shape[0] modulo  batch_size <> 0, there will be some samples to  be left, here fill some samples to testX------end--------
            
            '''
            for i in range(entropyShape):
                fopen.write(str(int(y_true[i])) +  "\t" + str(y_pred[i]) + "\t" + '{:.4f}'.format(max(softmax_entropy[i])) + '\t')                 
                fopen.write(str('{:.4f}'.format(entropy(softmax_entropy[i])) ) + '\t')
                for j in range(softmax_entropy.shape[1]):
                    fixFloat = '{:.4f}'.format(softmax_entropy[i][j])
                    fopen.write(str(fixFloat) + "\t")
                fopen.write("\n")
        
            for i in range(entropyShapeEma):
                fopen_ema.write(str(int(y_true[i])) +  "\t" + str(y_pred_ema[i]) + "\t" + '{:.4f}'.format(max(softmax_entropy_ema[i])) + '\t') 
                fopen_ema.write(str('{:.4f}'.format(entropy(softmax_entropy_ema[i])) ) + '\t')
                for j in range(softmax_entropy_ema.shape[1]):
                    fixFloat = '{:.4f}'.format(softmax_entropy_ema[i][j])
                    fopen_ema.write(str(fixFloat) + "\t")
                fopen_ema.write("\n")
            '''    
            for i in range(entropyShapeEma):
                fopen_predict_ema.write( str(y_pred_ema[i] +1 ) + "\t" + '{:.4f}'.format(max(softmax_entropy_ema[i])) 
                                         + "\t" + str(y_pred[i] +1 )+ "\t" + '{:.4f}'.format(max(softmax_entropy[i])) + "\n")
            
            # --------added for computing entropy------xiao--------20191128-------end--------
            acc2= metrics.accuracy_score(y_true, y_pred)          #----for outputing full test samples------20191206-----------
            acc2_ema= metrics.accuracy_score(y_true, y_pred_ema)  #----for outputing full test samples------20191206-----------
            #print('acc:' + str(acc) + '\t' + 'acc2:' + str(acc2))
            #print('acc_ema:' + str(acc_ema) + '\t' + 'acc2_ema:' + str(acc2_ema))
            f1 = metrics.f1_score(y_true, y_pred,average="weighted")         # weighted  macro  micro
            f1_ema = metrics.f1_score(y_true, y_pred_ema,average="weighted") #
            precsion = metrics.precision_score(y_true, y_pred,average="micro")
            recall = metrics.recall_score(y_true, y_pred,average="micro")
            acc_all2 += acc2                                      #----for outputing full test samples------20191206-----------
            acc_ema_all2 += acc2_ema                              #----for outputing full test samples------20191206-----------
            if(not (modNumber !=0 and t== nr_batches_test-1)):    #----for outputing full test samples------20191206-----------
                acc_all += acc
                acc_ema_all += acc_ema        
                f1_all += f1
                f1_ema_all += f1_ema               
                precsion_all += precsion
                recall_all += recall
        nr_batches_test2 = nr_batches_test                        #----for outputing full test samples------20191206-----------
        if(modNumber !=0):                                        #----for outputing full test samples------20191206-----------
            nr_batches_test -= 1
        acc_all2 /= nr_batches_test2                              #----for outputing full test samples------20191206-----------      
        acc_ema_all2 /= nr_batches_test2                          #----for outputing full test samples------20191206-----------
        
        acc_all /= nr_batches_test
        acc_ema_all /= nr_batches_test
        f1_all /= nr_batches_test
        f1_ema_all /= nr_batches_test 
        precsion_all /= nr_batches_test
        recall_all /= nr_batches_test

        #print("%ds testAcc=%.2f testF1=%0.2f testAccE=%0.2f testF1E=%0.2f" 
        #      % (time.time()-begin, acc_all*100,f1_all*100,acc_ema_all*100,f1_ema_all*100))
        print("%ds testAcc=%.2f testF1=%0.2f testAccE=%0.2f testF1E=%0.2f --- testAcc2=%0.2f testAcc2E=%0.2f" 
              % (time.time()-begin, acc_all*100,f1_all*100,acc_ema_all*100,f1_ema_all*100, acc_all2*100, acc_ema_all2*100))               
        # acc_all is the result excluding the last epoch if testx.shape[0]%batch_size <> 0
        # acc_all2 is the result including all the epoches



if __name__ == '__main__':
    tf.app.run()
