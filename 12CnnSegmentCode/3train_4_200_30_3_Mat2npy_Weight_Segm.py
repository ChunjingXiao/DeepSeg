import os
import time

import numpy as np
import tensorflow as tf
import sys
from sklearn import metrics
import h5py

from networkCNN_Seg120 import classifier


import argparse
#parser is used to accept parameters from commandlines,such as seting epoch=10:python train_CSI.py --epoch 10 
parser = argparse.ArgumentParser(description='')
#parser.add_argument('--epoch', dest='epoch', type=int, default=200, help='# of epoch')
parser.add_argument('--dataDir',    dest='dataDir',  default='data',              help='directory of data')
parser.add_argument('--trainCsi',   dest='trainCsi', default='segmentTrainCsi.mat', help='CSI data File For Train')
parser.add_argument('--trainLab',   dest='trainLab', default='segmentTrainLab.mat', help='Label data File For Train')
parser.add_argument('--testCsi',    dest='testCsi',  default='segmentTestCsi.mat', help='CSI data File For Test')
parser.add_argument('--testLab',    dest='testLab',  default='segmentTestLab.mat', help='Label data File For Test')
parser.add_argument('--lr', dest='learning_rate', type=float, default=0.009, help='initial learning rate for adam [0.0003]')
parser.add_argument('--wtrain', dest='wtrain', type=bool, default=False, help='if input any values, this will be true')
args = parser.parse_args()
'''
if not os.path.exists(args.checkpoint_dir):
    os.makedirs(args.checkpoint_dir)
if not os.path.exists(args.sample_dir):
    os.makedirs(args.sample_dir)
if not os.path.exists(args.test_dir):
    os.makedirs(args.test_dir)
'''


flags = tf.app.flags
flags.DEFINE_integer('gpu', 0, 'gpu [0]')   # which GPU is used. If it is beyong the number of GPU, CPU will is used.
flags.DEFINE_integer('batch_size', 16, "batch size [25]")  # --------60:0.8417(1169)-------70:0.8597--------------
flags.DEFINE_integer('category_number', 10, 'number of categories in the dataset [125]') #---categoryNum = 125----
flags.DEFINE_integer('epoch', 1600, 'epochs [1400]')
flags.DEFINE_integer('decay_start', 1200, 'start learning rate decay [1200]')
#flags.DEFINE_float('learning_rate', 0.0009, 'learning_rate[0.0003]')

#flags.DEFINE_string('data_dir', './data/cifar-10-python/','data directory')
flags.DEFINE_string('saveDir', './saveModel', 'save model directory')
flags.DEFINE_integer('seed', 10, 'seed numpy')
flags.DEFINE_float('lbl_weight', 1.0, 'unlabeled weight [1.]')
flags.DEFINE_float('ma_decay', 0.9999, 'exponential moving average for inference [0.9999]')
#flags.DEFINE_boolean('validation', False, 'validation [False]')  
flags.DEFINE_boolean('clamp', False, 'validation [False]')
flags.DEFINE_boolean('abs', False, 'validation [False]')
flags.DEFINE_float('lmin', 1.0, 'unlabeled weight [1.]')
flags.DEFINE_float('lmax', 1.0, 'unlabeled weight [1.]')
flags.DEFINE_integer('nabla', 1, 'choose regularization [1]') 
flags.DEFINE_float('gamma', 0.001, 'weight regularization')
#flags.DEFINE_float('alpha', 40., 'displacement along data manifold') 
flags.DEFINE_float('eta', 1., 'perturbation latent code')
flags.DEFINE_integer('freq_print', 10000, 'frequency image print tensorboard [10000]')
flags.DEFINE_integer('step_print', 50, 'frequency scalar print tensorboard [50]')
flags.DEFINE_integer('freq_test', 1, 'frequency test [500]')
flags.DEFINE_integer('freq_save', 50, 'frequency saver epoch[50]')
FLAGS = flags.FLAGS



def get_getter(ema):
    def ema_getter(getter, name, *args, **kwargs):
        var = getter(name, *args, **kwargs)
        ema_var = ema.average(var)
        return ema_var if ema_var else var
    return ema_getter


def display_progression_epoch(j, id_max):
    batch_progression = int((j / id_max) * 100)
    sys.stdout.write(str(batch_progression) + ' % epoch' + chr(13))
    _ = sys.stdout.flush


def linear_decay(decay_start, decay_end, epoch):
    return min(-1 / (decay_end - decay_start) * epoch + 1 + decay_start / (decay_end - decay_start),1)


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

    mat=h5py.File(path,'r') #   print(mat.keys()) # mat = sio.loadmat(path)  
    fileName = list(mat.keys())[0] #print(list(mat.keys())[0]) #fileName = fileName.replace('.mat','')  
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
def loadData(dataDir,trainCsi,trainLab, testCsi, testLab):
    #if not os.path.exists(FLAGS.logdir):
    #    os.makedirs(FLAGS.logdir)
    dataDir = dataDir + '/' #dataDir = 'data/'
    #testx= mat2Npy(dataDir,'actionTestCsi.mat','Csi')
    #testy= mat2Npy(dataDir,'actionTestLab.mat','Label') 
    trainx= mat2Npy(dataDir,trainCsi,'Csi')
    trainy= mat2Npy(dataDir,trainLab,'Label')     
    testx= mat2Npy(dataDir,testCsi,'Csi')
    testy= mat2Npy(dataDir,testLab,'Label') 
    '''   # load Data    
    data_dir = 'data/'
    trainx=np.load(data_dir+'actionBaseTrainCsi.npy')   #-----------------xiao------load data--------------------
    trainy=np.load(data_dir+'actionBaseTrainLab.npy')
    testx=np.load(data_dir+'actionTestCsi.npy')
    testy=np.load(data_dir+'actionTestLab.npy')
    '''
    print('trainx.shape     ::', trainx.shape)
    print('trainy.shape     ::', trainy.shape)
    print('testx.shape      ::', testx.shape)
    print('testy.shape      ::', testy.shape)      
    
    trainx = scaler(trainx)                         #-----------------xiao--------normalize data--------------
    testx = scaler(testx)  
    '''
    # Random seed
    rng = np.random.RandomState(FLAGS.seed)  # seed labels
    rng_data = np.random.RandomState(rng.randint(0, 2**10))  # seed shuffling      
    inds = rng_data.permutation(trainx.shape[0])   #-----------------xiao--------shuffling data-------------
    trainx = trainx[inds]
    trainy = trainy[inds]
    inds = rng_data.permutation(testx.shape[0])
    testx = testx[inds]
    testy = testy[inds]
    '''
    return (trainx,trainy,testx,testy)      
def main(_):
    '''
    print("\nParameters:")
    for attr,value in tf.app.flags.FLAGS.flag_values_dict().items():
        print("{}={}".format(attr,value))
    print("")
    '''
    os.environ["CUDA_VISIBLE_DEVICES"] = str(FLAGS.gpu)  # ----------which GPU is used-----------
    rng = np.random.RandomState(FLAGS.seed)  # seed labels
    
    (trainx,trainy,testx,testy) = loadData(args.dataDir, args.trainCsi,args.trainLab,args.testCsi,args.testLab)
    rng_data = np.random.RandomState(rng.randint(0, 2**10))  # seed shuffling    
    inds = rng_data.permutation(trainx.shape[0])   #-----------------xiao--------shuffling data-------------
    trainx = trainx[inds]
    trainy = trainy[inds]
    
    if(args.wtrain):         #--------------------------add for sample Weight-------20191214------xiao------------   
        sWeight = np.loadtxt(args.dataDir + '/' +'actionTrainCsi_entropy_ema')
        print('sWeight.shape    ::', sWeight.shape) 
        sWeight= sWeight[inds]
    inds = rng_data.permutation(testx.shape[0])
    testx = testx[inds]
    testy = testy[inds]

    
    nr_batches_train = int(trainx.shape[0] / FLAGS.batch_size)
    nr_batches_test = int(testx.shape[0] / FLAGS.batch_size)
    print("args.wtrain=%s, args.learning_rate=%.4f, args.trainCsi=%s" % (args.wtrain, args.learning_rate, args.trainCsi)) 
    
    print("Data:")
    print('train examples %d, batch %d, test examples %d, batch %d' % (trainx.shape[0], nr_batches_train, testx.shape[0], nr_batches_test))
    print('histogram train', np.histogram(trainy, bins=10)[0])
    print('histogram test ', np.histogram(testy, bins=10)[0])
    print("histogram labeled", np.histogram(trainy, bins=10)[0])
    print("")
    
    
    
    '''construct graph'''
    is_training_pl = tf.placeholder(tf.bool, [], name='is_training_pl')
    inp = tf.placeholder(tf.float32, [FLAGS.batch_size, trainx.shape[1], 30, 3], name='labeled_data_input_pl') # ------data format------  
    lbl = tf.placeholder(tf.int32, [FLAGS.batch_size], name='lbl_input_pl')
    # scalar pl
    lr_pl = tf.placeholder(tf.float32, [], name='learning_rate_pl')
    acc_train_pl = tf.placeholder(tf.float32, [], 'acc_train_pl')
    acc_test_pl = tf.placeholder(tf.float32, [], 'acc_test_pl')
    acc_test_pl_ema = tf.placeholder(tf.float32, [], 'acc_test_pl')

    classifier(inp, is_training_pl, init=True,category=FLAGS.category_number) # initiate classifier
    
    logits_lab, layer_label = classifier(inp, is_training_pl, init=False, reuse=True,category=FLAGS.category_number)  # labeled_data_input_pl
    tf.add_to_collection('logits_lab_save', logits_lab)        #------------------add for save and load model-------20191127------xiao------------
    tf.add_to_collection('batch_size_save', FLAGS.batch_size)  #------------------add for save and load model-------20191127------xiao------------
    correct_pred = tf.equal(tf.cast(tf.argmax(logits_lab, 1), tf.int32), tf.cast(lbl, tf.int32))
    accuracy_classifier = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    y_predict = tf.cast(tf.argmax(logits_lab, 1), tf.int32)
    
    with tf.name_scope('loss_functions'):
        
        if(args.wtrain):         #--------------------------add for sample Weight-------20191214------xiao------------   
            sampleWeight = tf.placeholder(tf.float32, [FLAGS.batch_size], name='weight_for_loss')
            loss_lab = tf.reduce_mean(tf.multiply(sampleWeight,tf.nn.sparse_softmax_cross_entropy_with_logits(labels=lbl, logits=logits_lab))) 
        else:
            loss_lab = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=lbl, logits=logits_lab)) # labeled_data_input_pl
        loss_class = FLAGS.lbl_weight * loss_lab    #------------consider manifold to improve performance-------------------
        
    with tf.name_scope('optimizers'):
        # control op dependencies for batch norm and trainable variables
        tvars = tf.trainable_variables()
        dvars = [var for var in tvars if 'discriminator_model' in var.name]

        #update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        #update_ops_dis = [x for x in update_ops if ('discriminator_model' in x.name)]
        optimizer_class = tf.train.AdamOptimizer(learning_rate=lr_pl, beta1=0.5, name='class_optimizer')
       
        class_op = optimizer_class.minimize(loss_class, var_list=dvars)
        ema = tf.train.ExponentialMovingAverage(decay=FLAGS.ma_decay)
        maintain_averages_op = ema.apply(dvars)

        with tf.control_dependencies([class_op]):
            train_class_op = tf.group(maintain_averages_op)

        logits_ema, _ = classifier(inp, is_training_pl, getter=get_getter(ema), reuse=True,category=FLAGS.category_number)
        tf.add_to_collection('logits_ema_save', logits_ema)  #------------------add for save and load model-------20191127------xiao------------
        correct_pred_ema = tf.equal(tf.cast(tf.argmax(logits_ema, 1), tf.int32), tf.cast(lbl, tf.int32))
        accuracy_ema = tf.reduce_mean(tf.cast(correct_pred_ema, tf.float32))
        y_predict_ema = tf.cast(tf.argmax(logits_ema, 1), tf.int32)
    # training global varialble
    global_epoch = tf.Variable(0, trainable=False, name='global_epoch')
    global_step = tf.Variable(0, trainable=False, name='global_step')
    inc_global_step = tf.assign(global_step, global_step+1)
    inc_global_epoch = tf.assign(global_epoch, global_epoch+1)


    
    max_acc = tf.Variable([0.000001,0.0])              #----------[maxAcc, epoch]-------------------xiao----save max accuracy-----------
    accTemp = tf.placeholder(tf.float32,shape=(2)) 
    maxAccAssign = tf.assign(max_acc,accTemp)
    max_acc_ema = tf.Variable([0.000001,0.000001,0.0]) #----------[maxAcc_ema, f1_ema, epoch]-------xiao----save max ema----------------
    accTemp_ema = tf.placeholder(tf.float32,shape=(3)) 
    maxAccAssign_ema = tf.assign(max_acc_ema,accTemp_ema)
   
    op = tf.global_variables_initializer()
    init_feed_dict = {inp: trainx[:FLAGS.batch_size], is_training_pl: True}
    saveModel = tf.train.Saver(max_to_keep = 2000)    #----------to overcome the thing that Saver has 5 models limit--------------------
    sv = tf.train.Supervisor(global_step=global_epoch, summary_op=None, save_model_secs=0,init_op=op,init_feed_dict=init_feed_dict)
    print('start training')
    with sv.managed_session() as sess:   # tf.train.Supervisor----------------mainly used for saving model-------------xiao-------------
        tf.set_random_seed(rng.randint(2 ** 10))  # tf.train.Supervisor-------https://www.cnblogs.com/zhouyang209117/p/7088051.html
        print('\ninitialization done')            # tf.train.Supervisor-------https://blog.csdn.net/qq_37008037/article/details/86236808
        print('Starting training from epoch :%d, step:%d \n'%(sess.run(global_epoch),sess.run(global_step)))

        #writer = tf.summary.FileWriter(FLAGS.logdir, sess.graph)
        #sv.saver(max_to_keep=2000)

        while not sv.should_stop():
            epoch = sess.run(global_epoch)
            train_batch = sess.run(global_step)
            

            if (epoch >= FLAGS.epoch):
                print("Training done")
                sv.stop()
                break

            begin = time.time()
            train_loss_lab=train_acc=test_acc=test_acc_ema=train_j_loss = 0
            lr = args.learning_rate #lr = FLAGS.learning_rate * linear_decay(FLAGS.decay_start,FLAGS.epoch,epoch)
            precsionAll = recallAll = f1All = acc2All= f1All_ema= 0


            # construct randomly permuted batches
            inds = rng.permutation(trainx.shape[0])
            trainx = trainx[inds]
            trainy = trainy[inds]
            if(args.wtrain):         #--------------------------add for sample Weight-------20191214------xiao------------      
                sWeight = sWeight[inds] 
            # training
            for t in range(nr_batches_train):

                display_progression_epoch(t, nr_batches_train)
                ran_from = t * FLAGS.batch_size
                ran_to = (t + 1) * FLAGS.batch_size
                

                # train classifier
                if(args.wtrain):         #--------------------------add for sample Weight-------20191214------xiao------------                    
                    feed_dict = {is_training_pl: True,inp: trainx[ran_from:ran_to],lbl: trainy[ran_from:ran_to],lr_pl: lr,
                                 sampleWeight:sWeight[ran_from:ran_to]}   
                else:
                    feed_dict = {is_training_pl: True,inp: trainx[ran_from:ran_to],lbl: trainy[ran_from:ran_to],lr_pl: lr}
                _, acc, lb = sess.run([train_class_op, accuracy_classifier, loss_lab],feed_dict=feed_dict)
               
                train_loss_lab += lb
                train_acc += acc
               
                if (train_batch % FLAGS.freq_print == 0) & (train_batch != 0):
                    ran_from = np.random.randint(0, trainx.shape[0] - FLAGS.batch_size)
                    ran_to = ran_from + FLAGS.batch_size
                    #sm = sess.run(sum_op_im,feed_dict={is_training_pl: True, unl: trainx_unl[ran_from:ran_to]})
                    #writer.add_summary(sm, train_batch)

                train_batch += 1
                sess.run(inc_global_step)

            train_loss_lab /= nr_batches_train
            train_acc /= nr_batches_train

            # Testing moving averaged model and raw model
            if (epoch % FLAGS.freq_test == 0) | (epoch == FLAGS.epoch-1):
                for t in range(nr_batches_test):
                    ran_from = t * FLAGS.batch_size
                    ran_to = (t + 1) * FLAGS.batch_size
                    feed_dict = {inp: testx[ran_from:ran_to],
                                 lbl: testy[ran_from:ran_to],
                                 is_training_pl: False}
                    #acc, acc_ema = sess.run([accuracy_classifier, accuracy_ema], feed_dict=feed_dict)
                    acc, acc_ema,y_pred,y_pred_ema = sess.run([accuracy_classifier, accuracy_ema,y_predict,y_predict_ema], feed_dict=feed_dict)
                    y_true = testy[ran_from:ran_to]
                    f1 = metrics.f1_score(y_true, y_pred,average="weighted")      # weighted  macro  micro
                    f1_ema = metrics.f1_score(y_true, y_pred_ema,average="weighted") #acc2= metrics.accuracy_score(y_true, y_pred) 
                    precsion = metrics.precision_score(y_true, y_pred,average="micro")
                    recall = metrics.recall_score(y_true, y_pred,average="micro")           
                
                    f1All += f1
                    f1All_ema += f1_ema # acc2All += acc2                  
                    precsionAll += precsion
                    recallAll += recall                
                    
                    test_acc += acc
                    test_acc_ema += acc_ema
                test_acc /= nr_batches_test
                test_acc_ema /= nr_batches_test

                f1All /= nr_batches_test
                f1All_ema /= nr_batches_test # acc2All /= nr_batches_test  
                precsionAll /= nr_batches_test
                recallAll /= nr_batches_test
                
                accMax = sess.run(max_acc)
                if(accMax[0] < test_acc):
                    accMax = [test_acc,epoch]
                    sess.run(maxAccAssign,feed_dict={accTemp:[test_acc,epoch]})
                accMax_ema = sess.run(max_acc_ema)
                if(accMax_ema[0] < test_acc_ema):
                    accMax_ema = [test_acc_ema,f1All_ema,epoch]
                    sess.run(maxAccAssign_ema,feed_dict={accTemp_ema:[test_acc_ema,f1All_ema,epoch]})
                    
                print("Ep%d %ds lossLab=%.4f trainAcc=%.4f testAcc=%.2f maxAcc=%0.2f Ep%d testAccE=%0.2f testF1E=%0.2f maxAccE=%0.2f maxF1E=%0.2f Ep%d"
                      % (epoch, time.time()-begin, train_loss_lab, train_acc,test_acc*100,accMax[0]*100, accMax[1], \
                         test_acc_ema*100,f1All_ema*100, accMax_ema[0]*100,accMax_ema[1]*100, accMax_ema[2]))
              
            else:
                print("Epoch%d %ds" % (epoch, time.time() - begin))                   
                

            sess.run(inc_global_epoch)

            # save snapshots of model
            if ((epoch % FLAGS.freq_save == 0)) | (epoch == FLAGS.epoch-1):
                string = 'model-' + str(epoch)
                save_path = os.path.join(FLAGS.saveDir, string)
                #sv.saver.save(sess, save_path)
                saveModel.save(sess, save_path)  #----------to overcome the thing that Saver has 5 models limit---------------------
                print("Model saved in file: %s" % (save_path))
            


if __name__ == '__main__':
    tf.app.run()
