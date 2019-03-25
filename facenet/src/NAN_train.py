#!/usr/bin/env python
#encoding=utf8
from feature_embedding import *
import random
import numpy as np
import os, stat,sys
import tensorflow as tf
import build_dataset
import shutil
from tensorflow.python.framework import graph_util
from datetime import datetime

MAX_IAMGESET_SIZE=200 #max image set size
MAX_ITER_STEPS=5000 #max iterate steps
TEST_ITERVAL=500 #test interval
SAVE_ITERVAL=2000 #save iterval

Lr = 0.01 #learning rate
d=512 #feature dimentions
momentumn=0.9



modelfile=os.path.expanduser("~/models/facenet/20180402-114759/20180402-114759.pb")
#embd_module=FeatureEembedding(modelfile)
embd_module=None

def get_batch(train_data, batch_index, BATCH_SIZE=1, FEATURE=False):
    '''
    get a batch image data and embed them
    return the features and label_one_hot
    of one batch
    '''
    nrof_examples = len(train_data)
    j = (batch_index*BATCH_SIZE) % nrof_examples
    
    label=[]
    if j+BATCH_SIZE<=nrof_examples:
        batch = train_data[j:j+BATCH_SIZE]
        label=range(j,j+BATCH_SIZE)
    else:
        batch = train_data[j:nrof_examples]
        label=range(j,nrof_examples)
        x2 = train_data[0:BATCH_SIZE-nrof_examples+j]
        label.extend(0,BATCH_SIZE-nrof_examples+j)
        batch.extend(x2)
    
    '''build the one-hot output label'''
    label_one_hot=np.zeros((BATCH_SIZE,len(train_data)))
    for k in range(BATCH_SIZE):
        label_one_hot[k,label[k]]=1
    
    ''''   
    load features
    random sample from medias
    for each class random sample a subclass which means a media
    read and embed them
    '''
    features=[None]*BATCH_SIZE #BATCH_SIZE*m*d
    len_medias=[] #store the media size of each template
    for i in range(BATCH_SIZE):
        media=random.sample(batch[i].image_paths,1)
        if FEATURE:
            emb=np.load(media[0])
            emb=emb/np.sqrt(np.sum(np.square(emb),axis=1).reshape(((emb.shape[0]),1)))#normalized to unit vector
            #print(emb.shape)
            features[i]=emb
            len_medias.append(emb.shape[0])
        else:
            #print(batch[i].image_paths)
            imageset=random.sample(media[0],min(len(media[0]),MAX_IAMGESET_SIZE))
            #print(imageset)
            fea=np.zeros((MAX_IAMGESET_SIZE,d))
            emb=embd_module.embed(imageset)
            emb=emb/np.sqrt(np.sum(np.square(emb),axis=1).reshape(((emb.shape[0]),1)))#normalized to unit vector
            fea[0:len(imageset),:]=emb
            len_medias.append(len(imageset))
            features[i]=fea

    features=np.stack(features)
    return features, label_one_hot, np.array(len_medias,dtype=np.int32).reshape((1,len(len_medias)))

NAN_graph=tf.Graph()

def NAN_module(x):
    '''
    input the features of a batch imageset dim=(batch_size*max_imageset_size*d)
    output the feature aggregated dim=(batch_size*d)
    '''

    #attention block 1
    q_0=tf.Variable(tf.random_normal([1,d],stddev=0.1),dtype=tf.float32)
    q_0_t=tf.transpose(q_0)
    e_0=tf.matmul(x, q_0_t)
    a_0=tf.nn.softmax(tf.transpose(e_0))
    a_0=tf.transpose(a_0)
    
    r_0=tf.reduce_sum(x*a_0,0)

    #attention block 2
    r_0=tf.reshape(r_0,[1,d])
    w=tf.Variable(tf.random_normal([d,d],stddev=0.1),dtype=tf.float32)
    bia=tf.Variable(tf.constant(0.1, shape=[d]))
    
    q_1=tf.nn.tanh(tf.matmul(r_0, w)+bia)
    q_1_t = tf.transpose(q_1)
    e_1=tf.matmul(x, q_1_t)
    a_1=tf.nn.softmax(tf.transpose(e_1))
    a_1=tf.transpose(a_1,name='score')

    r_1=tf.reduce_sum(x*a_1,0,name='NAN_feature')

    return r_1


global log_file

def train_with_softmax(train_data, valid_set=None):
    log_file=open("training_log.txt","w")
    #model define
    train_number=len(train_data)

    #define the module
    with NAN_graph.as_default():
        x=tf.placeholder(tf.float32, shape=(None,d), name="x_input")
        
        y_=tf.placeholder(tf.float32, shape=(None,train_number), name="y_output")
        
        #aggregation module
        r_1=NAN_module(x)
        
        #fc_layer+softmax
        w_fc=tf.Variable(tf.random_normal([d,train_number],stddev=0.1),dtype=tf.float32)
        bia_fc=tf.Variable(tf.constant(0.1, shape=[train_number]))
        fc=tf.add(tf.matmul(tf.reshape(r_1,[1,d]),w_fc),bia_fc,name="output")
        
        loss=tf.nn.softmax_cross_entropy_with_logits(labels=y_,logits=fc)
        
        train_step=tf.train.AdamOptimizer(Lr).minimize(loss)

        saver=tf.train.Saver()
        save_path=os.path.expanduser("~/models/NAN_model_with_raw_train_model")
        if not os.path.exists(save_path):
            os.makedirs(save_path)
            print("note: create save directory: %s"%save_path)

    time_stamp = datetime.strftime(datetime.now(), '%Y%m%d-%H%M%S')
    log_dir=os.path.expanduser("~/logs/NAN_video_faces")
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    #initialize train
    with NAN_graph.as_default():
        with tf.Session() as sess:
            summary_writer = tf.summary.FileWriter(log_dir + '/train_%s'%time_stamp, sess.graph)

            init_opt=tf.initialize_all_variables()
            sess.run(init_opt)
            for i in range(MAX_ITER_STEPS):
                feas, labels, len_medias=get_batch(train_data,i,FEATURE=True)

                feed_dict={x: feas[0], y_:labels}
                sess.run(train_step,feed_dict)

                #test
                if (i+1)%TEST_ITERVAL==0:
                    total_loss_train=0
                    for j in range(len(train_data)):
                        feas, labels,_ = get_batch(train_data,j,FEATURE=True)
                        feed_dict={x: feas[0], y_:labels}
                        total_loss_train+=np.sum(sess.run(loss, feed_dict),axis=0)
                    total_loss_train_mean=total_loss_train/len(train_data)
                    print("after %d training steps, loss in training data is: %g"%(i+1, total_loss_train_mean))

                    summary=tf.Summary()
                    summary.value.add(tag='train/entropy_loss', simple_value=total_loss_train_mean)
                    summary_writer.add_summary(summary, global_step=i)

                    log_file.write("%g\n"%(total_loss_train_mean))

                if (i+1)%SAVE_ITERVAL==0:
                    save_file=os.path.join(save_path,"NAN_models_%d.ckpt"%(i+1))
                    saver.save(sess, save_file)
                    print("save model to %s"%save_file)

            constant_graph = graph_util.convert_variables_to_constants(sess, sess.graph_def, ["output"])
            with tf.gfile.FastGFile(save_path + '/%s_NAN_%d_model.pb'%(time_stamp, MAX_ITER_STEPS), mode='wb') as f:
                f.write(constant_graph.SerializeToString())

        
def main():
    os.environ["CUDA_VISIBLE_DEVICES"]='0'
    #dataset_path=os.path.expanduser("~/dataset/aligned_images_DB_YTF")
    #dataset_path=os.path.expanduser("~/dataset/ytf_features_128_align")
    dataset_path=os.path.expanduser("/home/xuhao/dataset/video-faces/face_cap_cluster_aug_align_raw_train_30_epochs")

    dataset=facenet.get_fea_dataset(dataset_path)
    print("size of dataset is: %d"%len(dataset))

    train_set, test_set = facenet.split_dataset(dataset,0.95,mode='SPLIT_CLASSES')
    #test_set, valid_set = facenet.split_dataset(test_set,0.5,mode='SPLIT_CLASSES')

    test_set_store_path=os.path.expanduser("~/dataset/NAN/NAN_test")

    if not os.path.exists(test_set_store_path):
        os.makedirs(test_set_store_path)
    else:
        shutil.rmtree(test_set_store_path)
        os.makedirs(test_set_store_path)

    print("size of train set is: %d"%len(train_set))
    print("size of test set is: %d"%len(test_set))

    build_dataset.store(test_set_store_path, test_set)
    print("store the test data set to %s"%test_set_store_path)

    train_with_softmax(train_set)
    
if __name__=="__main__":
    main()
    log_file.close()
    

