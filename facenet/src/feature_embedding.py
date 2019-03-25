#!/usr/bin/env python
#encoding=utf8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from scipy import misc
import tensorflow as tf
import numpy as np
import sys
import os
import argparse
from facenet.src.facenet import *
import align.detect_face
from time import time
import random
import shutil
from os.path import *


MAX_IAMGESET_SIZE=200

class FeatureEembedding():
    def __init__(self, modelfile):
        modelfile_exp=os.path.expanduser(modelfile)
        if not os.path.exists(modelfile_exp):
            sys.stderr.write("error: no such file: %s\n"%modelfile_exp)
            exit(1)
        sys.stderr.write("note: embedding module initializing...\n")
        self.graph=tf.Graph()
        with self.graph.as_default():
            with tf.Session() as sess:
                start=time()
                load_model(modelfile_exp)
                self.images_placeholder=tf.get_default_graph().get_tensor_by_name("input:0")
                self.embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
                self.phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
                end=time()
                sys.stderr.write("note: model loaded success \ncost time: %ds \n"%(end-start))
        self.config()
        sys.stderr.write("note: embeding module initialized success\n")
    def embed_dataset(self, dataset, store_path,align=True):
        with self.graph.as_default():
            with tf.Session() as sess: 
                count=0
                for identity in dataset:
                    if len(identity.image_paths)==0:
                        print("skip one class %s"%identity.name)
                        continue
                    emb_save_filename = os.path.join(store_path, identity.name)
                    if identity.subnames:
                        for i, paths in enumerate(identity.image_paths):
                            if len(paths)==0:
                                print("skip class %s for index %d"%(identity.name, i))
                                continue
                            if not align:
                                images=load_image_data(paths[:min(len(paths),200)], self.image_size)
                            else:
                                images=load_and_align_data(paths[:min(len(paths),200)], self.image_size, self.margin, self.gpu_memory_fraction)
                            if not os.path.exists(emb_save_filename):
                                os.makedirs(emb_save_filename)
                            feed_dict = {self.images_placeholder: images, self.phase_train_placeholder: False}
                            embeddings = sess.run(self.embeddings, feed_dict=feed_dict)
                            np.save(os.path.join(emb_save_filename, identity.subnames[i]),embeddings)
                        count+=1
                        print("%d th features have computed"%(count))

                    else:
                        if not align:
                            images=load_image_data(identity.image_paths,self.image_size)
                        else:
                            images=load_and_align_data(identity.image_paths, self.image_size, self.margin, self.gpu_memory_fraction)
                        feed_dict={self.images_placeholder: images, self.phase_train_placeholder:False}
                        embeddings=sess.run(self.embeddings, feed_dict=feed_dict)
                        np.save(emb_save_filename,embeddings)
                        count+=1
                        print("%d th features have computed"%(count))
                
                             
    def embed(self,images_paths,align=True):
        if not align:
            images=load_image_data(images_paths,self.image_size)
        else:
            images=load_and_align_data(images_paths, self.image_size, self.margin, self.gpu_memory_fraction)		
        with self.graph.as_default():
            with tf.Session() as sess: 
                start=time()
                feed_dict={self.images_placeholder: images, self.phase_train_placeholder:False}
                embeddings=sess.run(self.embeddings, feed_dict=feed_dict)
        end=time()
        sys.stderr.write("note: run embeddings success \ncost time: %ds\n"%(end-start))
        return embeddings
        
    def config(self, image_size=160, margin=32, gpu_memory_fraction=1.0):
        self.image_size=image_size
        self.margin=margin
        self.gpu_memory_fraction=1.0
        
def create_or_rm(path):
    if not os.path.exists(path):
        os.makedirs(path)
    else:
        shutil.rmtree(path)
        os.makedirs(path)
    return path
    
def load_image_data(image_paths,image_size=160):
    nrof_samples = len(image_paths)
    #img_list = [None] * nrof_samples
    img_list=[]
    for i in range(nrof_samples):
        img = misc.imread(os.path.expanduser(image_paths[i]), mode='RGB')
        aligned = misc.imresize(img, (image_size, image_size), interp='bilinear')
        prewhitened = prewhiten(aligned)
        img_list.append(prewhitened) 
    images = np.stack(img_list)
    return images

def align_data_set(src, dist, margin=32, image_size=160,gpu_memory_fraction=1):
    if not exists(src):
        print("src path does not exist")
        exit(-1)
    create_or_rm(dist)
    #load dataset
    dataset=get_dataset(src, has_class_directories=False, gci=True)

    minsize = 20 # minimum size of face
    threshold = [ 0.6, 0.7, 0.7 ]  # three steps's threshold
    factor = 0.709 # scale factor
    print('load mtcnn model...')
    with tf.Graph().as_default():
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_memory_fraction)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
        with sess.as_default():
            pnet, rnet, onet = align.detect_face.create_mtcnn(sess, None)
    print('load mtcnn model ok')
    count=0
    for identity in dataset:
        if len(identity.image_paths)==0:
            print("skip one class %s"%identity.name)
            continue
        save_path_dir = os.path.join(dist, identity.name)
        create_or_rm(save_path_dir)
        
        if identity.subnames:
            for i, paths in enumerate(identity.image_paths):
                if len(paths)==0:
                    print("skip class %s for index %d"%(identity.name, i))
                    continue
                nrof_samples=len(paths)
                nrof_process=min(nrof_samples, 200)
                sub_dir=join(save_path_dir,identity.subnames[i])
                create_or_rm(sub_dir)
                for j in range(nrof_samples):
                    img = misc.imread(os.path.expanduser(paths[j]), mode='RGB')
                    img_size = np.asarray(img.shape)[0:2]
                    bounding_boxes, _ = align.detect_face.detect_face(img, minsize, pnet, rnet, onet, threshold, factor)
                    if(bounding_boxes.shape[0]==0):#if there is no face continue
                        continue
                    det = np.squeeze(bounding_boxes[0,0:4])
                    bb = np.zeros(4, dtype=np.int32)
                    bb[0] = np.maximum(det[0]-margin/2, 0)
                    bb[1] = np.maximum(det[1]-margin/2, 0)
                    bb[2] = np.minimum(det[2]+margin/2, img_size[1])
                    bb[3] = np.minimum(det[3]+margin/2, img_size[0])
                    cropped = img[bb[1]:bb[3],bb[0]:bb[2],:]
                    aligned = misc.imresize(cropped, (image_size, image_size), interp='bilinear')
                    filename=join(sub_dir,basename(paths[j]))
                    misc.imsave(filename, aligned)
            count+=1
            print("%d th identity have croped"%(count))
        else:
            nrof_samples=len(identity.image_paths)
            paths=identity.image_paths
            for j in range(nrof_samples):
                    img = misc.imread(os.path.expanduser(paths[j]), mode='RGB')
                    img_size = np.asarray(img.shape)[0:2]
                    bounding_boxes, _ = align.detect_face.detect_face(img, minsize, pnet, rnet, onet, threshold, factor)
                    if(bounding_boxes.shape[0]==0):#if there is no face continue
                        continue
                    det = np.squeeze(bounding_boxes[0,0:4])
                    bb = np.zeros(4, dtype=np.int32)
                    bb[0] = np.maximum(det[0]-margin/2, 0)
                    bb[1] = np.maximum(det[1]-margin/2, 0)
                    bb[2] = np.minimum(det[2]+margin/2, img_size[1])
                    bb[3] = np.minimum(det[3]+margin/2, img_size[0])
                    cropped = img[bb[1]:bb[3],bb[0]:bb[2],:]
                    aligned = misc.imresize(cropped, (image_size, image_size), interp='bilinear')
                    filename=join(save_path_dir,basename(paths[j]))
                    misc.imsave(filename, aligned)
            count+=1
            print("%d th identity have croped"%(count))
            



def load_and_align_data(image_paths, image_size, margin, gpu_memory_fraction):

    minsize = 20 # minimum size of face
    threshold = [ 0.6, 0.7, 0.7 ]  # three steps's threshold
    factor = 0.709 # scale factor
    #face_lib=os.path.expanduser("~/project/face_recognition/face_lib")
    print('read image and detect face and align')
    with tf.Graph().as_default():
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_memory_fraction)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
        with sess.as_default():
            pnet, rnet, onet = align.detect_face.create_mtcnn(sess, None)
  
    nrof_samples = len(image_paths)
    #img_list = [None] * nrof_samples
    img_list=[]
    for i in range(nrof_samples):
        img = misc.imread(os.path.expanduser(image_paths[i]), mode='RGB')
        img_size = np.asarray(img.shape)[0:2]
        bounding_boxes, _ = align.detect_face.detect_face(img, minsize, pnet, rnet, onet, threshold, factor)
        if(bounding_boxes.shape[0]==0):#if there is no face continue
            continue
        det = np.squeeze(bounding_boxes[0,0:4])
        bb = np.zeros(4, dtype=np.int32)
        bb[0] = np.maximum(det[0]-margin/2, 0)
        bb[1] = np.maximum(det[1]-margin/2, 0)
        bb[2] = np.minimum(det[2]+margin/2, img_size[1])
        bb[3] = np.minimum(det[3]+margin/2, img_size[0])
        cropped = img[bb[1]:bb[3],bb[0]:bb[2],:]
        aligned = misc.imresize(cropped, (image_size, image_size), interp='bilinear')
        prewhitened = prewhiten(aligned)
        img_list.append(prewhitened) 
    images = np.stack(img_list)
    return images


def savelist(path, list):
    with open(path,"w") as savefile:
        for item in list:
            savefile.write(item+"\n")


def load_graph(frozen_graph_filename):
    with tf.gfile.GFile(frozen_graph_filename, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    with tf.Graph().as_default() as graph:
        tf.import_graph_def(
            graph_def,
            input_map=None,
            return_elements=None,
            op_dict=None,
            producer_op_list=None
        )

    return graph

if __name__ == '__main__':
    

    os.environ["CUDA_VISIBLE_DEVICES"]='0'

    dataset_path=os.path.expanduser("~/project/NVR/data/face_cap_test")
    align_dist_path=os.path.expanduser("~/dataset/video-faces/face_cap_cluster_aug_align")
    #align_data_set(dataset_path,align_dist_path)


    #model_file=os.path.expanduser("~/models/facenet/20170512-110547.pb")

    #use CASIA-WebFace train
    model_file = os.path.expanduser("~/models/facenet/20180408-102900/20180408-102900.pb")

    # #use vggface2 train
    # #model_file = os.path.expanduser("~/models/facenet/20180402-114759/20180402-114759.pb")
    
    # #model_file = os.path.expanduser("/home/xuhao/models/facenet/fine-tune/20180501-200521/facenet.pb")

    # #use video faces raw trained 20 epochs
    # # model_file_raw_train=os.path.expanduser("~/models/facenet/raw_train/20180520-151319/freeze.pb")

    #use video faces trained 30 epochs
    model_file_raw_train=os.path.expanduser("/home/xuhao/models/facenet/raw_train/20180528-224637/freeze.pb")
    # #use video faces fine-tune 3 epochs
    # #model_file_fine_tune=os.path.expanduser("/home/xuhao/models/facenet/fine-tune/20180528-103416/freeze.pb")

    # #model_file = os.path.expanduser("~/models/facenet/raw_train/20180513-192632/freeze.pb")

    embd_module=FeatureEembedding(model_file_raw_train)

    dataset=get_dataset(align_dist_path, has_class_directories=True, gci=True)

    print("size of dataset is: %d"%len(dataset))
    store_path=os.path.expanduser("/home/xuhao/dataset/video-faces/face_cap_cluster_aug_align_raw_train_30_epochs")
    if not os.path.exists(store_path):
        os.makedirs(store_path)
        print("create store path %s"%store_path)
    else:
        shutil.rmtree(store_path)
        os.makedirs(store_path)
        
    embd_module.embed_dataset(dataset, store_path, align=False)



    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    