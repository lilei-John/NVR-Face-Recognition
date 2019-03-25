#!/usr/bin/env python
#encoding=utf8
import tensorflow as tf
import facenet.src.facenet as facenet
import os
import build_dataset
import numpy as np
import random
from feature_embedding import *
import sys
from sklearn import metrics
from scipy.optimize import brentq
from scipy import interpolate
import lfw
import shutil
import cv2
from NAN_train import MAX_ITER_STEPS
MAX_IAMGESET_SIZE=200
nrof_output=89
graph_file = os.path.expanduser("~/models/NAN_model/NAN_models_%d.ckpt.meta"%MAX_ITER_STEPS)
model_file = os.path.expanduser("~/models/NAN_model/NAN_models_%d.ckpt"%MAX_ITER_STEPS)

def get_batch(test_data, batch_index, FEATURE=False):

    nrof_examples = len(test_data)

    '''build the one-hot output label'''
    label_one_hot = np.zeros((1, len(test_data)))
    label_one_hot[0, batch_index] = 1

    ''''   
    load features
    random sample from medias
    for each class random sample a subclass which means a media
    read and embed them
    '''
    media = random.sample(test_data[batch_index].image_paths, 1)

    imageset = random.sample(media[0], min(len(media[0]), MAX_IAMGESET_SIZE))

    emb = embd_module.embed(imageset)
    emb = emb / np.sqrt(
        np.sum(np.square(emb), axis=1).reshape(((emb.shape[0]), 1)))  # normalized to unit vector

    return emb, label_one_hot, imageset



def load_NAN_model():
    # NAN module
    saver = tf.train.import_meta_graph(graph_file)
    return model_file, saver

def cal_sim(fea1, fea2):
    return np.sqrt(np.sum(np.square(np.subtract(fea1, fea2))))
def to_unit_vec(vec):
    vec = vec / np.sqrt(np.sum(np.square(vec),
                                     axis=1).reshape(((vec.shape[0]), 1)))
    return vec

def get_dataset_average_features(embeddings_path, store_path):
    if not os.path.exists(embeddings_path):
        print("ERROR: %s does not exists"%embeddings_path)
        exit(-1)
    if not os.path.exists(store_path):
        os.makedirs(store_path)
    names = os.listdir(embeddings_path)
    # run average embeddings
    count = 0
    for i in range(len(names)):
        embd=np.load(os.path.join(embeddings_path,names[i]))
        np.save(os.path.join(store_path, names[i]),np.mean(embd, axis=0))
        count += 1
        print("%dth features have aggregated" % (count))
    #embeddings.append(np.mean(first, axis=0))


class NAN():
    def __init__(self, modelfile):
        modelfile_exp=os.path.expanduser(modelfile)
        if not os.path.exists(modelfile_exp):
            sys.stderr.write("error: no such file: %s \n"%modelfile_exp)
            exit(1)

        sys.stderr.write("note: NAN module initializing...\n")
        self.graph=tf.Graph()
        with self.graph.as_default():
            with tf.Session() as sess:
                start=time()
                facenet.load_model(modelfile_exp)
                self.input_place_holder=tf.get_default_graph().get_tensor_by_name("x_input:0")
                self.score = tf.get_default_graph().get_tensor_by_name("score:0")
                self.NAN_feature = tf.get_default_graph().get_tensor_by_name("NAN_feature:0")
                self.phase_train_placeholder=tf.get_default_graph().get_tensor_by_name("output:0")
                shape=self.phase_train_placeholder.shape.as_list()
                global nrof_output
                nrof_output=shape[1]
                end=time()
                sys.stderr.write("note: NAN model loaded success \ncost time: %ds\n"%(end-start))

        sys.stderr.write("note: NAN module initialized success\n")
        self.train_feed=np.zeros((1, nrof_output))
    def GetScore(self, embed):
        with self.graph.as_default():
            with tf.Session() as sess:
                with tf.device("/cpu:1"):
                    feed_dict = {self.input_place_holder: embed, self.phase_train_placeholder: self.train_feed}
                    NAN_score = sess.run(self.score, feed_dict=feed_dict)
        return NAN_score

    def Aggregate(self, embed, mode="NAN"):
        if mode=="NAN":
            with tf.device('/cpu:12'):
                with self.graph.as_default():
                    with tf.Session() as sess:
                        with tf.device("/cpu:1"):
                            feed_dict = {self.input_place_holder: embed, self.phase_train_placeholder: self.train_feed}
                            NAN_feature = sess.run(self.NAN_feature, feed_dict=feed_dict)
        elif mode=="AVG":
            NAN_feature=np.mean(embed, axis=0)
        elif mode=="MAX":
            NAN_feature=np.max(embed, axis=0)
        elif mode=="MAX_SCORE":
            scores=self.GetScore(embed)
            max_idx=np.argmax(scores)
            NAN_feature=embed[max_idx,:]
        else:
            NAN_feature=embed[0,:]

        return NAN_feature
    def AggreageDataset(self, dataset, dist_path):
        return


def get_dataset_NAN_features(embeddings_path, store_path):
    if not os.path.exists(embeddings_path):
        print("ERROR: %s does not exists"%embeddings_path)
        exit(-1)
    if not os.path.exists(store_path):
        os.makedirs(store_path)

    names=os.listdir(embeddings_path)
    # run NAN embeddings
    NAN_embeddings = []
    model_file, saver = load_NAN_model()
    with tf.Session() as sess:
        saver.restore(sess, model_file)
        input_place_holder = tf.get_default_graph().get_tensor_by_name("x_input:0")
        output_place_holder = tf.get_default_graph().get_tensor_by_name("y_output:0")
        # score_tensor=tf.get_default_graph().get_tensor_by_name("score:0")
        feature_tensor = tf.get_default_graph().get_tensor_by_name("NAN_feature:0")
        y_output = np.zeros((1, nrof_output))
        count = 0
        for i in range(len(names)):
                emb=np.load(os.path.join(embeddings_path,names[i]))
                feed_dict = {input_place_holder: emb, output_place_holder: y_output}
                NAN_emb=sess.run(feature_tensor, feed_dict=feed_dict)
                NAN_embeddings.append(NAN_emb)
                np.save(os.path.join(store_path,names[i]),NAN_emb)
                count += 1
                print("%dth features have aggregated" % (count))

    NAN_embeddings = np.stack(NAN_embeddings)
    np.save("NAN_embeddings.npy", NAN_embeddings)
        

    return NAN_embeddings,names
    
def score_test(NAN_model_file, embed_model, image_paths, result_path="../NAN_result",topN=5):
    #NAN
    NAN_module=NAN(NAN_model_file)

    #embedding module
    embd_module = FeatureEembedding(embed_model)
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    else:
        shutil.rmtree(result_path)
        os.makedirs(result_path)

    embds=embd_module.embed(image_paths,align=False)
    scores=NAN_module.GetScore(embds)

    scores=np.transpose(scores)
    indexs=np.argsort(-scores)
    indexs=indexs[0]
    print(indexs[:5])
    print(np.sum(scores))
    font=cv2.FONT_HERSHEY_SIMPLEX
    for i in indexs[:topN]:
        path=image_paths[i]
        score=scores[0,i]
        im=cv2.imread(path)
        im = cv2.putText(im, '{:.4f}'.format(score), (0, 18), font, 0.75, (0, 97, 255), 2)
        save_path = os.path.join(result_path, os.path.basename(path))
        cv2.imwrite(save_path, im)


def get_better_dataset(NAN_model, embed_model, dataset, dist, topN=5):
    if not os.path.exists(NAN_model) or not os.path.exists(embed_model):
        print("error: model paths does not exist")
        return

    NAN_module = NAN(NAN_model)
    embd_module=FeatureEembedding(embed_model)
    for c in dataset:
        save_class_dir=os.path.join(dist, c.name)
        if not os.path.exists(save_class_dir):
            os.makedirs(save_class_dir)
        for i, paths in enumerate(c.image_paths):
            if len(paths)<5:
                continue
            save_dir = os.path.join(save_class_dir, c.subnames[i])
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)

            embds = embd_module.embed(paths, align=False)
            scores = NAN_module.GetScore(embds)
            scores = np.transpose(scores)
            indexs = np.argsort(-scores)
            indexs = indexs[0]
            print(indexs[:topN])

            # font = cv2.FONT_HERSHEY_SIMPLEX
            for i in indexs[:topN]:
                path = paths[i]
                score = scores[0, i]
                im = cv2.imread(path)
                save_path = os.path.join(save_dir, os.path.basename(path))
                cv2.imwrite(save_path, im)


def get_better_imageset(model_file, src, dist, topN=5):
    # NAN
    NAN_module = NAN(model_file)

    # embedding module
    modelfile = os.path.expanduser("~/models/facenet/20170512-110547.pb")
    modelfile = os.path.expanduser("/home/xuhao/models/facenet/fine-tune/20180501-200521/facenet.pb")

    embd_module = FeatureEembedding(modelfile)

    names=os.listdir(src)
    sub_dirs=[os.path.join(src,name) for name in names]
    dist_dirs=[os.path.join(dist, name) for name in names]

    for i,dir in enumerate(sub_dirs):
        if len(os.listdir(dir))<10:
            continue
        else:
            image_paths=[os.path.join(dir,img) for img in os.listdir(dir)]

        if not os.path.exists(dist_dirs[i]):
            os.makedirs(dist_dirs[i])
        else:
            shutil.rmtree(dist_dirs[i])
            os.makedirs(dist_dirs[i])
        result_path=dist_dirs[i]

        embds = embd_module.embed(image_paths, align=False)
        scores = NAN_module.GetScore(embds)
        scores = np.transpose(scores)
        indexs = np.argsort(-scores)
        indexs = indexs[0]
        print(indexs[:5])

        #font = cv2.FONT_HERSHEY_SIMPLEX
        for i in indexs[:topN]:
            path = image_paths[i]
            score = scores[0, i]
            im = cv2.imread(path)
            #im = cv2.putText(im, '{:.4f}'.format(score), (0, 18), font, 0.75, (255, 255, 0), 1)
            save_path = os.path.join(result_path, os.path.basename(path))
            cv2.imwrite(save_path, im)



def topN_identify_test():
    # graph_file = os.path.expanduser("~/models/NAN_model/NAN_models_30000.ckpt.meta")
    # model_file = os.path.expanduser("~/models/NAN_model/NAN_models_30000.ckpt")
    # saver = tf.train.import_meta_graph(graph_file)
    model_file, saver = load_NAN_model()
    #load test person with feature
    test_set_store_path = os.path.expanduser("~/dataset/NAN/NAN_test")
    if not os.path.exists(test_set_store_path):
        print("error: %s directory is not exists"%test_set_store_path)
        exit(-1)
    test_set=build_dataset.load(test_set_store_path)
    if len(test_set)>0:
        print("load test dataset success")
    else:
        print("error in load dataset")
        exit(-1)

    with tf.Session() as sess:
        saver.restore(sess, model_file)
        input_place_holder = tf.get_default_graph().get_tensor_by_name("x_input:0")
        output_place_holder = tf.get_default_graph().get_tensor_by_name("y_output:0")
        #score_tensor=tf.get_default_graph().get_tensor_by_name("score:0")
        feature_tensor=tf.get_default_graph().get_tensor_by_name("NAN_feature:0")
        y_output=np.zeros((1,nrof_output))

        dist_matrix=np.zeros((len(test_set),len(test_set)))

        for i in range(len(test_set)):
            anchor=np.load(test_set[i].image_paths[0])
            anchor_other=np.load(test_set[i].image_paths[1])
            anchor = anchor / np.sqrt(np.sum(np.square(anchor),
                                             axis=1).reshape(((anchor.shape[0]), 1)))
            anchor_other = anchor_other / np.sqrt(np.sum(np.square(anchor_other),
                                                         axis=1).reshape(((anchor_other.shape[0]), 1)))

            feed_dict={input_place_holder: anchor, output_place_holder: y_output}
            fea1=sess.run(feature_tensor,feed_dict=feed_dict)

            for j in range(len(test_set)):
                if i==j:
                    feed_dict = {input_place_holder: anchor_other, output_place_holder: y_output}
                    fea2 = sess.run(feature_tensor, feed_dict=feed_dict)
                    dist_matrix[i, j] = cal_sim(fea1, fea2)
                else:
                    other=np.load(random.sample(test_set[j].image_paths,1)[0])
                    other = other / np.sqrt(np.sum(np.square(other), axis=1).reshape(((other.shape[0]), 1)))

                    feed_dict = {input_place_holder: other, output_place_holder: y_output}
                    fea2=sess.run(feature_tensor, feed_dict=feed_dict)
                    dist_matrix[i, j] = cal_sim(fea1, fea2)
            print("compute %d anchor"%(i+1))
        np.save("dist_matrix.npy", dist_matrix)

def topNaccuracy(dist_matrix, N):
    sort_indxes = np.argsort(dist_matrix, axis=1)
    count = 0
    for i in range(matrix.shape[0]):
        for j in range(N):
            if sort_indxes[i, j] == i:
                count += 1
                break
    return float(count)/float(dist_matrix.shape[0])

def load_veri_pairs(pair_path, feature_root="~/dataset/ytf_features_128_align"):
    fea_path_root=os.path.expanduser(feature_root)
    pairs = []
    with open(pair_path, 'r') as f:
        for line in f.readlines()[1:]:
            pair = line.strip().split(',')
            pair[2]=os.path.join(fea_path_root,pair[2].strip()+".npy")
            pair[3]=os.path.join(fea_path_root,pair[3].strip()+".npy")
            pair[4]=int(pair[4])
            pairs.append(pair)
    pairs = np.array(pairs)[:,2:]
    pairs_path = pairs[:,0:2]
    issame = pairs[:,2:]
    issame = issame.transpose()
    # print(pairs_path.shape)
    # print(issame[0])
    # exit(1)
    return pairs_path, issame

global debug
debug=True

def get_avg_embeddings_issame():
    if os.path.exists("./data/embeddings_avg.npy") and os.path.exists("./data/issame_avg.npy") and debug==False:
        embeddings=np.load("./data/embeddings_avg.npy")
        actual_issame=np.load("./data/issame_avg.npy")
        actual_issame=actual_issame.tolist()
    else:
        val_pairs_path = "../data/splits.txt"
        if not os.path.exists(val_pairs_path):
            print("error: validate pairs path is not existed, %s is not existed" % val_pairs_path)
            exit(1)
        pairs_path, issame = load_veri_pairs(val_pairs_path)
        count=0
        # run average embeddings
        embeddings = []
        actual_issame=[]
        for i in range(pairs_path.shape[0]):
            if os.path.exists(pairs_path[i, 0]) and os.path.exists(pairs_path[i, 1]):
                first = to_unit_vec(np.load(pairs_path[i, 0]))
                second = to_unit_vec(np.load(pairs_path[i, 1]))
                # embeddings.append(np.max(first,axis=0))
                # embeddings.append(np.max(second,axis=0))
                embeddings.append(first[0,:])
                embeddings.append(second[0,:])
                count += 1
                actual_issame.append(int(issame[0, i]))
                print("%dth pair's features have computed" % (count))
            else:
                print("the pair is not existed")
                continue

        embeddings = np.stack(embeddings)
    # print((actual_issame))
    np.save("./data/embeddings_avg.npy", embeddings)
    np.save("./data/issame_avg.npy", np.array(actual_issame))

    return embeddings, actual_issame

def get_embeddings_issame(mode=NAN):
    if os.path.exists("./data/embeddings.npy") and os.path.exists("./data/issame.npy") and debug==False:
        embeddings=np.load("./data/embeddings.npy")
        actual_issame=np.load("./data/issame.npy")
        actual_issame=actual_issame.tolist()
    else:
        val_pairs_path = "../data/splits.txt"
        if not os.path.exists(val_pairs_path):
            print("error: validate pairs path is not existed, %s is not existed" % val_pairs_path)
            exit(1)
        pairs_path, issame = load_veri_pairs(val_pairs_path)

        # run NAN embeddings
        embeddings = []
        model_file, saver = load_NAN_model()
        with tf.Session() as sess:
            saver.restore(sess, model_file)
            input_place_holder = tf.get_default_graph().get_tensor_by_name("x_input:0")
            output_place_holder = tf.get_default_graph().get_tensor_by_name("y_output:0")
            score_tensor=tf.get_default_graph().get_tensor_by_name("score:0")
            feature_tensor = tf.get_default_graph().get_tensor_by_name("NAN_feature:0")
            global nrof_output
            nrof_output=output_place_holder.shape.as_list()[1]

            y_output = np.zeros((1, nrof_output))
            count = 0
            actual_issame = []
            for i in range(pairs_path.shape[0]):
                if os.path.exists(pairs_path[i, 0]) and os.path.exists(pairs_path[i, 1]):
                    first = to_unit_vec(np.load(pairs_path[i, 0]))
                    second = to_unit_vec(np.load(pairs_path[i, 1]))
                    if mode=="NAN":
                        feed_dict = {input_place_holder: first, output_place_holder: y_output}
                        embeddings.append(sess.run(feature_tensor, feed_dict=feed_dict))
                        feed_dict = {input_place_holder: second, output_place_holder: y_output}
                        embeddings.append(sess.run(feature_tensor, feed_dict=feed_dict))
                    else:
                        feed_dict = {input_place_holder: first, output_place_holder: y_output}
                        scores=(sess.run(score_tensor, feed_dict=feed_dict))
                        embeddings.append(first[np.argmax(scores),:])
                        feed_dict = {input_place_holder: second, output_place_holder: y_output}
                        scores=(sess.run(score_tensor, feed_dict=feed_dict))
                        embeddings.append(second[np.argmax(scores),:])

                    count += 1
                    actual_issame.append(int(issame[0, i]))
                    print("%dth pair's features have computed" % (count))
                else:
                    print("the pair is not existed")
                    continue

        embeddings = np.stack(embeddings)
        # print((actual_issame))
        np.save("./data/embeddings.npy", embeddings)
        np.save("./data/issame.npy", np.array(actual_issame))
    #print(actual_issame)
    return embeddings, actual_issame

def validate_on_ytf():
    mode="AVG"
    #embeddings, actual_issame=get_embeddings_issame(mode)
    embeddings, actual_issame=get_avg_embeddings_issame()

    #calculate the accuracy
    lfw_nrof_folds=10
    tpr, fpr, accuracy, val, val_std, far = lfw.evaluate(embeddings,
                                                         actual_issame, nrof_folds=lfw_nrof_folds)
    print('Accuracy: %1.3f+-%1.3f' % (np.mean(accuracy), np.std(accuracy)))
    print('Validation rate: %2.5f+-%2.5f @ FAR=%2.5f' % (val, val_std, far))

    np.save("./data/fpr_%s.npy"%mode, fpr)
    np.save("./data/tpr_%s.npy"%mode, tpr)
    #print(tpr)

    auc = metrics.auc(fpr, tpr)
    print('Area Under Curve (AUC): %1.3f' % auc)
    # eer = brentq(lambda x: 1. - x - interpolate.interp1d(fpr, tpr)(x), 0., 1.)
    # print('Equal Error Rate (EER): %1.3f' % eer)


if __name__=="__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    #topN_identify_test()
    # matrix=np.load("dist_matrix.npy")
    # print("number of identites: %d"%matrix.shape[0])
    # sort_indxes=np.argsort(matrix,axis=1)
    # N=1
    # print("top %d accuracy is: %.4f"%(N, topNaccuracy(matrix, N)))

    #validate_on_ytf()

    # # embds_path=os.path.expanduser("~/dataset/video-faces/face_cap_features")
    # # store_path=os.path.expanduser("~/dataset/video-faces/NAN_features")
    # # #get_dataset_average_features(embds_path, store_path)
    # # get_dataset_NAN_features(embds_path,store_path)
    #


    '''
    NAN model
    '''
    #model_file=os.path.expanduser("~/models/NAN_model/20180527-153228_NAN_20000_model.pb")
    NAN_model_file=os.path.expanduser("~/models/NAN_model_video_faces/20180528-112547_NAN_5000_model.pb")

    '''
    embedding model
    '''
    #modelfile = os.path.expanduser("~/models/facenet/20170512-110547.pb")
    #modelfile = os.path.expanduser("/home/xuhao/models/facenet/fine-tune/20180501-200521/facenet.pb")
    embed_modelfile=os.path.expanduser("/home/xuhao/models/facenet/fine-tune/20180528-103416/freeze.pb")

    # embd_model = os.path.expanduser("~/models/facenet/raw_train/20180513-192632/freeze.pb")
    #
    # NAN_module=NAN(model_file)
    # start=time()
    # aggregate_fea=NAN_module.Aggregate(np.load("/home/xuhao/dataset/video-faces/features/1.npy"))
    # print("time is: %f"%(time()-start))
    # #print(aggregate_fea)

    # # src=os.path.expanduser("~/dataset/video-faces/face_cap_clear_features")
    # # dis=os.path.expanduser("~/dataset/video-faces/face_cap_clear_avg_features")
    #
    # # get_dataset_average_features(src, dis)
    # #
    # # src=os.path.expanduser("~/dataset/video-faces/face_cap")
    # # dist=os.path.expanduser("~/dataset/video-faces/face_cap_clear_new")
    # # get_better_dataset(model_file, src, dist)

    #score test
    images_dir=sys.argv[1]
    image_paths=[os.path.join(images_dir,path) for path in os.listdir(images_dir)]
    score_test(NAN_model_file, embed_modelfile, image_paths,topN=len(image_paths))


    # dataset_path=os.path.expanduser("~/dataset/video-faces/face_cap_cluster")
    # dataset_clean_path=os.path.expanduser("~/dataset/video-faces/face_cap_cluster_clean")
    # if not os.path.exists(dataset_clean_path):
    #     os.makedirs(dataset_clean_path)
    # else:
    #     shutil.rmtree(dataset_clean_path)
    #     os.makedirs(dataset_clean_path)

    # dataset=facenet.get_dataset(dataset_path, has_class_directories=True, gci=True)

    # get_better_dataset(model_file, embd_model, dataset, dataset_clean_path)


