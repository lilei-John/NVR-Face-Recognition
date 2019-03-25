#!/usr/bin/env python
#encoding=utf8

from NAN_test import *
import sys
from os.path import expanduser, join, exists
from os import listdir
import numpy as np
import shutil


def calc_dist_mat(mat1, mat2):
    rows=mat2.shape[0]
    cols=mat1.shape[0]
    dist_mat=np.zeros((cols, rows))

    for i in range(cols):
        sub=mat2-mat1[i,:]
        #print(sub)
        dis1=np.sqrt(np.sum(np.square((mat2-mat1[i,:])),axis=1))
        #print(dis1)
        dist_mat[i,:]=dis1
    return dist_mat

def test_ytf():
    identity1 = int(sys.argv[1])
    identity2 = int(sys.argv[2])

    data_root = expanduser("~/dataset/YTF_features")
    names=os.listdir(data_root)
    identity1 = names[identity1]
    identity2 = names[identity2]

    id1_emb_path = join(data_root, identity1 + "/0.npy")
    id2_emb_path = join(data_root, identity2 + "/0.npy")

    if not exists(id1_emb_path):
        print("Error: id1 path does not exist")
        exit(1)
    if not exists(id2_emb_path):
        print("Error: id2 path does not exist")
        exit(1)

    id1_emb = np.load(id1_emb_path)
    id2_emb = np.load(id2_emb_path)
    id1_inner_dist = calc_dist_mat(id1_emb, id1_emb)
    id2_inner_dist = calc_dist_mat(id2_emb, id2_emb)

    cross_dist = calc_dist_mat(id1_emb, id2_emb)

    id1_inner_dist_avg = np.mean(id1_inner_dist)
    id2_inner_dist_avg = np.mean(id2_inner_dist)
    cross_dist_avg = np.mean(cross_dist)

    NAN_emb1=get_NAN_feature(to_unit_vec(id1_emb))
    NAN_emb2=get_NAN_feature(to_unit_vec(id2_emb))

    print("id1 inner dist: %f" % id1_inner_dist_avg)
    print("id2 inner dist: %f" % id2_inner_dist_avg)
    print("cross dist: %f" % cross_dist_avg)
    print("NAN dist: %f" % (cal_sim(NAN_emb1, NAN_emb2)))



def main():
    identity1=sys.argv[1]
    identity2=sys.argv[2]

    data_root=expanduser("~/dataset/video-faces/face_cap_features")
    #NAN_feature_root=expanduser("~/dataset/video-faces/NAN_features")

    id1_emb_path=join(data_root,identity1+".npy")
    id2_emb_path=join(data_root,identity2+".npy")

    if not exists(id1_emb_path):
        print("Error: id1 path does not exist")
        exit(1)
    if not exists(id2_emb_path):
        print("Error: id2 path does not exist")
        exit(1)

    id1_emb=np.load(id1_emb_path)
    id2_emb=np.load(id2_emb_path)
    id1_inner_dist=calc_dist_mat(id1_emb, id1_emb)
    id2_inner_dist=calc_dist_mat(id2_emb, id2_emb)

    cross_dist=calc_dist_mat(id1_emb, id2_emb)

    id1_inner_dist_avg=np.mean(id1_inner_dist)
    id2_inner_dist_avg=np.mean(id2_inner_dist)
    cross_dist_avg=np.mean(cross_dist)

    print("id1 inner dist: %f"%id1_inner_dist_avg)
    print("id2 inner dist: %f"%id2_inner_dist_avg)
    print("cross dist: %f"%cross_dist_avg)

    # id1_NAN_path = join(NAN_feature_root, identity1 + ".npy")
    # id2_NAN_path = join(NAN_feature_root, identity2 + ".npy")
    #
    # print("NAN dist: %f"%(cal_sim(np.load(id1_NAN_path),np.load(id2_NAN_path))))


if __name__=="__main__":
    # mat1=np.array([[1,2,3],[3,4,5]])
    # mat2=np.array([[1,2,3],[1,1,1],[2,2,2]])
    #
    # print(calc_dist_mat(mat1, mat2))
    #test_ytf()
    main()

