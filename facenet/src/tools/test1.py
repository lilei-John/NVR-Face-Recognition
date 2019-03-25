#!/usr/bin/env python
#encoding=utf8
from NAN_test import *
import sys
from os.path import expanduser, join, exists
from os import listdir
import numpy as np
import shutil
from test2 import *

anchor=sys.argv[1]
SUB_DIR=True
model_file=os.path.expanduser("~/models/NAN_model/20180502-204253_NAN_10000_model.pb")
NAN_module=NAN(model_file)

def load_features_index(db_path):
	dataset=facenet.get_fea_dataset(db_path)
	embds=[]
	ids=[]
	for i in range(len(dataset)):
		if len(dataset[i].image_paths)==0:
			print("skip class %s"%dataset[i].name)
			continue
		for path in dataset[i].image_paths:
			embds.append(np.load(path))

		ids+=[i]*len(dataset[i].image_paths)

	embds=np.stack(embds)
	return dataset, embds, ids


def calc_dist(emb1, emb2, NAN=True):
    if not NAN:
        return np.mean(calc_dist_mat(emb1, emb2))
    else:
        emb1=NAN_module.Aggregate(emb1)
        emb2=NAN_module.Aggregate(emb2)

        return cal_sim(emb1, emb2)

NAN_feature_path=expanduser("~/dataset/video-faces/face_cap_cluster_clean_features")
if not exists(NAN_feature_path):
    print("Error: %s does not exist"%NAN_feature_path)
    exit(1)

names=listdir(NAN_feature_path)
anchor_other=None

if SUB_DIR:
    anchor_sub_dir=join(NAN_feature_path, anchor)
    anchor_path=join(anchor_sub_dir, listdir(anchor_sub_dir)[0])
    if len(listdir(anchor_sub_dir))>1:
        anchor_other=join(anchor_sub_dir, listdir(anchor_sub_dir)[1])

else:
    anchor_path=join(NAN_feature_path, anchor+".npy")

if not exists(anchor_path):
    print("Error: %s does not exist"%anchor_path)
    exit(1)

anchor_fea=np.load(anchor_path)

if anchor_other:
    anchor_other_fea=np.load(anchor_other)

dist_dict={}

for name in names:
    id=name.split(".")[0]
    if anchor==id:
        if anchor_other:
            dist=calc_dist(anchor_fea, anchor_other_fea)
            dist_dict[id]=dist

        continue
    if SUB_DIR:
        sub_dir=join(NAN_feature_path, name)
        fea_path=join(sub_dir, listdir(sub_dir)[0])
        fea=np.load(fea_path)
    else:
        fea=np.load(join(NAN_feature_path, name))

    dist=calc_dist(anchor_fea, fea)
    dist_dict[id]=dist

sorted_dict=sorted(dist_dict.items(), key=lambda d: d[1])

result_path=expanduser("~/project/face_recognition/facenet/data")
result_path_store=join(result_path, anchor)
if not exists(result_path_store):
    os.makedirs(result_path_store)
else:
    shutil.rmtree(result_path_store)
    os.makedirs(result_path_store)

data_root=expanduser("~/dataset/video-faces/face_cap_cluster_clean")
if not exists(data_root):
    print("Error: %s does not exist"%data_root)
    exit(1)

for i,e in enumerate(sorted_dict[:10]):
    id=e[0]
    src_path=join(data_root,id)
    if SUB_DIR:
        src_sub_path=join(src_path,listdir(src_path)[0])
        src_file=join(src_sub_path, listdir(src_sub_path)[0])
    else:
        names=listdir(src_path)
        src_file=join(src_path,names[0])

    shutil.copyfile(src_file, join(result_path_store,str(i)+"_"+os.path.basename(src_file)))

print(sorted_dict[:10])



