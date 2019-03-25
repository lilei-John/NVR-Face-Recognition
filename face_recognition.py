#!/usr/bin/env python
#encoding=utf8
from __future__ import print_function
import sys
sys.path.append("facenet/src") 
from os.path import *
import os
import numpy as np
from facenet.src.NAN_test import *
import facenet.src.facenet as facenet
from facenet.src.feature_embedding import *
from datetime import datetime

#FACE_DB_PATH=expanduser("~/dataset/video-faces/face_cap_cluster_clean_align_features")#embed with fine-tune

FACE_DB_PATH=expanduser("./data/dataset/video-face-lib/face_cap_cluster_clean_align_features")#embed with pretrain
FACE_IMAGE_DB_PATH=expanduser("./data/dataset/video-face-lib/face_cap_cluster_clean_align")

class FaceRecognition(object):
	"""docstring for FaceRecognation"""
	def __init__(self, DB_PATH=FACE_DB_PATH, add_person=False):
		super(FaceRecognition, self).__init__()
		#self.db_features,self.names = load_db_features(DB_PATH)
		self.mode="NAN"
		self.embd_module, self.NAN_module=self.init()#
		if not add_person:
			self.dataset,self.db_features, self.ids = self.load_features_index(DB_PATH, self.mode)
		
		
	def recog(self,image_paths,topN=1,ret_id=False):
		emb=self.embd_module.embed(image_paths,align=False)
		NAN_fea=self.NAN_module.Aggregate(emb, self.mode)

		sub=self.db_features-NAN_fea
        #print(sub)
		dist=np.sqrt(np.sum(np.square(sub),axis=1))
		sort_dist=np.argsort(dist)
		ret_list=[]
		if ret_id:
			return sort_dist[:topN], dist

		for i in sort_dist[:topN]:

			ret_list.append(((self.dataset[self.ids[i]].name),dist[i]))
			
		return ret_list, emb, NAN_fea
	def init(self):
		#modelfile=os.path.expanduser("~/models/facenet/20170512-110547.pb")
		#modelfile="/home/xuhao/models/facenet/fine-tune/20180501-200521/facenet.pb"

		#fine-tune
		model_file=os.path.expanduser("./models/NAN/facenet/fine-tune/20180528-103416/freeze.pb")
		#use CASIA-WebFace train
		#model_file = os.path.expanduser("~/models/facenet/20180408-102900/20180408-102900.pb")
		
		#raw_train
		#model_file=os.path.expanduser("/home/xuhao/models/facenet/raw_train/20180528-224637/freeze.pb")
		#modelfile=os.path.expanduser("~/models/facenet/20180402-114759/20180402-114759.pb") #vggface2
		#modelfile = os.path.expanduser("~/models/facenet/20180408-102900/20180408-102900.pb")#celebface
		embd_module=FeatureEembedding(model_file)

		#load_NAN_module
		#raw_train_nan_model
		#NAN_model_file=os.path.expanduser("~/models/NAN_model_with_raw_train_model/20180529-094539_NAN_5000_model.pb")

		#NAN_model_file=os.path.expanduser("~/models/NAN_model_with_pretrain_model/20180528-142801_NAN_5000_model.pb")
		NAN_model_file=os.path.expanduser("./models/NAN/NAN_model_video_faces/20180528-112547_NAN_5000_model.pb")
		NAN_module=NAN(NAN_model_file)

		return embd_module, NAN_module

	def load_features_index(self, db_path, mode="NAN"):
		dataset=facenet.get_fea_dataset(db_path)[0:300]
		embds=[]
		ids=[]
		for i in range(len(dataset)):
			if len(dataset[i].image_paths)==0:
				sys.stderr.write("skip class %s\n"%dataset[i].name)
				continue
			for path in dataset[i].image_paths:
				emb=np.load(path)
				embds.append(self.NAN_module.Aggregate(emb,mode))

			ids+=[i]*len(dataset[i].image_paths)

		embds=np.stack(embds)
		return dataset, embds, ids
	def get_max_id(self):
		names=os.listdir(FACE_IMAGE_DB_PATH)
		max_id=-1
		for name in names:
			if int(name)>max_id:
				max_id=int(name)
		return str(max_id+1)

	def add_one_person(self,image_paths):
		new_id=self.get_max_id()
		new_emb_path=os.path.join(FACE_DB_PATH,new_id)
		new_image_path=os.path.join(FACE_IMAGE_DB_PATH,new_id)
		create_or_rm(new_emb_path)
		create_or_rm(new_image_path)
		timestamp = datetime.strftime(datetime.now(), '%Y%m%d-%H%M%S')
		embds=self.embd_module.embed(image_paths,align=False)
		np.save(os.path.join(new_emb_path,timestamp),embds)
		image_save_path=create_or_rm(os.path.join(new_image_path,timestamp))
		for image in image_paths:
			shutil.copy(image,image_save_path)
				


def load_db_features(db_path):
	if not exists(db_path):
		print("Error: %s does not exist"%db_path)
		exit(1)
	facial_features=[]
	names=os.listdir(db_path)
	for name in names:
		fea_file=join(db_path, name)
		facial_features.append(np.load(fea_file))
	facial_features=np.stack(facial_features)
	return facial_features,names



if __name__=="__main__":
	import shutil
	os.environ["CUDA_VISIBLE_DEVICES"]='0'
	
	# face_dir=sys.argv[1]
	# #face_dir=os.path.expanduser("~/project/NVR/consoleDemo/data/face_cap/1")
	# image_paths=[os.path.join(face_dir,name) for name in os.listdir(face_dir)]
	# ids,_,_=recog_module.recog(image_paths,10)
	# print(ids)

	#add many
	# ytf_align_path=os.path.expanduser("/home/xuhao/dataset/ytf_images_align_with_mtcnn")
	# persons=os.listdir(ytf_align_path)[600:1000]
	# count=0
	# sum=len(persons)
	# for person in persons:
	# 	person_dir=os.path.join(ytf_align_path,person)
	# 	face_dir=os.path.join(person_dir,os.listdir(person_dir)[0])
	# 	image_paths=[os.path.join(face_dir,name) for name in os.listdir(face_dir)]
	# 	recog_module.add_one_person(image_paths[:min(100,len(image_paths))])
	# 	count+=1
	# 	print("add %d/%d person"%(count,sum))

	if len(sys.argv)>1:
		recog_module=FaceRecognition(add_person=True)
		for dir in sys.argv[1:]:
			face_dir=dir
			image_paths=[os.path.join(face_dir,name) for name in os.listdir(face_dir)]
			recog_module.add_one_person(image_paths)
		print("add persons success")
		exit(0)

	'''
	top N test
	'''
	recog_module=FaceRecognition()
	N=1
	test_path_root=os.path.expanduser("~/project/NVR/data/face_test_align")
	test_result_root=os.path.expanduser("~/project/NVR/data/top_%d_result"%N)
	if not os.path.exists(test_result_root):
		os.makedirs(test_result_root)

	anchors=[os.path.join(test_path_root, path) for path in os.listdir(test_path_root)]
	count=0
	sum=len(anchors)
	for anchor in anchors:
		result_path=create_or_rm(os.path.join(test_result_root, os.path.basename(anchor)))
		image_paths=[os.path.join(anchor,im) for im in os.listdir(anchor)]
		ids,dist=recog_module.recog(image_paths,N,ret_id=True)
		shutil.copy(image_paths[0],os.path.join(result_path,"anchor.png"))

		for i,id in enumerate(ids):
			image_subdir=os.path.join(FACE_IMAGE_DB_PATH,recog_module.dataset[recog_module.ids[id]].name)
			sub_dir0=os.path.join(image_subdir,os.listdir(image_subdir)[0])
			image0=os.path.join(sub_dir0,os.listdir(sub_dir0)[0])

			shutil.copy(image0, os.path.join(result_path, str(i)+"_%.3f_"%dist[id]+".png"))
		count+=1
		print("%d/%d has computed"%(count,sum))