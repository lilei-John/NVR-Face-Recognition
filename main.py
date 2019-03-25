#!/usr/bin/env python
#encoding=utf8

import sys
from subprocess import *
from multiprocessing import Queue
import threading
import signal
import os
from face_recognition import *
import time
from datetime import datetime
import shutil
from os.path import join
from facenet.src.facenet import ImageClass
import numpy as np

global fin
global fout
global is_exit
is_exit=False

DATA_ROOT_PATH="./data/face_cap"
result_path="./data/result"
reslut_log_file="./data/result/result_log.log"
time_log_file_path="./data/result/time_log.log"
global res_log_file

res_log_file=open(reslut_log_file, "a+", buffering=1)
time_log_file=open(time_log_file_path,"a+",buffering=1)

#load time list
time_list=[]
lines=time_log_file.readlines()
for line in lines:
    line=line.split("\n")[0]
    id=line.split("\t")[0]
    record_time=line.split("\t")[1]
    time_list.append((id,record_time))
#print(time_list)

if not os.path.exists(result_path):
    os.makedirs(result_path)
os.environ["CUDA_VISIBLE_DEVICES"]='0'
detect_module_path='./video-face-capture/bin/nvr_face_capture'
if not os.path.exists(detect_module_path):
	print("error: %s not exists"%detect_module_path)
	exit(1)

log_file=open("./video-face-capture/logs/runtime.log","w")


proc = Popen(detect_module_path, bufsize=1024, stdin=PIPE, stdout=PIPE, stderr=log_file)
(fin, fout) = (proc.stdin, proc.stdout)

sys.stderr.write("open detect module subprocess ok\n")

recog_module=FaceRecognition()

def get_last_time(id, record_time):
    global time_list
    global time_log_file
    ret_time=None
    time_log_file.write("%s\t%s\n"%(id, record_time))
    if len(time_list)==0:
        time_list.append((id,record_time))
        return ret_time
    else:
        for item in time_list:
            if id==item[0]:
                ret_time=item[1]
        time_list.append((id,record_time))
        return ret_time
            
class Producer(threading.Thread):
    """
    @:param queue 阻塞队列
    @:param name 线程名字
    """
    def __init__(self,queue,name):
        threading.Thread.__init__(self)
        self.queue = queue
        self.name = name

    def run(self):
        while not is_exit:
            data = fout.readline()
            if len(data)>4 or len(data)==0:
            	continue
            sys.stderr.write("data from detect module")
            data=data.split("\n")[0]
            self.queue.put(data)
            sys.stderr.write("put data %s to queue \n"%str(data))
        

class Consumer(threading.Thread):
    def __init__(self,queue,name):
        threading.Thread.__init__(self)
        self.queue = queue
        self.name = name

    def run(self):
        while not is_exit:
            data = self.queue.get()
            sys.stderr.write("get data %s from queue\n"%data)
            self.queue.task_done()
            face_dir=os.path.join(DATA_ROOT_PATH, data)
            image_paths=[os.path.join(face_dir,name) for name in os.listdir(face_dir)]
            start=time.time()
            dist_list, emb, NAN_feature=recog_module.recog(image_paths)

            timestamp = datetime.strftime(datetime.now(), '%Y%m%d-%H%M%S')
            person_id=dist_list[0][0]
            dist=float(dist_list[0][1])
            known_image=getOneImage(join(FACE_IMAGE_DB_PATH,person_id))
            if dist<0.80 and len(image_paths)>8:
                subdir = join(join(FACE_IMAGE_DB_PATH, person_id), timestamp)

                #save new features
                np.save(join(join(FACE_DB_PATH, person_id), timestamp+".npy"), emb)

                # result_dir = join(result_path, timestamp)
                # if not os.path.exists(result_dir):
                #     os.makedirs(result_dir)
                # shutil.copy(getOneImage(join(FACE_IMAGE_DB_PATH,person_id)), result_dir)
                # new_image_path=join(result_dir, os.path.basename(image_paths[0]).split('.')[0]+"_new.png")
                # shutil.copy(image_paths[0], new_image_path)

                if not os.path.exists(subdir):
                    os.makedirs(subdir)
                for path in image_paths:
                    shutil.copy(path, subdir)
                res_log_file.write("[%s]: id_come: %s, id_min: %s, dist: %f, duration: %.2f\n"%(timestamp,data,person_id,\
                dist, time.time()-start-2))
                print("[%s]: id_come: %s, id_min: %s, dist: %f, duration: %.2f"%(timestamp,data,person_id, dist, \
                    time.time()-start-2))

                last_date=get_last_time(person_id,timestamp)
                if last_date:
                    res_log_file.write("[%s]: come last time: %s\n"%(timestamp,last_date))
                    print("[%s]: come last time: %s"%(timestamp,last_date)) 
                else:
                    res_log_file.write("[%s]: this person has no time record\n"%(timestamp))
                    print("[%s]: this person has no time record"%(timestamp))

            elif dist>=0.80 and len(image_paths)>20:
                save_id=recog_module.get_max_id()

                #update dataset ids features of recog_module
                recog_module.db_features=np.row_stack((recog_module.db_features, NAN_feature))
                recog_module.ids.append(len(recog_module.dataset))
                recog_module.dataset.append(ImageClass(save_id,None, int(save_id)))

                #save new person's images and embeddings
                save_dir=join(FACE_IMAGE_DB_PATH, save_id)
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)

                image_subdir=join(save_dir, timestamp)
                if not os.path.exists(image_subdir):
                    os.makedirs(image_subdir)
                    
                for path in image_paths:
                    shutil.copy(path, image_subdir)

                feature_save_dir=join(FACE_DB_PATH, save_id)
                if not os.path.exists(feature_save_dir):
                    os.makedirs(feature_save_dir)
                np.save(join(feature_save_dir, timestamp+".npy"), emb)

                res_log_file.write("[%s]: id_come: %s, id_min: %s, dist: %f, duration: %.2f\n"%(timestamp,data,person_id,\
                dist, time.time()-start-2))
                print("[%s]: id_come: %s, id_min: %s, dist: %f, duration: %.2f"%(timestamp,data,person_id, dist, \
                time.time()-start-2))
                res_log_file.write("[%s]: new person: %s registerd id: %s\n"%(timestamp,data,save_id))
                print("[%s]: new person: %s registerd id: %s"%(timestamp,data,save_id))

            
            result_dir = join(result_path, timestamp)
            if not os.path.exists(result_dir):
                os.makedirs(result_dir)
            shutil.copy(known_image, result_dir)
            new_image_path=join(result_dir, os.path.basename(image_paths[0]).split('.')[0]+"_new.png")
            shutil.copy(image_paths[0], new_image_path)
            

def quit(signum, frame):
    sys.stderr.write('You choose to stop me. \n') 
    is_exit=True
    res_log_file.close()
    os.kill(proc.pid,signal.SIGKILL) #kill detect process
    os.kill(os.getpid(),signal.SIGKILL)#kill main process  

def getOneImage(dir):
    subdir=join(dir,os.listdir(dir)[0])
    image_path=join(subdir, os.listdir(subdir)[0])
    return image_path

def main():
    queue = Queue(100)
    try:
        signal.signal(signal.SIGINT, quit)
        signal.signal(signal.SIGTERM, quit)

        consumer = Consumer(queue,'p')
        producer = Producer(queue,'c')
        producer.setDaemon(True)
        producer.start()
        consumer.setDaemon(True)
        consumer.start()
        while True:
            proc.wait()
            print("Note: detection module exit")
            exit(1)
            time.sleep(5)

    except Exception as exc:
        print(exc)   

if __name__ == '__main__':
    main()

