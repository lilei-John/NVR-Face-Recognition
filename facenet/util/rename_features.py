#encoding=utf8
import os
import shutil

src_dir_root=os.path.expanduser("~/dataset/YTF_features")

dist_dir_root=os.path.expanduser("~/dataset/NAN/YTF_features")

if not os.path.exists(dist_dir_root):
    os.makedirs(dist_dir_root)
    print("create the dist dir %s"%dist_dir_root)
    
cls_names=os.listdir(src_dir_root)


def getNewname(txtfile):
    with open(txtfile,"r") as f:
        line=f.readline().split("\n")[0]
        newname=line.split("/")[-2]
        
    return newname
    
count=0
skip=0
for cls in cls_names:
    dist_dir=os.path.join(dist_dir_root, cls)
    src_dir=os.path.join(src_dir_root, cls)
    if not os.path.exists(dist_dir):
        os.makedirs(dist_dir)
    sub_files=os.listdir(src_dir)
    for file in sub_files:
        if file.find(".txt")!=-1:
            old_name=file.split(".")[0]
            txt_file=os.path.join(src_dir, file)
            newname=getNewname(txt_file)
            #print(os.path.join(src_dir, old_name+".txt"), os.path.join(dist_dir, newname+".txt"))
            if os.path.exists(os.path.join(src_dir, old_name+".txt")):
                shutil.copyfile(os.path.join(src_dir, old_name+".txt"),os.path.join(dist_dir, newname+".txt"))
                shutil.copyfile(os.path.join(src_dir, old_name+".npy"),os.path.join(dist_dir, newname+".npy"))
                count+=1
                print("%d pairs have copied"%count)
            else:
                skip+=1
                print("file %s not exists, skip=%d"%(os.path.join(src_dir, old_name+".txt"), skip))
                