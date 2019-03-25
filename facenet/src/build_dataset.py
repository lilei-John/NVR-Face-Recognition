import os
import sys
from facenet.src.facenet import ImageClass

#use three txt to store the dataset
def store(path,dataset):
    if not os.path.exists(path):
        print("error: %s directory is not exists"%path)
        return

    for index, ins in enumerate(dataset):
        sub_dir=os.path.join(path,ins.name)
        if not os.path.exists(sub_dir):
            os.makedirs(sub_dir)
        indexfile = open(os.path.join(sub_dir, "index.txt"), "w")
        pathfile = open(os.path.join(sub_dir, "paths.txt"), "w")
        indexfile.write(str(ins.id_number))

        for p in ins.image_paths:
            pathfile.write(p+"\n")#last line is empty

def load(path):
    if not os.path.exists(path):
        print("error: %s directory is not exists" % path)
        return
    names=os.listdir(path)
    if len(names)==0:
        print("error: there is no instance in %s"%path)
        return
    dataset=[]
    for name in names:
        subdir=os.path.join(path, name)
        with open(os.path.join(subdir,"index.txt"),"r") as id_file:
            id_num=int(id_file.readline().strip())
        with open(os.path.join(subdir, "paths.txt"),"r") as path_file:
            image_paths=path_file.read().splitlines()
            #image_paths=paths[0:len(paths)-1]
        dataset.append(ImageClass(name, image_paths,id_num))
    return dataset

def main():
    #dataset_path=os.path.expanduser("~/dataset/aligned_images_DB_YTF")
    # dataset_path=os.path.expanduser("~/dataset/YTF_features")
    # dataset=facenet.get_fea_dataset(dataset_path)
    # print("class number is: %d"%len(dataset))
    store_path=os.path.expanduser("~/dataset/NAN/NAN_test")
    #store(store_path, dataset)
    re_dataset=load(store_path)
    print("length of test set is: %d"%len(re_dataset))
    print(len(re_dataset[0].image_paths))
    with open("/home/xuhao/dataset/NAN/NAN_test/Katalin_Kollat/paths.txt") as file:
        paths = file.read().splitlines()
        print(len(paths))
        #image_paths = paths[0:len(paths) - 1]

    print(re_dataset[0].name)

if __name__=="__main__":
    main()



















    