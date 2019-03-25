import os
import sys
import copy

files=[]

def _findImages(path,filter=".*"):
    parents = os.listdir(path)
    if(len(parents) <= 0):
        return True
    else:
        for parent in parents:
            child = os.path.join(path,parent)
            if os.path.isdir(child):
                _findImages(child)
            else:
                if (child.find(".jpg")!=-1) or (child.find(".png")!=-1):
                    files.append(os.path.join(path,child))
                    print(os.path.join(path,child))

def getAllImages(path,filter=".*"):
    global files
    files=[]
    _findImages(path)
    print("Done: there have %d images in path %s "%(len(files),path))
    return copy.copy(files)
def getSubClass(path):
    if not os.path.exists(path):
        raise("error: %s directory is not exists"%path)
    subclass=[]
    retImages=[]
    for c in os.listdir(path):
        facedir=os.path.join(path, c)
        if os.path.isdir(facedir):
            subclass.append(c)
            faces=[]
            faces=[os.path.join(facedir,f) for f in os.listdir(facedir)]
            retImages.append(copy.copy(faces))
            
    return subclass, retImages
        
    
if __name__=="__main__":
#    getAllImages(os.path.expanduser("~/project/darknet/data"))
#    getAllImages(os.path.expanduser("~/project/darknet/data"))
    path=os.path.expanduser("~/dataset/aligned_images_DB_YTF/Pilar_Montenegro")
    subc, images = getSubClass(path)
    print("len of subclasses is: %d\nlen of subclass 0's images: %d "%(len(subc), len(images[0])))
    print(subc[0])
    

