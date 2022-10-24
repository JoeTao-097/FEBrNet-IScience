import os
from keras import backend as K

import numpy as np
import cv2
from skimage.transform import resize



def getallfilesofwalk(root):
    """
    使用listdir循环遍历文件夹中所有文件
    """
    if not os.path.isdir(root):
        print(root)
        return []

    dirlist = os.walk(root)
    allfiles = []
    for root, dirs, files in dirlist:
        for file in files:
            #            print(os.path.join(root, file))
            allfiles.append(os.path.join(root, file))

    return allfiles


def create_dir(dir_name):
    try:
        # Create target Directory
        os.makedirs(dir_name)
    except FileExistsError:
        print("Directory ", dir_name, " already exists")

def get_label(fmap, test_model, get_index=2):
    fc_layer = test_model.get_layer(index=get_index)
    iterate = K.function(inputs=[fc_layer.input], outputs=[fc_layer.output])
    fc_layer_output = iterate([fmap])[0]

    return fc_layer_output


def zero_pad(img, size=448):
    '''
    pad zeros to make a square img for resize
    '''
    h, w, c = img.shape
    if h > w:
        zeros = np.zeros([h, h - w, c]).astype(np.uint8)
        img_padded = np.hstack((img, zeros))
    elif h < w:
        zeros = np.zeros([w - h, w, c]).astype(np.uint8)
        img_padded = np.vstack((img, zeros))
    else:
        img_padded = img

    img_resized = (255*resize(img_padded, (size, size), anti_aliasing=True)).astype(np.uint8)

    return img_resized


def read_avi(fname):
    cap = cv2.VideoCapture(fname)

    wid = int(cap.get(3))
    hei = int(cap.get(4))
    framerate = int(cap.get(5))
    framenum = int(cap.get(7))

    video = np.zeros((framenum, hei, wid, 3), dtype=np.uint8)

    for i in range(framenum):
        a, b = cap.read()
        video[i] = b[..., ::-1]

    return video