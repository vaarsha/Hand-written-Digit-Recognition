import numpy as np
import gzip
import os
import pickle
from PIL import Image

def conversion(fl):
    val = 8
    img_size = 784

    if "images" in fl:
        val = 16

    with gzip.open(fl, 'rb') as f:
        data = np.frombuffer(f.read(), np.uint8, offset=val)

    if "images" in fl:
        data = data.reshape(-1, img_size)

    return data

def file_init():
    d_set = {}
    d_set['tr_img'] = conversion('train-images-idx3-ubyte.gz')
    d_set['tr_label'] = conversion('train-labels-idx1-ubyte.gz')
    d_set['ts_img'] = conversion('t10k-images-idx3-ubyte.gz')
    d_set['ts_label'] = conversion('t10k-labels-idx1-ubyte.gz')

    print("Converted to numpy")
    return d_set

def load_file():

    d_set = file_init()

    for key in ('tr_img', 'ts_img'):
        d_set[key] = d_set[key].reshape(-1, 1, 28, 28)

    return d_set

def img_show(img):
    pl = Image.fromarray(np.uint8(img))
    pl.show()

print("Loading file")
ld_set = file_init()

"""
print(ld_set['tr_label'][0])
img = ld_set['tr_img'][0]
img = img.reshape(28, 28)
img_show(img)
"""

np.savetxt("trlabel.csv", ld_set["tr_label"], delimiter=",")
np.savetxt("tslabel.csv", ld_set["ts_label"], delimiter=",")
ld_set["tr_img"].tofile("trimg.csv", sep=",")
ld_set["ts_img"].tofile("tsimg.csv", sep=",")
print("Created csv files")
