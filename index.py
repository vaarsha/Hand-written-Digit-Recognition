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
    with open('dset.pickle','wb') as f:
        pickle.dump(d_set, f, -1)

    print("Created new file dset.pickle")

def load_file():

    file_init()

    with open('dset.pickle', 'rb') as f:
        d_set = pickle.load(f)

    for key in ('tr_img', 'ts_img'):
        d_set[key] = d_set[key].reshape(-1, 1, 28, 28)

    return d_set

def img_show(img):
    pl = Image.fromarray(np.uint8(img))
    pl.show()

print("Loading file")
ld_set = load_file()
print(ld_set['tr_label'][0])
img = ld_set['tr_img'][0]
img = img.reshape(28, 28)

img_show(img)
