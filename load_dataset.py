from skimage import data, transform
import numpy as np
import os
from tqdm import tqdm

CELEBA_IMGS_DIR = '/home/media/dataset/celeba/img_align_celeba/'

def load_celeba(nb_imgs, is_gray=False):
    bbox=(40, 218-30, 15, 178-15)
    if is_gray:
        img_chns = 1
    else:
        img_chns = 3
    imgs = np.empty((nb_imgs, 64, 64, img_chns), dtype='float32')

    for i in tqdm(range(nb_imgs), desc='loading images'):
        img = transform.resize(data.imread(os.path.join(CELEBA_IMGS_DIR, '{0:06d}.jpg'.format(i+1)), is_gray)[bbox[0]:bbox[1], bbox[2]:bbox[3]], (64, 64, img_chns))
        imgs[i, :, :, :] = img

    return imgs

