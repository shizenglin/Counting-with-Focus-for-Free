import math
import numpy as np
import scipy.io as sio
from skimage.transform import rescale, resize
import copy
from random import *
from PIL import Image

def SaveDmap(predicted_label, labeling_path):
    
    sio.savemat(labeling_path+'.mat', {'dmap':predicted_label})
    
    predicted_label=(predicted_label-np.min(predicted_label))/(np.max(predicted_label)-np.min(predicted_label))
    img=Image.fromarray(np.array(predicted_label*255.0).astype('uint8'))
    img.save(labeling_path+'.jpg')

def SaveMap(predicted_label, labeling_path):
    
    sio.savemat(labeling_path+'.mat', {'map':predicted_label})

def SavePmap(predicted_label, labeling_path):
    
    img=Image.fromarray(np.array(predicted_label*255).astype('uint8'))
    img.save(labeling_path+'.jpg')

def ReadImage(imPath,mirror = False,scale=1.0):
    """
    Read gray images.
    """ 
    imArr = np.array(Image.open(imPath))#.convert('L'))
    if(scale!=1):
        imArr = rescale(imArr, scale,preserve_range=True)
    if (len(imArr.shape)<3):
        imArr = imArr[:,:,np.newaxis]
        imArr = np.tile(imArr,(1,1,3))
        
    return imArr

def ResizeDmap(densitymap,scale = 1.0):
    
    b,w,h=densitymap.shape
    rescale_densitymap=np.zeros([b, int(w*scale), int(h*scale)]).astype('float32')
    for i in xrange(b):
        dmap_sum = densitymap[i,:,:].sum()
        rescale_densitymap[i,:,:] = rescale(densitymap[i,:,:], scale, preserve_range=True)
        res_sum = rescale_densitymap[i,:,:].sum()
        if res_sum != 0:
            rescale_densitymap[i,:,:]= rescale_densitymap[i,:,:] * (dmap_sum/res_sum)
    return rescale_densitymap

def ResizePmap(pmap,scale = 1.0):
    
    b,w,h=pmap.shape
    rescale_pmap=np.zeros([b, int(w*scale), int(h*scale)]).astype('float32')
    for i in xrange(b):
        rescale_pmap[i,:,:] = rescale(pmap[i,:,:], scale, preserve_range=True)#
    return rescale_pmap

def ReadMap(mapPath,name):
    """
    Load the density map from matfile.
    """ 
    map_data = sio.loadmat(mapPath)
    return map_data[name]
    
def load_data_pairs(img_path, dmap_path, pmap_path):

    img_data = ReadImage(img_path)
    dmap_data = ReadMap(dmap_path,'dmap')
    pmap_data = ReadMap(pmap_path,'pmap')
    
    img_data = img_data.astype('float32')
    dmap_data = dmap_data.astype('float32')
    pmap_data = pmap_data.astype('int32')
    
    dmap_data = dmap_data*100.0        
    img_data = img_data/255.0

    return img_data, dmap_data, pmap_data

def get_batch_patches(img_path, dmap_path, pmap_path, patch_dim, batch_size):
    rand_img, rand_dmap, rand_pmap = load_data_pairs(img_path, dmap_path, pmap_path)

    if np.random.random() > 0.5:
        rand_img=np.fliplr(rand_img)
        rand_dmap=np.fliplr(rand_dmap)    
        rand_pmap=np.fliplr(rand_pmap) 

    w, h, c = rand_img.shape
    
    patch_width = int(patch_dim[0])
    patch_heigh = int(patch_dim[1])
        
    batch_img = np.zeros([batch_size, patch_width, patch_heigh, c]).astype('float32')
    batch_dmap = np.zeros([batch_size, patch_width, patch_heigh]).astype('float32')
    batch_pmap = np.zeros([batch_size, patch_width, patch_heigh]).astype('int32')
    batch_num = np.zeros([batch_size]).astype('int32')

    rand_img = rand_img.astype('float32')
    rand_dmap = rand_dmap.astype('float32')
    rand_pmap = rand_pmap.astype('int32')

    for k in range(batch_size):
        # randomly select a box anchor        
        w_rand = randint(0, w - patch_width)
        h_rand = randint(0, h - patch_heigh)
        
        pos = np.array([w_rand, h_rand])
        # crop
        img_norm = copy.deepcopy(rand_img[pos[0]:pos[0]+patch_width, pos[1]:pos[1]+patch_heigh, :])
        dmap_temp = copy.deepcopy(rand_dmap[pos[0]:pos[0]+patch_width, pos[1]:pos[1]+patch_heigh])
        pmap_temp = copy.deepcopy(rand_pmap[pos[0]:pos[0]+patch_width, pos[1]:pos[1]+patch_heigh])

        batch_img[k, :, :, :] = img_norm
        batch_dmap[k, :, :] = dmap_temp
        batch_pmap[k, :, :] = pmap_temp
        # global density step siz, L which is estimated by equation 5 in the paper
        L = 8
        batch_num[k] = dmap_temp.sum()/L

    return batch_img, batch_dmap, batch_pmap, batch_num
