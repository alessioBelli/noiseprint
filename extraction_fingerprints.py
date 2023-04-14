#Compute the fingerprints of the datasets

from sys import argv
import os
from time import time
from numpy.fft import fft2, ifft2
import numpy as np
from glob import glob
from noiseprint.noiseprint import genNoiseprint
from noiseprint.utility.utilityRead import imread2f
from noiseprint.utility.utilityRead import jpeg_qtableinv

directory = "./data/"
fingerprints = "./fingerprints/"

def crosscorr_2d(k1: np.ndarray, k2: np.ndarray) -> np.ndarray:
    """
    PRNU 2D cross-correlation
    :param k1: 2D matrix of size (h1,w1)
    :param k2: 2D matrix of size (h2,w2)
    :return: 2D matrix of size (max(h1,h2),max(w1,w2))
    """
    assert (k1.ndim == 2)
    assert (k2.ndim == 2)

    max_height = max(k1.shape[0], k2.shape[0])
    max_width = max(k1.shape[1], k2.shape[1])

    k1 -= k1.flatten().mean()
    k2 -= k2.flatten().mean()

    k1 = np.pad(k1, [(0, max_height - k1.shape[0]), (0, max_width - k1.shape[1])], mode='constant', constant_values=0)
    k2 = np.pad(k2, [(0, max_height - k2.shape[0]), (0, max_width - k2.shape[1])], mode='constant', constant_values=0)

    k1_fft = fft2(k1, )
    k2_fft = fft2(np.rot90(k2, 2), )

    return np.real(ifft2(k1_fft * k2_fft)).astype(np.float32)


def pce(cc: np.ndarray, neigh_radius: int = 2) -> dict:
    """
    PCE position and value
    :param cc: as from crosscorr2d
    :param neigh_radius: radius around the peak to be ignored while computing floor energy
    :return: {'peak':(y,x), 'pce': peak to floor ratio, 'cc': cross-correlation value at peak position
    """
    assert (cc.ndim == 2)
    assert (isinstance(neigh_radius, int))

    out = dict()

    max_idx = np.argmax(cc.flatten())
    max_y, max_x = np.unravel_index(max_idx, cc.shape)

    peak_height = cc[max_y, max_x]

    cc_nopeaks = cc.copy()
    cc_nopeaks[max_y - neigh_radius:max_y + neigh_radius, max_x - neigh_radius:max_x + neigh_radius] = 0

    pce_energy = np.mean(cc_nopeaks.flatten() ** 2)

    out['peak'] = (max_y, max_x)
    out['pce'] = (peak_height ** 2) / pce_energy * np.sign(peak_height)
    out['cc'] = peak_height

    return out

def create_fingerprint():
    ff_dirlist = np.array(sorted(glob('data/*')))
    ff_device = np.array([os.path.split(i)[1].rsplit('_', 1)[0] for i in ff_dirlist])
    fingerprint_device = sorted(np.unique(ff_device))

    
    for device in fingerprint_device:
        print("I'm computing the "+device+" device")
        imgs = []
        a = []
        for img_path in ff_dirlist[ff_device == device]:
            img, mode = imread2f(img_path, channel=1)
            try:
                QF = jpeg_qtableinv(strimgfilenameeam)
            except:
                QF = 200
            res = genNoiseprint(img,QF)
            out_dict = res
            a.append(out_dict)
        a = np.array(a)
        temp = np.average(a, axis=0)
        np.save("./fingerprints_resized/"+device, temp)
    
    

def correlation():
    name_file = "test/test.jpg"
    img, mode = imread2f(name_file, channel=1)
    noise_test = genNoiseprint(img,200)

    fingerprints = os.listdir("./fingerprints_resized/")
    max_score = 0
    min_score = 100000000
    name = ""
    name_distance = ""
    for device in fingerprints:
        fingerprint = np.load('./fingerprints_resized/'+device)
        cc2d = crosscorr_2d(fingerprint, noise_test)
        
        print(device+" V.S. "+name_file+" :")
        pce_score = pce(cc2d)['pce'] 
        print(pce_score)
        dist = np.linalg.norm(fingerprint-noise_test)
        print(dist)
        if pce_score > max_score:
            max_score = pce_score
            name = device
        if dist < min_score:
            min_score = dist
            name_distance = device
        
    
    print("The photo is taken by "+name+" using the correlation")
    print("The photo is taken by "+name_distance+" using the euclidian distance")
        



if __name__ == "__main__":
    #create_fingerprint()
    correlation()