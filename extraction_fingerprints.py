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
from sklearn.metrics import roc_curve, auc, accuracy_score

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

def training():
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
    
def gt_function(l1: list or np.ndarray, l2: list or np.ndarray) -> np.ndarray:
    """
    Determine the Ground Truth matrix given the labels
    :param l1: fingerprints labels
    :param l2: residuals labels
    :return: groundtruth matrix
    """
    l1 = np.array(l1)
    l2 = np.array(l2)

    assert (l1.ndim == 1)
    assert (l2.ndim == 1)

    gt_arr = np.zeros((len(l1), len(l2)), np.bool)

    for l1idx, l1sample in enumerate(l1):
        gt_arr[l1idx, l2 == l1sample] = True
    
    return gt_arr

def stats(cc: np.ndarray, gt: np.ndarray, ) -> dict:
    """
    Compute statistics
    :param cc: cross-correlation or normalized cross-correlation matrix
    :param gt: boolean multidimensional array representing groundtruth
    :return: statistics dictionary
    """
    assert (cc.shape == gt.shape)
    assert (gt.dtype == np.bool)

    assert (cc.shape == gt.shape)
    assert (gt.dtype == np.bool)

    fpr, tpr, th = roc_curve(gt.flatten(), cc.flatten())
    auc_score = auc(fpr, tpr)

    # EER
    eer_idx = np.argmin((fpr - (1 - tpr)) ** 2, axis=0)
    eer = float(fpr[eer_idx])

    outdict = {
        'tpr': tpr,
        'fpr': fpr,
        'th': th,
        'auc': auc_score,
        'eer': eer,
    }

    return outdict

def correlation():
    list_test_images = os.listdir("./test/")
    for image in list_test_images:
        try:
            img, mode = imread2f("./test/"+image, channel=1)
            noise_test = genNoiseprint(img,200)
            fingerprints = os.listdir("./fingerprints_resized/")
            max_score = 0
            min_score = 100000000
            name = ""
            name_distance = ""
            for device in fingerprints:
                fingerprint = np.load('./fingerprints_resized/'+device)
                cc2d = crosscorr_2d(fingerprint, noise_test)     
                pce_score = pce(cc2d)['pce'] 
                dist = np.linalg.norm(fingerprint-noise_test)
                if pce_score > max_score:
                    max_score = pce_score
                    name = device
                if dist < min_score:
                    min_score = dist
                    name_distance = device
                
            print(dist)
            print(image+" ---> "+name+" using the correlation")
            print(image+" ---> "+name_distance+" using the euclidian distance")

            name = ""
            name_distance = ""
            max_score = 0
            min_score = 100000000
        except:
            print("")

def testing():
    
    ff_dirlist = np.array(sorted(glob('fingerprints_resized/*')))
    ff_device = np.array([os.path.split(i)[1].rsplit('_', 1)[0] for i in ff_dirlist])
    fingerprint_device = sorted(np.unique(ff_device))
    k = []
    for fingerprint in ff_dirlist:
        k.append(np.load(fingerprint))

    nat_dirlist = np.array(sorted(glob('./test/*.jpg')))
    nat_device = np.array([os.path.split(i)[1].rsplit('_', 1)[0] for i in nat_dirlist])

    w = []
    for img_path in nat_dirlist:
        img, mode = imread2f(img_path, channel=1)
        try:
            QF = jpeg_qtableinv(strimgfilenameeam)
        except:
            QF = 200
        w+=[genNoiseprint(img,QF)]
    
    gt = gt_function(fingerprint_device, nat_device)
    euclidian_rot = np.zeros((len(fingerprint_device), len(nat_device)))

    for fingerprint_idx, fingerprint_k in enumerate(k):
        for natural_idx, natural_w in enumerate(w):
            dist = np.linalg.norm(fingerprint_k-natural_w)
            euclidian_rot[fingerprint_idx, natural_idx] = dist

    print(euclidian_rot)
    accuracy = accuracy_score(gt.argmax(0), euclidian_rot.argmin(0))
    print(accuracy)

    '''
    print('Computing PCE...')
    pce_rot = np.zeros((len(fingerprint_device), len(nat_device)))

    for fingerprint_idx, fingerprint_k in enumerate(k):
        natural_indices = []
        for natural_idx, natural_w in enumerate(w):
            dist = np.linalg.norm(fingerprint_k-natural_w)
            cc2d = crosscorr_2d(fingerprint_k, natural_w)
            prnu_pce = pce(cc2d)['pce']
            pce_rot[fingerprint_idx, natural_idx] = dist
            natural_indices.append(natural_idx)

    print(gt)
    print(pce_rot)
    print('Computing statistics on PCE...')
    stats_pce = stats(pce_rot, gt)

    print('AUC on PCE {:.2f}'.format(stats_pce['auc']))
    '''
    


if __name__ == "__main__":
    #training()
    #correlation()
    testing()