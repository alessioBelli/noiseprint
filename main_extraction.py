# This is the code to extract Noiseprint
#    python main_extraction.py input.png noiseprint.mat
#    python main_showout.py input.png noiseprint.mat
#
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#
# Copyright (c) 2019 Image Processing Research Group of University Federico II of Naples ('GRIP-UNINA').
# All rights reserved.
# This work should only be used for nonprofit purposes.
#
# By downloading and/or using any of these files, you implicitly agree to all the
# terms of the license, as specified in the document LICENSE.txt
# (included in this package) and online at
# http://www.grip.unina.it/download/LICENSE_OPEN.txt
#

from sys import argv
from time import time
from noiseprint.noiseprint import genNoiseprint
from noiseprint.utility.utilityRead import imread2f
from noiseprint.utility.utilityRead import jpeg_qtableinv
from noiseprint.utility.functions import cut_ctr, crosscorr_2d, pce, stats, gt, aligned_cc
import numpy as np
import glob
from PIL import Image
import os
from sklearn import metrics
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
'''
imgfilename = argv[1]
outfilename = argv[2]

timestamp = time()
img, mode = imread2f(imgfilename, channel=1)
try:
    QF = jpeg_qtableinv(strimgfilenameeam)
except:
    QF = 200
res = genNoiseprint(img,QF)
timeApproach = time() - timestamp

out_dict = dict()
out_dict['noiseprint'] = res
out_dict['QF'] = QF
out_dict['time'] = timeApproach

if outfilename[-4:] == '.mat':
    import scipy.io as sio
    sio.savemat(outfilename, out_dict)
else:
    import numpy as np
    np.savez(outfilename, **out_dict)
'''
#Our adaptation
ff_dirlist = np.array(sorted(glob.glob('data/train/*')))
ff_device = np.array([os.path.split(i)[1].rsplit('_', 1)[0] for i in ff_dirlist])
fingerprint_device = sorted(np.unique(ff_device))
nat_dirlist = np.array(sorted(glob.glob('data/test/*')))
nat_device = np.array([os.path.split(i)[1].rsplit('_', 1)[0] for i in nat_dirlist])

def compute_noiseprints(crop_size):
    print('Computing fingerprints...')
    # for each device, we extract the images belonging to that device and we compute the corresponding noiseprint, which is saved in the array k
    k = []
    for device in fingerprint_device:
        print('Computing fingerprint of device: ' + device + '...')
        noises = []
        for img_path in ff_dirlist[ff_device == device]:
            img, mode = imread2f(img_path, channel=1)
            img = cut_ctr(img, crop_size)
            try:
                QF = jpeg_qtableinv(strimgfilenameeam)
            except:
                QF = 200
            res = genNoiseprint(img,QF)
            noises.append(res)
        fingerprint = np.average(noises, axis=0)
        np.save("noiseprints/fingerprint_"+device+".npy", fingerprint)
        k+=[fingerprint]
    return k

def load_noiseprints():
    print("Loading noiseprints...")
    k = []
    noiseprints_dirlist = np.array(sorted(glob.glob('noiseprints/*')))
    for noise in noiseprints_dirlist:
        k+=[np.load(noise)]
    return k

def compute_residuals(crop_size):
    print('Computing residuals...')
    # for each device, we extract the images belonging to that device and we compute the corresponding noiseprint, which is saved in the array w
    w=[]
    for img_path in nat_dirlist:
        img, mode = imread2f(img_path, channel=1)
        img = cut_ctr(img, crop_size)
        try:
            QF = jpeg_qtableinv(strimgfilenameeam)
        except:
            QF = 200
        w+=[genNoiseprint(img,QF)]
    return w

def plot_device(fingerprint_device, natural_indices, values, label):
    # here we took 3 as our input
    n = 10 #number of test images for each device
    avgResult = []
    # calculates the average
    avgResult = np.average(np.asarray(values).reshape(-1, n), axis=1)
    avgResult = avgResult.tolist()
    plt.title('Noiseprint for ' + str(fingerprint_device))
    plt.xlabel('query images')
    plt.ylabel(label)

    plt.bar(np.unique(natural_indices), avgResult)
    # plt.axhline(y=60, color='r', linestyle='-')
    plt.xticks(np.unique(natural_indices), rotation=90)
    plt.tight_layout()
    plt.savefig('plots/'+ label + '/' +str(fingerprint_device)+'.png')

    plt.clf()
    
def plot_roc_curve(stats_cc, stats_pce):
    roc_curve_cc = metrics.RocCurveDisplay(fpr=stats_cc['fpr'], tpr=stats_cc['tpr'], roc_auc=stats_cc['auc'], estimator_name='ROC curve')
    roc_curve_pce = metrics.RocCurveDisplay(fpr=stats_pce['fpr'], tpr=stats_pce['tpr'], roc_auc=stats_pce['auc'], estimator_name='ROC curve')
    plt.style.use('seaborn')
    roc_curve_pce.plot(linestyle='--', color='blue')
    plt.savefig('plots/' +'roc_curve_pce.png')
    roc_curve_cc.plot(linestyle='--', color='blue')
    plt.savefig('plots/' +'roc_curve_cc.png')
    plt.clf()

def plot_confusion_matrix(cm, name):
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=fingerprint_device)
    fig, ax = plt.subplots(figsize=(10,10))
    disp.plot(cmap=plt.cm.Blues, xticks_rotation=90, ax=ax)
    plt.grid(False)
    plt.tight_layout()
    plt.savefig('plots/'+name, pad_inches=5)
    plt.clf()

def test(k, w):
    # Computing Ground Truth
    # gt function return a matrix where the number of rows is equal to the number of cameras used for computing the fingerprints, and number of columns equal to the number of natural images
    # True means that the image is taken with the camera of the specific row
    gt_ = gt(fingerprint_device, nat_device)

    print('Computing cross correlation')
    cc_aligned_rot = aligned_cc(k, w)['cc']

    print('Computing statistics cross correlation')
    stats_cc = stats(cc_aligned_rot, gt_)
    print('AUC on CC {:.2f}, expected {:.2f}'.format(stats_cc['auc'], 0.98))
    accuracy_cc = accuracy_score(gt_.argmax(0), cc_aligned_rot.argmax(0))
    print('Accuracy CC {:.2f}'.format(accuracy_cc))
    cm_cc = confusion_matrix(gt_.argmax(0), cc_aligned_rot.argmax(0))
    plot_confusion_matrix(cm_cc, "Confusion_matrix_CC.png")

    print('Computing PCE...')
    pce_rot = np.zeros((len(fingerprint_device), len(nat_device)))

    for fingerprint_idx, fingerprint_k in enumerate(k):
        pce_values = []   ###
        natural_indices = []    ###
        for natural_idx, natural_w in enumerate(w):
            cc2d = crosscorr_2d(fingerprint_k, natural_w)
            prnu_pce = pce(cc2d)['pce']   ###
            pce_values.append(prnu_pce)   ###
            pce_rot[fingerprint_idx, natural_idx] = pce(cc2d)['pce']
            ###
            natural_indices.append(nat_device[natural_idx][:-2])

        plot_device(fingerprint_device[fingerprint_idx], natural_indices, pce_values, "PCE")

    print('Computing statistics on PCE...')
    stats_pce = stats(pce_rot, gt_)
    print('AUC on PCE {:.2f}, expected {:.2f}'.format(stats_pce['auc'], 0.81))
    accuracy_pce = accuracy_score(gt_.argmax(0), pce_rot.argmax(0))
    print('Accuracy PCE {:.2f}'.format(accuracy_pce))
    cm_pce = confusion_matrix(gt_.argmax(0), pce_rot.argmax(0))
    plot_confusion_matrix(cm_pce, "Confusion_matrix_PCE.png")

    plot_roc_curve(stats_cc, stats_pce)

    print("Computing Euclidean Distance...")
    euclidean_rot = np.zeros((len(fingerprint_device), len(nat_device)))

    for fingerprint_idx, fingerprint_k in enumerate(k):
        dist_values = []
        natural_indices = []
        for natural_idx, natural_w in enumerate(w):
            dist = np.linalg.norm(fingerprint_k-natural_w)
            euclidean_rot[fingerprint_idx, natural_idx] = dist
            dist_values.append(dist)
            natural_indices.append(nat_device[natural_idx][:-2])

        plot_device(fingerprint_device[fingerprint_idx], natural_indices, dist_values, "EuclDist")

    accuracy_dist = accuracy_score(gt_.argmax(0), euclidean_rot.argmin(0))
    cm_dist = confusion_matrix(gt_.argmax(0), euclidean_rot.argmin(0))
    print('Accuracy with Euclidean Distance {:.2f}'.format(accuracy_dist))

    plot_confusion_matrix(cm_dist, "Confusion_matrix_Euclidean_Distance.png")



if __name__ == '__main__':
    crop_size = (128, 128)
    #k = compute_noiseprints(crop_size)
    k = load_noiseprints()
    w = compute_residuals(crop_size)
    test(k, w)