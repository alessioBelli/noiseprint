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

def compute_noiseprints():
    print('Computing fingerprints...')
    # for each device, we extract the images belonging to that device and we compute the corresponding noiseprint, which is saved in the array k
    k = []
    for device in fingerprint_device:
        i=0
        noises = []
        for img_path in ff_dirlist[ff_device == device]:
            img, mode = imread2f(img_path, channel=1)
            img = cut_ctr(img, (512, 512))
            try:
                QF = jpeg_qtableinv(strimgfilenameeam)
            except:
                QF = 200
            res = genNoiseprint(img,QF)
            noises.append(res)
            np.save("noises/"+device+"_"+str(i), res)
            i=i+1
        fingerprint = np.average(noises, axis=0)
        np.save("noiseprints/fingerprint_"+device+".npy", fingerprint)
        k+=[fingerprint]
    return k
'''
def load_noiseprints():
    k = []
    noiseprints_dirlist = np.array(sorted(glob.glob('noises/*')))
    noise_device = np.array([os.path.split(i)[1].rsplit('_', 1)[0] for i in noiseprints_dirlist])
    for device in fingerprint_device:
        noises = []
        for noise in noiseprints_dirlist[noise_device == device]:
            noises.append(np.load(noise))
        fingerprint = np.average(noises, axis=0)
        np.save("noiseprints/fingerprint_"+device+".npy", fingerprint)
        k+=[fingerprint]'''
'''
# Compute the average noiseprint for each device and save the noiseprints in k
k = []
noiseprints_dirlist = np.array(sorted(glob.glob('noises/*')))
noise_device = np.array([os.path.split(i)[1].rsplit('_', 1)[0] for i in noiseprints_dirlist])
for device in fingerprint_device:
    noises = []
    for noise in noiseprints_dirlist[noise_device == device]:
        noises.append(np.load(noise))
    fingerprint = np.average(noises, axis=0)
    np.save("noiseprints/fingerprint_"+device+".npy", fingerprint)
    k+=[fingerprint]'''

def compute_residuals():
    print('Computing residuals...')
    # for each device, we extract the images belonging to that device and we compute the corresponding noiseprint, which is saved in the array w
    w=[]
    for img_path in nat_dirlist:
        img, mode = imread2f(img_path, channel=1)
        img = cut_ctr(img, (512, 512))
        try:
            QF = jpeg_qtableinv(strimgfilenameeam)
        except:
            QF = 200
        w+=[genNoiseprint(img,QF)]
    return w

def plot_device(fingerprint_device, natural_indices, pce_values):
    plt.title('Noiseprint for ' + str(fingerprint_device))
    plt.xlabel('query images')
    plt.ylabel('PCE')

    plt.bar(natural_indices, pce_values)
    plt.axhline(y=60, color='r', linestyle='-')
    plt.xticks(natural_indices)
    plt.savefig('plots/' +str(fingerprint_device)+'.png')

    plt.clf()
    
def plot_roc_curve(stats_cc, stats_pce):
    roc_curve_cc = metrics.RocCurveDisplay(fpr=stats_cc['fpr'], tpr=stats_cc['tpr'], roc_auc=stats_cc['auc'], estimator_name='ROC curve')
    roc_curve_pce = metrics.RocCurveDisplay(fpr=stats_pce['fpr'], tpr=stats_pce['tpr'], roc_auc=stats_pce['auc'], estimator_name='ROC curve')
    plt.style.use('seaborn')
    roc_curve_pce.plot(linestyle='--', color='blue')
    plt.savefig('plots/' +'roc_curve_pce.png')
    roc_curve_cc.plot(linestyle='--', color='blue')
    plt.savefig('plots/' +'roc_curve_cc.png')

def plot_confusion_matrix(cm):
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=fingerprint_device)
    disp.plot(cmap=plt.cm.Blues, xticks_rotation=45)
    plt.grid(False)
    plt.tight_layout()
    plt.savefig("cm.png", pad_inches=5)

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

    print('Computing PCE...')
    pce_rot = np.zeros((len(fingerprint_device), len(nat_device)))

    for fingerprint_idx, fingerprint_k in enumerate(k):
        pce_values = []   ###
        natural_indices = []    ###
        for natural_idx, natural_w in enumerate(w):
            cc2d = crosscorr_2d(fingerprint_k, natural_w)
            euclidean_distance = np.linalg.norm(fingerprint_k - natural_w)
            prnu_pce = pce(cc2d)['pce']   ###
            pce_values.append(prnu_pce)   ###
            pce_rot[fingerprint_idx, natural_idx] = pce(cc2d)['pce']
            ###
            natural_indices.append(natural_idx)

        plot_device(fingerprint_device[fingerprint_idx], natural_indices, pce_values)

    print('Computing statistics on PCE...')
    stats_pce = stats(pce_rot, gt_)
    print('AUC on PCE {:.2f}, expected {:.2f}'.format(stats_pce['auc'], 0.81))

    plot_roc_curve(stats_cc, stats_pce)

    print("Computing Euclidean Distance...")
    euclidian_rot = np.zeros((len(fingerprint_device), len(nat_device)))

    for fingerprint_idx, fingerprint_k in enumerate(k):
        for natural_idx, natural_w in enumerate(w):
            dist = np.linalg.norm(fingerprint_k-natural_w)
            euclidian_rot[fingerprint_idx, natural_idx] = dist

    accuracy = accuracy_score(gt_.argmax(0), euclidian_rot.argmin(0))
    cm = confusion_matrix(gt_.argmax(0), euclidian_rot.argmin(0))
    print('Accuracy with Euclidean Distance {:.2f}'.format(accuracy))

    plot_confusion_matrix(cm)



if __name__ == '__main__':
    k = compute_noiseprints()
    #k = load_noiseprints()
    w = compute_residuals()
    test(k, w)