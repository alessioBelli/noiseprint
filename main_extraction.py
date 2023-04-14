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
from noiseprint.utility.functions import cut_ctr, crosscorr_2d, pce, stats, gt
import numpy as np
import glob
from PIL import Image
import os
from sklearn import metrics
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

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

nat_dirlist = np.array(sorted(glob.glob('data/test/*')))
nat_device = np.array([os.path.split(i)[1].rsplit('_', 1)[0] for i in nat_dirlist])

print('Computing fingerprints...')
fingerprint_device = sorted(np.unique(ff_device))

# for each device, we extract the images belonging to that device and we compute the corresponding noiseprint, which is saved in the array k
for device in fingerprint_device:
    imgs = []
    i=0
    for img_path in ff_dirlist[ff_device == device]:
        img, mode = imread2f(img_path, channel=1)
        img = cut_ctr(img, (512, 512))
        try:
            QF = jpeg_qtableinv(strimgfilenameeam)
        except:
            QF = 200
        res = genNoiseprint(img,QF)
        np.save("noises/"+device+"_"+str(i), res)
        i=i+1

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
    k+=[fingerprint]


print('Computing residuals...')
w=[]
# for each device, we extract the images belonging to that device and we compute the corresponding noiseprint, which is saved in the array w
for img_path in nat_dirlist:
    img, mode = imread2f(img_path, channel=1)
    img = cut_ctr(img, (512, 512))
    try:
        QF = jpeg_qtableinv(strimgfilenameeam)
    except:
        QF = 200
    w+=[genNoiseprint(img,QF)]


# Computing Ground Truth
# gt function return a matrix where the number of rows is equal to the number of cameras used for computing the fingerprints, and number of columns equal to the number of natural images
# True means that the image is taken with the camera of the specific row
gt = gt(fingerprint_device, nat_device)

print('Computing PCE...')
pce_rot = np.zeros((len(fingerprint_device), len(nat_device)))

for fingerprint_idx, fingerprint_k in enumerate(k):
    tn, tp, fp, fn = 0, 0, 0, 0  ###
    pce_values = []   ###
    natural_indices = []    ###
    for natural_idx, natural_w in enumerate(w):
        cc2d = crosscorr_2d(fingerprint_k, natural_w)
        prnu_pce = pce(cc2d)['pce']   ###
        print(prnu_pce)
        pce_values.append(prnu_pce)   ###
        pce_rot[fingerprint_idx, natural_idx] = pce(cc2d)['pce']
        ###
        natural_indices.append(natural_idx)
        if fingerprint_device[fingerprint_idx] == nat_device[natural_idx]:
            if prnu_pce > 60.:
                tp += 1.
            else:
                fn += 1.
        else:
            if prnu_pce > 60.:
                fp += 1.
            else:
                tn += 1.
    tpr = tp / (tp + fn)
    fpr = fp / (fp + tn)

    plt.title('PRNU for ' + str(fingerprint_device[fingerprint_idx]))
    plt.xlabel('query images')
    plt.ylabel('PCE')

    plt.bar(natural_indices, pce_values)
    plt.text(0.85, 0.85, 'TPR: ' + str(round(tpr, 2)) + '\nFPR: '+ str(round(fpr, 2)),
        fontsize=10, color='k',
        ha='left', va='bottom',
        transform=plt.gca().transAxes)
    plt.axhline(y=60, color='r', linestyle='-')
    plt.xticks(natural_indices)
    plt.savefig('plots/' +str(fingerprint_device[fingerprint_idx])+'.png')

    plt.clf()

print('Computing statistics on PCE...')
stats_pce = stats(pce_rot, gt)

print('AUC on PCE {:.2f}, expected {:.2f}'.format(stats_pce['auc'], 0.81))
roc_curve_pce = metrics.RocCurveDisplay(fpr=stats_pce['fpr'], tpr=stats_pce['tpr'], roc_auc=stats_pce['auc'], estimator_name='ROC curve')
plt.style.use('seaborn')
roc_curve_pce.plot(linestyle='--', color='blue')
plt.savefig('plots/' +'roc_curve_pce.png')