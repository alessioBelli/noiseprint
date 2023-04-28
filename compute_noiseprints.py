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

import argparse
from noiseprint.noiseprint import genNoiseprint
from noiseprint.utility.utilityRead import imread2f
from noiseprint.utility.utilityRead import jpeg_qtableinv
from noiseprint.utility.functions import cut_ctr
import numpy as np
import glob
from random import randrange
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

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

def random_crop(img, crop_size):
    img = img.copy()
    for axis in range(img.ndim):
        axis_target_size = crop_size[axis]
        axis_original_size = img.shape[axis]
        if axis_target_size > axis_original_size:
            raise ValueError(
                'Can\'t have target size {} for axis {} with original size {}'.format(axis_target_size, axis,
                                                                                      axis_original_size))
        elif axis_target_size < axis_original_size:
            axis_start_idx = randrange(0,axis_original_size - crop_size[axis])
            axis_end_idx = axis_start_idx + crop_size[axis]
            img = np.take(img, np.arange(axis_start_idx, axis_end_idx), axis)

    '''
    plt.imshow(img, interpolation='nearest')
    plt.show()
    '''

    return img

def compute_noiseprints(crop_size):
    print('Computing fingerprints...')
    # for each device, we extract the images belonging to that device and we compute the corresponding noiseprint, which is saved in the array k
    for device in fingerprint_device:
        print('Computing fingerprint of device: ' + device + '...')
        noises = []
        n=0
        for img_path in ff_dirlist[ff_device == device]:
            img, mode = imread2f(img_path, channel=1)
            img = cut_ctr(img, crop_size)
            #img = random_crop(img, crop_size)
            try:
                QF = jpeg_qtableinv(strimgfilenameeam)
            except:
                QF = 200
            res = genNoiseprint(img,QF)
            noises.append(res)
            n=n+1
            if(n==80):
                break
        fingerprint = np.average(noises, axis=0)
        np.save("noiseprints/fingerprint_"+device+".npy", fingerprint)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Noiseprint extraction", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-c", "--crop_size", type=int, action="store", help="Specifies the crop size", required=True)
    parser.add_argument("-n", "--test_images", type=int, action="store", help="Specifies the # of test images for each device", required=True)
    args = parser.parse_args()


    crop_size = (args.crop_size, args.crop_size)
    n = args.test_images
    compute_noiseprints(crop_size)