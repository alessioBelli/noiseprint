# Noiseprint: a CNN-based camera model fingerprint
[Noiseprint](https://ieeexplore.ieee.org/document/8713484) is a CNN-based camera model fingerprint
extracted by a fully Convolutional Neural Network (CNN).

## License :page_with_curl:
Copyright (c) 2019 Image Processing Research Group of University Federico II of Naples ('GRIP-UNINA').
Modified by Alberto Casagrande and Alessio Belli (University of Trento) in collaboration with the research group from the University of Bergen.

All rights reserved.

This software should be used, reproduced and modified only for informational and nonprofit purposes.

By downloading and/or using any of these files, you implicitly agree to all the
terms of the license, as specified in the document LICENSE.txt
(included in this package) 

## Installation :wrench:
The code requires Python 3.5 and Tensorflow 1.2.1 .
In order to install everything that is needed, create and activate a conda environment through the following command:
```
conda create -n noiseprint python=3.5 pip
conda activate noiseprint
```

### Installation with GPU
Install Cuda8 and Cudnn5, more informetion on sites:
- https://developer.nvidia.com/cuda-downloads
- https://developer.nvidia.com/cudnn

Then install the requested libraries using:
```
cat noiseprint/requirements-gpu.txt | xargs -n 1 -L 1 pip install
```

### Installation without GPU
Install the requested libraries using:
```
cat noiseprint/requirements-cpu.txt | xargs -n 1 -L 1 pip install
```

## Usage :key:
Firstly, the dataset has to be included in the `data` folder, divided in training (`train`) and testing (`test`) images.
The images belonging to a specific camera should have the name in the form *Brand_Model_ID_i.jpg* (example: *Apple_iPhone6_0_0.jpg*). 

To run the code:

```
python main_extraction.py -c dimention_squared_crop_size -n number_of_testing_images_per_camera
```

You can choose the crop size used for computing the noiseprints with the parameter `-c`. You must also state the number of test images for each device (just for visualization of the plots) by using the parameter `-n`.
The noiseprints of the cameras are saved in the `noiseprints` directory, while the charts showing the performance of the method are saved in the `plot` folder.

## How it works :gear:
### Training
The noiseprint characterizing each device is computed by making the average over the noiseprints of the images belonging to the specific device.

### Testing
We perform the pairwise comparison between the noiseprints of the test images and the cameras fingerprints using different methods:
- Cross correlation
- PCE (Peak to Correlation Energy)
- Euclidean Distance


According to the results, the latter is the most reliable method.

## Authors :man_technologist: :man_technologist:

**Alberto Casagrande**

**Alessio Belli**

## Reference

```js
@article{Cozzolino2019_Noiseprint,
  title={Noiseprint: A CNN-Based Camera Model Fingerprint},
  author={D. Cozzolino and L. Verdoliva},
  journal={IEEE Transactions on Information Forensics and Security},
  doi={10.1109/TIFS.2019.2916364},
  pages={144-159},
  year={2020},
  volume={15}
} 
```
The reference code can be found [here](https://github.com/grip-unina/noiseprint)
