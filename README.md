# Automatic Signature Verification using Deep Learning

The project aims to create a system that will distinguish between an authentic signature and a forged one

## Getting started

Run the following command in a linux terminal to clone the repository

```
git clone https://github.com/rmvanarse/ps1_project
```

### Dependencies

The system requires Python 3.6 and Pytorch installed.

The following libraries need to be installed:

numpy
cv2
PIL
time
os
matplotlib
argparse

##Usage Instructions

The user only needs to run the executable to use the system. To run the executable, run the following command on the terminal:

```
./model_predict.py --<anchor_path> -<original_signature_path> -<input_img_path> -<threshold>
```
