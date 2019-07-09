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

- numpy
- cv2
- PIL
- time
- os
- matplotlib
- argparse

## Usage Instructions

The user only needs to run the executable to use the system. To run the executable, run the following command on the terminal:

```
./model_predict.py --<anchor_path> -<original_signature_path> -<input_img_path> -<threshold>
```

Arguments:
- ```
<anchor_path>
```
This is the path to the anchor image, i.e. the original signature of the person from the
database of the bank.
- ```
<original_signature_path>
```
This is the path to another known authentic signature of the same person.
The input signature will be compared to the anchor as well as this image and the two differences
obtained will be compared. This file is currently needed for reliability of the system and may not be needed for better models.
- ```
<input_img_path>
```
This is the path to the preprocessed input signature.
- ```
<threshold>
```
Used for comparison. The value of the threshold that was used in this
implementation was 1. Since this value gives a good accuracy, we recommend keeping this value as 1.

## Built with

* [Puthon 3.6]
* [Pytorch]
* [Google Colab]
* [Jupyter Notebook]

