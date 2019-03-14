# A-simple-image-classification

This article uses keras, which is a simple classification of six types of images of glaciers, rocks, cities, waters, farmlands, and forests (without the ROI box).

---

## Quick Start

1. Download data.
BaiduYun: https://pan.baidu.com/s/1d2wVJJDwYj7rEP6GGepxjQ

2. Change the image input path of the main function and run train.py for training.
3. Change the image input path of the main function, run test.py to see the effect and exit the space.

     Python3 train.py
     Python3 test.py

Python2 is also OK, I am writing based on python3. It is best to train with GPU, otherwise it is very slow. If the weight of the training is too great, it will not be uploaded. Please believe me if you want.

---

Epochs=55, batch_size=32, norm_size=256, GTX1080 (8G) training for about an hour.

---

# Author press

1. Some .py codes;

2. A dataset data folder with training and validation sets;

3. Result folder, my results show;

4. pic_for_test folder, provide the pictures used for testing, you can also find it yourself.
