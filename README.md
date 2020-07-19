# Dense-Scale-Network-for-Crowd-Counting

An unofficial implement of [paper](https://arxiv.org/abs/1906.09707) "Dense Scale Network for Crowd Counting".

## Dataset setup

Download the shanghaitech dataset from [here](https://pan.baidu.com/s/1nuAYslz), UCF-QNRF dataset from [here](https://www.crcv.ucf.edu/data/ucf-qnrf/).

## Data preparation

In `make_sh_gt.py`, modify variable `root` in line 18 to your dataset path and set the `min_size` in line 16 for image. Then run the .py file. It will save images and .h5 file in `root/{dataset}_preprocessed/train/` and `root/{dataset}_preprocessed/test/`.

## Train

In `main.py`, set `train_path` to `root/{dataset}_preprocessed/train/` and `test_path` to `root/{dataset}_preprocessed/test/` in line 81 and 82. Also specify the `save_path`.
When training `shanghaitech PartA` dataset, the model shows faster convergence if learning rate  is set as 1e-4 compared to 1e-5 which is claimed by the paper.

## Test

### Test on one image

```(python)
python test_one_image.py --gpu 0 --model_path pretrained_model_path --test_img_path your_image_path
```

### Test on a dataset

```(python)
python test_dataset.py --gpu 0 --model_path pretrained_model_path --test_img_dir your_image_directory
```

## Result

Dataset|MAE|MSE
-|-|-
sha|89.76|142.23
shb|to be done|tbd
qrnf|tbd|tbd

Anyone interested in implementing crowd counting models is welcomed to contact me.
