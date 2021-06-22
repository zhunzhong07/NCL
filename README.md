# Neighborhood Contrastive Learning for Novel Class Discovery
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
<img alt="PyTorch" height="20" src="https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?&style=for-the-badge&logo=PyTorch&logoColor=white" />

This repository contains the official implementation of our paper:

**[Neighborhood Contrastive Learning for Novel Class Discovery, CVPR 2021](http://zhunzhong.site/project/ncl.html/)**
<br>
[Zhun Zhong](http://zhunzhong.site), Enrico Fini, Subhankar Roy, Zhiming Luo, Elisa Ricci, Nicu Sebe
<br>


## Requirements

PyTorch >= 1.1


## Data preparation

**We follow [AutoNovel](https://github.com/k-han/AutoNovel) to prepare the data**

By default, we save the dataset in `./data/datasets/` and trained models in `./data/experiments/`.

- For CIFAR-10 and  CIFAR-100, the datasets can be automatically downloaded by PyTorch.

- For ImageNet, we use the exact split files used in the experiments following existing work. To download the split files, run the command:
``
sh scripts/download_imagenet_splits.sh
``
. The ImageNet dataset folder is organized in the following way:

    ```
    ImageNet/imagenet_rand118 #downloaded by the above command
    ImageNet/images/train #standard ImageNet training split
    ImageNet/images/val #standard ImageNet validation split
    ```

## Pretrained models
We use the pretrained models (self-supervised learning and supervised learning) provided by [AutoNovel](https://github.com/k-han/AutoNovel). To download, run:
```
sh scripts/download_pretrained_models.sh
```
If you would like to train the self-supervised learning and supervised learning models by yourself, please refer to [AutoNovel](https://github.com/k-han/AutoNovel) for more details.

After downloading, you can go to perform our neighbor contrastive learning below.


## Neighborhood Contrastive Learning for Novel Class Discovery

### CIFAR10/CIFAR100


##### Without Hard Negative Generation (w/o HNG)

```shell
# Train on CIFAR10
CUDA_VISIBLE_DEVICES=0 sh scripts/ncl_cifar10.sh ./data/datasets/CIFAR/ ./data/experiments/ ./data/experiments/pretrained/supervised_learning/resnet_rotnet_cifar10.pth

# Train on CIFAR100
CUDA_VISIBLE_DEVICES=0 sh scripts/ncl_cifar100.sh ./data/datasets/CIFAR/ ./data/experiments/ ./data/experiments/pretrained/supervised_learning/resnet_rotnet_cifar100.pth

```

##### With Hard Negative Generation (w/ HNG)

```shell
# Train on CIFAR10
CUDA_VISIBLE_DEVICES=0 sh scripts/ncl_hng_cifar10.sh ./data/datasets/CIFAR/ ./data/experiments/ ./data/experiments/pretrained/supervised_learning/resnet_rotnet_cifar10.pth

# Train on CIFAR100
CUDA_VISIBLE_DEVICES=0 sh scripts/ncl_hng_cifar100.sh ./data/datasets/CIFAR/ ./data/experiments/ ./data/experiments/pretrained/supervised_learning/resnet_rotnet_cifar100.pth
```

**Note that, for cifar-10, we suggest to train the model w/o HNG, because the results of w HNG and w/o HNG for cifar-10 are similar. In addition, the model w/ HNG sometimes will collapse, but you can try different seeds to get the normal result.**

### ImageNet

##### Without Hard Negative Generation (w/o HNG)

```shell
# Subset A
CUDA_VISIBLE_DEVICES=0 python ncl_imagenet.py --unlabeled_subset A --model_name resnet_imagenet_ncl

# Subset B
CUDA_VISIBLE_DEVICES=0 python ncl_imagenet.py --unlabeled_subset B --model_name resnet_imagenet_ncl

# Subset C
CUDA_VISIBLE_DEVICES=0 python ncl_imagenet.py --unlabeled_subset C --model_name resnet_imagenet_ncl
```

##### With Hard Negative Generation (w/o HNG)

```shell
# Subset A
CUDA_VISIBLE_DEVICES=0 python ncl_imagenet.py --hard_negative_start 3 --unlabeled_subset A --model_name resnet_imagenet_ncl_hng

# Subset B
CUDA_VISIBLE_DEVICES=0 python ncl_imagenet.py --hard_negative_start 3 --unlabeled_subset B --model_name resnet_imagenet_ncl_hng

# Subset C
CUDA_VISIBLE_DEVICES=0 python ncl_imagenet.py --hard_negative_start 3 --unlabeled_subset C --model_name resnet_imagenet_ncl_hng
```

## Acknowledgement

Our code is heavily designed based on [AutoNovel](https://github.com/k-han/AutoNovel). If you use this code, please also acknowledge their paper.


## Citation
We hope you find our work useful. If you would like to acknowledge it in your project, please use the following citation:
```
@InProceedings{Zhong_2021_CVPR,
      author    = {Zhong, Zhun and Fini, Enrico and Roy, Subhankar and Luo, Zhiming and Ricci, Elisa and Sebe, Nicu},
      title     = {Neighborhood Contrastive Learning for Novel Class Discovery},
      booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
      month     = {June},
      year      = {2021},
      pages     = {10867-10875}
}
```

## Contact me

If you have any questions about this code, please do not hesitate to contact me.

[Zhun Zhong](https://zhunzhong.site)