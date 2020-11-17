# Skip Connections Matter
 
This repository contains the code for [Skip Connections Matter: On the Transferability of Adversarial Examples Generated with ResNets](https://openreview.net/forum?id=BJlRs34Fvr) (ICLR 2020 Spotlight).

## News

- 11/18/2020 - We released more codes and the dataset used in our paper(a subset with 5000 images from ImageNet), to help reproduce the reported results in our paper.
- 02/20/2020 - arXiv posted and repository released.


## Method

We propose the Skip Gradient Method (SGM) to generate adversarial examples using gradients more from the skip connections rather than the
residual modules. In particular, SGM utilizes a decay factor (gamma) to reduce gradients from the residual modules, 

<img src="https://github.com/csdongxian/security-of-skip-connections/blob/master/figs/formula_of_sgm.jpg" width="80%" height="80%">


## Requisite

This code is implemented in PyTorch, and we have tested the code under the following environment settings:

- python = 37.6
- torch = 1.7.0
- torchvision = 0.8.1
- advertorch = 0.2.2
- pretrainedmodels = 0.7.4

## Run the code

1. Download the dataset from [Google Drive](https://drive.google.com/file/d/1RqDUGs7olVGYqSV_sIlqZRRhB9Mw48vM/view?usp=sharing) or [Baidu Drive](https://pan.baidu.com/s/1AUbdL6W7pHubyftCUgi3wQ) (pw:55rk), and extract images to the path `./SubImageNet224/`
2. Generate adversarial examples and save them into path `./adv_images/`. For ResNet-152 as the source model,
    ```bash
    python attack_sgm.py --gamma 0.2 --output_dir adv_images --arch densenet201 --batch-size 40
    ```
    For DenseNet-201 as the source model
    ```bash
    python attack_sgm.py --gamma 0.5 --output_dir adv_images --arch resnet152 --batch-size 40
    ```
3. Evaluate the transerability of generated adversarial examples in `./adv_images/`. For VGG19 with batch norm as the target model
    ```bash
    python evaluate.py --input_dir adv_images --arch vgg19_bn
    ```

## Results

- Visualization
<img src="https://github.com/csdongxian/security-of-skip-connections/blob/master/figs/examples.jpg" width="100%" height="100%">


- Reproduced results

We run this code, and the attack success (1 - acc) against VGG19 is close to the repored in our paper:


| Method      | PGD    | MI     | SGM    |
| ResNet-152  | 45.80% | 66.70% | 81.04% |
| DenseNet-201| 57.82% | 75.38% | 82.58% |

## Implementation

For easier reproduction, we provide more detailed information here.

#### register backward hook for SGM

In fact, we manipulate gradients flowing through ReLU in [utils_sgm](https://github.com/csdongxian/security-of-skip-connections/blob/master/utils_sgm.py), since there is no ReLU in skip-connections: 

- For ResNet, there are "downsampling" modules in which skip-connections are replaced by a conv layer. We do not manipulate gradients of "downsampling" module;

- For DenseNet, we manipulate gradients in all dense block.


#### Pretrained models

All pretrained models in our paper can be found online:

- For VGG/ResNet/DenseNet/SENet, we use pretrained models in [pretrainedmodels](https://github.com/Cadene/pretrained-models.pytorch);

- For Inception models, we use pretrained models in [slim](https://github.com/tensorflow/models/tree/master/research/slim) of Tensorflow;

- For secured models (e.g. ), they are trained by [Ensemble Adversarial Training](https://arxiv.org/abs/1705.07204) [2], and pretrained results can be found in [adv_imagenet_models](https://github.com/tensorflow/models/tree/master/research/adv_imagenet_models).

## Citing this work
```
@inproceedings{wu2020skip,
    title={Skip connections matter: On the transferability of adversarial examples generated with resnets},
    author={Wu, Dongxian and Wang, Yisen and Xia, Shu-Tao and Bailey, James and Ma, Xingjun},
    booktitle={ICLR},
    year={2020}
}
```

## Reference

[1] Yinpeng Dong, Fangzhou Liao, Tianyu Pang, Hang Su, Jun Zhu, Xiaolin Hu, and Jianguo Li. Boosting
adversarial attacks with momentum. In CVPR, 2018.

[2] Florian Tramèr, Alexey Kurakin, Nicolas Papernot, Ian Goodfellow, Dan Boneh, Patrick McDaniel. Ensemble Adversarial Training: Attacks and Defenses. In ICLR, 2018.
