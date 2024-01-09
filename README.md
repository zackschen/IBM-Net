# IBM - Code For: Towards Redundancy-Free Sub-networks in Continual Learning

## Dataset
Currently, we are using the following datasets for our experiments:

+ Sequential CIFAR-100
+ Sequential Tiny-ImageNet
+ Sequential Mini-ImageNet

After downloading the Datasets, please set the data path for the `DATA_PATH` variable in `dataset/utils/seq_cifar100`、`dataset/utils/seq_miniimagenet` and `dataset/utils/seq_tinyimagenet`.

## Setup
To execute the code for running experiments, please run the following command:

`pip install -r requirements.txt`

## Training
We provide several training examples within this repository for three datasets:

For IBM in CIFAR-100:

`CUDA_VISIBLE_DEVICES=0 bash ./config/CIFAR100/ib.sh`

For IBM in Tiny-ImageNet:

`CUDA_VISIBLE_DEVICES=0 bash ./config/TinyImageNet/ib.sh`

For IBM in Mini-ImageNet:

`CUDA_VISIBLE_DEVICES=0 bash ./config/CIFAR100/ib.sh`

### Hyper-parameters
The following are the key hyper-parameters:

+ vb_fre: This parameter determines the frequency of epochs for decomposing the hidden representation to calculate the compression ratio.
+ kl_fac: This is a balancing factor between the classification loss and our information bottleneck regularization.
+ svd: This parameter enables or disables the Feature Decomposing.

# Citation

`@article{chen2023towards,
  title={Towards Redundancy-Free Sub-networks in Continual Learning},
  author={Chen, Cheng and Song, Jingkuan and Gao, LianLi and Shen, Heng Tao},
  journal={arXiv preprint arXiv:2312.00840},
  year={2023}
}`