# embGAN: seGAN network for label-free cell detection in nematode embryos

Architecture and training code based on [SegAN: Semantic Segmentation with Adversarial Learning](https://github.com/YuanXue1993/SegAN/).

**Dependencies**

- python 3.10
- [Pytorch 1.13.1](http://pytorch.org/)

**Data**

- Download the dataset for []() to the data folder.
- If you want to use your own dataset to train, modify the dataloaders/path in LoadDataTiff.py.

**Training**
- Run with: CUDA_VISIBLE_DEVICES=X(your GPU id) python train.py --cuda.
	Default output folder is ./outputs. 
- Training includes validation, we report validation results every epoch, validation images will be saved in the outputs folder.