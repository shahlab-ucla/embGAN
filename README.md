# embGAN: seGAN network for label-free cell detection in 3D images
Architecture and training code based on [SegAN: Semantic Segmentation with Adversarial Learning](https://github.com/YuanXue1993/SegAN/).

Training data available from www.datadryad.org DOI: 10.5061/dryad.zcrjdfnkz

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

**Inference**

- Run with: CUDA_VISIBLE_DEVICES=X(your GPU id) python inference.py --model_path [path to weights] --data_path [path to data] --out_path [save path]
