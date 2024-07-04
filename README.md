# ConvNeXtV2_Install

We provide installation instructions for ImageNet classification experiments here.

## Dependency Setup
Create an new conda virtual environment
```
conda create -n convnextv2 python=3.8 -y
conda activate convnextv2
```

Install [Pytorch](https://pytorch.org/)>=1.8.0, [torchvision](https://pytorch.org/vision/stable/index.html)>=0.9.0 following official instructions. For example:
```
conda install cudatoolkit=11.1.1 -c conda-forge
conda install cudatoolkit-dev=11.1.1 -c conda-forge
conda install pytorch==1.8.0 torchvision==0.9.0 torchaudio==0.8.0 cudatoolkit=11.1 -c pytorch -c conda-forge
```
Check Permission denied(publickey), <br> 
refered to https://blog.csdn.net/qq_38825788/article/details/125859041, to authorize the GitHub account.
```
ssh -T git@github.com
```

Clone this repo and install required packages:
```
git clone https://github.com/facebookresearch/ConvNeXt-V2.git
pip install timm==0.3.2 tensorboardX six
pip install submitit
conda install openblas-devel -c anaconda -y
```

Install MinkowskiEngine:

*(Note: we have implemented a customized CUDA kernel for depth-wise convolutions, which the original MinkowskiEngine does not support.)*
```
cd ConvNeXt-V2
vim .gitmodules
branch = main
```
```
git submodule update --init --recursive
git submodule update --recursive --remote
cd MinkowskiEngine
python setup.py install --blas_include_dirs=${CONDA_PREFIX}/include --blas=openblas
```

Install apex
```
git clone https://github.com/NVIDIA/apex
```
comment the following code
```

```
```
vim apex/contrib/optimizers/distributed_fused_lamb.py
vim apex/transformer/tensor_parallel/layers.py
vim apex/transformer/tensor_parallel/utils.py
vim apex/transformer/tensor_parallel/mappings.py
```
```
cd apex
pip install -v --no-build-isolation --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
cd ..
```

## Dataset Preparation

Download the [ImageNet-1K](http://image-net.org/) classification dataset and structure the data as follows:
```
/path/to/imagenet-1k/
  train/
    class1/
      img1.jpeg
    class2/
      img2.jpeg
  val/
    class1/
      img3.jpeg
    class2/
      img4.jpeg
```

For pre-training on [ImageNet-22K](http://image-net.org/), download the dataset and structure the data as follows:
```
/path/to/imagenet-22k/
  class1/
    img1.jpeg
  class2/
    img2.jpeg
  class3/
    img3.jpeg
  class4/
    img4.jpeg
```
