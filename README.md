# ConvNeXtV2_Install

We provide installation instructions for ImageNet classification experiments here.

## Dependency Setup
Create an new conda virtual environment
```
conda create -n convnextv2_test python=3.8 -y
conda activate convnextv2_test
```

Install CUDA
```
conda install cudatoolkit=11.1.1 -c conda-forge
```
Check cuDNN installed. If not, install cuDNN (older version than CUDA) <br>
![image](https://github.com/noyaboy/ConvNeXtV2_Install/assets/99811508/2760601b-d92a-45f3-b1cd-341f84e685d2)
```
conda list
conda search cudnn -c conda-forge
conda install cudnn==8.1.0.77 -c conda-forge
```
Check CUDA installed. If not, install cudatoolkit-dev. Related to $CUDA_HOME variable issue)
```
nvcc --version
conda search cudatoolkit-dev -c conda-forge
conda install cudatoolkit-dev=11.1.1 -c conda-forge
```
Install [Pytorch](https://pytorch.org/)>=1.8.0, [torchvision](https://pytorch.org/vision/stable/index.html)>=0.9.0 following official instructions. <br>
Go to https://pytorch.org/get-started/previous-versions/ and search for the required PyTorch version
```
conda install pytorch==1.8.0 torchvision==0.9.0 torchaudio==0.8.0 cudatoolkit=11.1 -c pytorch -c conda-forge
```
Go to https://blog.csdn.net/qq_38825788/article/details/125859041 to authorize the GitHub account.
```
ssh -T git@github.com
>> git@github.com: Permission denied (publickey)
```
Way to authorize the GitHub account
```
ssh-keygen -t rsa
```
Three enters <br>
then to ~/.ssh/id_rsa.pub copy content, login github, press setting, SSH and GPG keys, New SSH keys, random title, paste content to key, press Add key
ssh -T git@github.com
>> Hi <username>, You've successfully ...
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
