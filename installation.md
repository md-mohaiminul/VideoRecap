# Install using [Conda-Pack](https://conda.github.io/conda-pack/)

The easiest way to install Video ReCap is using [conda-pack](https://conda.github.io/conda-pack/) with our provided [videorecap.tar.gz](https://drive.google.com/file/d/1lHm_-niZGW5f9bIT81ZWtMLH2pttI_lq/view?usp=sharing) environment. Here are the steps:

1. Install [conda-pack](https://conda.github.io/conda-pack/) using the following command or following [official documentation](https://conda.github.io/conda-pack/).
```
pip install conda-pack
```
2. Download [videorecap.tar.gz](https://drive.google.com/file/d/1lHm_-niZGW5f9bIT81ZWtMLH2pttI_lq/view?usp=sharing) environment.

3. (a) Unpack the environment into a directory, activate the environment, and conda-unpack.
```
mkdir -p videorecap
tar -xzf videorecap.tar.gz -C videorecap
source videorecap/bin/activate
conda-unpack
```
Or, 3 (b) Copy the environment to your anaconda 'envs' directory, activate the environment, and conda-unpack.
```
mkdir -p videorecap
tar -xzf videorecap.tar.gz -C videorecap
mv videorecap ANACONDA_DIRECTORY/envs
conda activate videorecap
conda-unpack
```

# Manual Installation
1. Create conda environment:
```
conda create --name videorecap python=3.8
conda activate videorecap
```
2. Install [pytorch](https://pytorch.org) with the correct CUDA version.
3. Install other packages:
```
pip install -r requirements.txt
```
4. Install and setup nlg-eval:
```
pip install git+https://github.com/Maluuba/nlg-eval.git@master
nlg-eval --setup
```