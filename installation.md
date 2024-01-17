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