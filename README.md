# Operator Flow Matching (NeurIPS 2025 Spotlight)
This repository contains the code for "Stochastic Process Learning via Operator Flow Matching". [https://arxiv.org/abs/2501.04126](https://arxiv.org/abs/2501.04126)

Our implementation relies on [torchcfm](https://github.com/atong01/conditional-flow-matching) , [neuraloperator](https://github.com/neuraloperator/neuraloperator) packages and benefits from repositories [FFM](https://github.com/GavinKerrigan/functional_flow_matching) and [OpFlow](https://github.com/yzshi5/OpFlow). 

## Example of applying OFM for Gaussian Process (GP) and non-GP data regression

`GP data example`
![image](https://github.com/user-attachments/assets/27430fc2-38ff-4557-91b3-3477e069e785)

`non-GP black hole example`
![image](https://github.com/user-attachments/assets/88d26ba7-b4bf-4db9-bb2f-63de3617f55b)



## Environment
Our implementation uses Anaconda and Jupyter Notebook. To set up the environment, create a conda environment:

```
# clone project
git clone https://github.com/yzshi5/SPL_OFM.git
cd SPL_OFM

# create conda environment
conda env create -f environment.yml

# Activate the `ofm` environment
conda activate ofm
```





Install the `ipykernel` to run the code in a jupyter notebook
```
conda install -c anaconda ipykernel

pip install ipykernel

python -m ipykernel install --user --name=ofm
```

## Description for folders and files
`ofm_OT_likelihood.py`, serves as the key file, see comments in the file for instructions

`util` folder contains the GP prior implementation and other helper functions

`model` folder includes FNO implementation, we also provide FNO with differential kernel 

`prior_learning` folder contains all prior learning tasks

`regression` folder contains all regression tasks

`sampling_FSGLD` folder contains the code for SGLD sampling

# Reference
```
@article{shi2025stochastic,
  title={Stochastic Process Learning via Operator Flow Matching},
  author={Shi, Yaozhong and Ross, Zachary E and Asimaki, Domniki and Azizzadenesheli, Kamyar},
  journal={arXiv preprint arXiv:2501.04126},
  year={2025}
}
```
