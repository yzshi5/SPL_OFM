# OFM
This repository contains the code for "Stochastic Process Learning via Operator Flow Matching.

Our implementation heavily relies on [torchcfm](https://github.com/atong01/conditional-flow-matching) , [neuraloperator](https://github.com/neuraloperator/neuraloperator) packages and benefits from repositories [FFM](https://github.com/GavinKerrigan/functional_flow_matching) and [OpFlow](https://github.com/yzshi5/OpFlow). 

## Example of applying OFM for Gaussian Process (GP) and non-GP data regression

`GP data example`
![image](https://github.com/user-attachments/assets/27430fc2-38ff-4557-91b3-3477e069e785)

`non-GP black hole example`
![image](https://github.com/user-attachments/assets/88d26ba7-b4bf-4db9-bb2f-63de3617f55b)



## Environment
Our implementation uses Anaconda and Jupyter Notebook. To set up the environment, unzip the folder and create a conda environment:


`cd OFM_code`

`conda env create -f environment.yml`

Activate the `ofm` environment

`conda activate ofm`

Install the `ipykernel` to run the code in a jupyter notebook
```
conda install -c anaconda ipykernel

pip install ipykernel

python -m ipykernel install --user --name=ofm
```

## Description for folders and files
`ofm_OT_likelihood.py`, serves as the key file. We also provide  `ofm_OT_likelihood_FFT.py`, which implements the GP prior using the FFT method.

`util` folder contains the GP prior implementation and other helper functions

`model` folder includes FNO implementation, we also provide FNO with differential kernel (https://arxiv.org/abs/2402.16845)

`prior_learning` folder contains all prior learning tasks

`regression` folder contains all regression tasks

`sampling_FSGLD` folder contains the code for SGLD sampling
