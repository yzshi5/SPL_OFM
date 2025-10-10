# Stochastic Process Learning via Operator Flow Matching (NeurIPS 2025 Spotlight)
This repository contains the code for "Stochastic Process Learning via Operator Flow Matching" (NeurIPS 2025 Spotlight, top 3.5%). [https://arxiv.org/abs/2501.04126](https://arxiv.org/abs/2501.04126)

Our implementation relies on [torchcfm](https://github.com/atong01/conditional-flow-matching) , [neuraloperator](https://github.com/neuraloperator/neuraloperator) packages and benefits from repositories [FFM](https://github.com/GavinKerrigan/functional_flow_matching) and [OpFlow](https://github.com/yzshi5/OpFlow). 

## Example of applying OFM for Gaussian Process (GP) and non-GP data regression
'two-phase strategy'
<img width="940" height="318" alt="Screenshot 2025-10-10 at 4 42 26 PM" src="https://github.com/user-attachments/assets/8ef24137-c105-441c-9737-58666453c517" />

`GP data example`
<img width="1023" height="197" alt="Screenshot 2025-10-10 at 4 48 26 PM" src="https://github.com/user-attachments/assets/07b556db-f8f4-4556-bbf7-fef6adf129bc" />

`non-GP black hole example`

<img width="885" height="349" alt="Screenshot 2025-10-10 at 4 46 31 PM" src="https://github.com/user-attachments/assets/e8b382f7-0493-4edc-afec-e8d724163b4e" />



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
