## Domain Adaptation and Uncertainty Quantification for Gravitational Lens Modeling

![status](https://img.shields.io/badge/License-MIT-lightgrey)

This project combines the emerging field of Domain Adaptation with Uncertainty Quantification, working towards applying machine learning to real scientific datasets with limited labelled data. For this project, simulated images of strong gravitational lenses are used as source and target dataset, and the Einstein radius $\theta_E$ and its uncertainty $\Delta \theta_E$ are determined through regression. 

Applying machine learning in science domains such as astronomy is difficult. With models trained on simulated data being applied to real data, models frequently underperform - simulations cannot perfectlty capture the true complexity of real data. Enter domain adaptation (DA). The DA techniques used in this work use Maximum Mean Discrepancy Loss to train a network to being embeddings of labelled "source" data gravitational lenses in line with unlabeled "target" gravitational lenses. With source and target datasets made similar, training on source datasets can be used with greater fidelity on target datasets.

Scientific analysis requires an estimate of uncertainty on measurements. We adopt an approach known as mean-variance estimation, which seeks to estimate the variance and control regression by minimizing the beta negative log-likelihood loss. To our knowledge, this is the first time that domain adaptation and uncertainty quantification are being combined, especially for regression on an astrophysical dataset.


### Installation 

#### Clone

Clone the package using:

> git clone https://github.com/deepskies/DAUQ_LensModeling

into any directory. No further setup is required once environments are installed.

#### Environments

This works on linux, but has not been tested for mac, windows.
Install the environments in `envs/` using conda with the following command:

> conda env create -f training_env.yml.
  
> conda env create -f deeplenstronomy_env.yml

The `training_env.yml` is required for training the Pytorch model, and `deeplenstronomy_env.yml` for simulating strong lensing datasets using `deeplenstronomy`.


### Quickstart

In order to reproduce results, you will first need to generate the datasets. Navigate to `src/sim/notebooks` and generate a source target dataset pair as specified in `src/sim/config`. You will need to use the `deeplens` environment to do so.

Once that is generated, you can navigate to `src/training/MVE/MVE_SL_DA_v1.ipynb` and run the training after updating the path to the data in the file. You will need the `neural` environment to do so.


### Citation 

```
@article{key , 
    author = {Shrihan Agarwal}, 
    title = {Domain-adaptive neural network prediction with
    uncertainty quantification for strong gravitational lens
    analysis}, 
    journal = {NEURIPS}, 
    volume = {v}, 
    year = {2024}, 
    number = {X}, 
    pages = {XX--XX}
}
```

### Acknowledgement 
Include any acknowledgements for research groups, important collaborators not listed as a contributor, institutions, etc. 
