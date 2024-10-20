# Domain Adaptation and Uncertainty Quantification for Gravitational Lens Modeling

![status](https://img.shields.io/badge/License-MIT-lightgrey)


<p align="justify"> 
This project combines Domain Adaptation (DA) with neural network Uncertainty Quantification (UQ) in the context of strong gravitational lens parameter prediction. We hope that this work helps take a step towards more accurate applications of deep learning models to real observed datasets, especially when the latter have limited labels. We predict the Einstein radius $\theta_\mathrm{E}$ from simulated multi-band images of strong gravitational lenses. Generally, to our knowledge, this is the first work in which domain adaptation and uncertainty quantification are combined, including for regression on an astrophysics dataset.
</p>

&nbsp;
&nbsp;

## UQ: Mean-variance Estimation (MVE)

<p align="justify"> 
For UQ, we use a mean-variance estimation (MVE) network to predict the Einstein radius $\theta_\mathrm{E}$ and its aleatoric uncertainty $\sigma_\mathrm{al}$. Scientific analysis requires an estimate of uncertainty on measurements. We adopt an approach known as mean-variance estimation, which seeks to estimate the variance and control regression by minimizing the beta negative log-likelihood loss.
</p>

&nbsp;



## Unsupervised Domain Adaptation (UDA)

<p align="justify">
Applying deep learning in science contexts like astronomy presents multiple challenges. For example, when models trained on simulated data are applied to real data, they tend to underperform because simulations rarely adequately represent the complexity of real data. Domain adaptation (DA) is a class of algorithms that are designed to address biases that can result from training networks on one data set and applying them to test data, where the training and testing data have significantly different generating parameters and/or features. Typically, the source data are from one domain, and the target data are from another distinct domain --- e.g., the data-generating parameters have different prior distributions. 
</p>

<p align="justify"> 
Usually, a supervised DA algorithm uses a large amount of labeled source data and a small amount of unlabeled target data to bridge the gap between domains. In this work, we use unsupervised DA (UDA), where the target data do not have labels. Unsupervised DA aligns the latent space embedding of an unlabeled target dataset with that of a labeled source dataset so that predictions can be performed on both. We use the Maximum Mean Discrepancy (MMD) loss to train a network of labeled source lenses in combination with unlabeled target lenses. That target domain has a domain shift that must be aligned. In this work, we use noise parameter settings to incur a domain shift between the source and target data: the source data has no noise, while the target data has the noise of the Dark Energy Survey. 
</p>

&nbsp;

<img src="../src/training/MVEUDA/figures/isomap_final.png" width=80% height=80%>

&nbsp;



## Datasets

<p align="justify">
  
We generate strong lensing images for training and testing with `deeplenstronomy`. In the figure below, we show a single simulated strong lens in three bands ($g$, $r$, $z$) without noise (source domain; upper panel) and with DES-like noise (target domain; lower panel). The datasets (images and labels) can be downloaded from the project's [Zenodo site](https://zenodo.org/records/13647416).

&nbsp;

<img src="../src/training/MVEUDA/figures/source_example.png" width=80% height=80%>
<img src="../src/training/MVEUDA/figures/target_example.png" width=80% height=80%>
 
</p>

&nbsp;


## Installation 

Clone the package into any directory:
> git clone https://github.com/deepskies/DomainAdaptiveMVEforLensModeling

Create environments with `conda` for training and for simulation, respectively:

> conda env create -f training_env.yml.

> conda env create -f deeplenstronomy_env.yml


<p align="justify">
  
A `yaml` file (i.e., `training_env.yml`) is required for training the `pytorch` neural network model, and `deeplenstronomy_env.yml` is required for simulating strong lensing datasets with `deeplenstronomy`. 

This code works on Linux but has not been tested for Mac or Windows.

There is a sky brightness-related bug in the PyPI 0.0.2.3 version of `deeplenstronomy`, and an update to the latest version will be required to reproduce the results. 

</p>

&nbsp;



## Reproducing the Paper Results

### Acquiring the Dataset

* __Option A: Generate the Dataset__
    * Navigate to `src/sim/notebooks/`.
    * Generate a source/target data pair in the `src/data/` directory by running `gen_sim.py` on the yaml files (`src/sim/config/source_config.yaml` and `src/sim/config/target_config.yaml` for source and target, respectively):
        * > gen_sim.py src/sim/config/source_config.yaml src/sim/config/target_config.yaml
  
* __Option B: Download the Dataset__
    * Zip files of the dataset are available through [Zenodo](https://zenodo.org/records/13647416).
    * The source and target data downloaded should be added to the `src/data/` directory.
        * Move or copy the directories `mb_paper_source_final` and `mb_paper_target_final` into the `src/data/` directory.

&nbsp;

### Training the Model

* __MVE-Only__
    * Navigate to `src/training/MVEonly/MVE_noDA_RunA.ipynb` (or the notebook for runs B, C, D, or E)
    * Activate the conda environment that is related to training:
         * > source activate "..."
    * Use the notebook `src/sim/notebooks/training.ipynb` to train the model.
    * The trained model parameters will be stored in the `models/` directory.
    
* __MVE-UDA__
    * Follow an identical procedure to the above, replacing `src/training/MVEonly/` with `src/training/MVEUDA/`.

&nbsp;

### Visualizing the Paper Results

* To generate the results in the paper, use the notebook `src/training/MVEUDA/ModelVizPaper.ipynb`.
    * Final figures from this notebook are stored in `src/training/MVEUDA/figures/`. 
    * Saved PyTorch models of the runs are provided in `src/training/MVE*/paper_models/`.

  
<div style="display: flex; justify-content: space-between;">
  <img src="../src/training/MVEUDA/figures/residual.png" alt="Residual Plot" style="width: 70%;"/>
  <img src="../src/training/MVEUDA/figures/resid_legend.png" alt="Residual Legend" style="width: 25%;"/>
</div>

&nbsp;

## Repository File Structure

```
DomainAdaptiveMVEforLensModeling/
│
├── src/
│   ├── sim/
│   │   ├── configs/
│   │   │   └── deeplenstronomy config files to generate the data
│   │   │
│   │   └── notebooks/
│   │       └── gen_sim.ipynb: used to generate the data in data/.
│   │   
│   │
│   ├── data/
│   │   └── Data should be stored here after download or generation.
│   │
│   └── training/
│       ├── MVEonly/
│       │   ├── paper_models/
│       │   │   └── Final PyTorch models in the MVEonly model + training information.
│       │   │
│       │   └── RunA.ipynb
│       │       └── Notebook(s) with different seeds required to run the MVEonly model.
│       │
│       └── MVEUDA/
│           ├── paper_models/
│           │   └── Final PyTorch models in the MVEonly model + training information.
│           │
│           ├── figures/
│           │   └── All figures in the paper are drawn from here.
│           | 
│           ├── RunA.ipynb
│           │   └── Notebook(s) with different seeds required to run the MVE-UDA model.
│           │
│           └── ModelVizPaper.ipynb
│               └── Notebook used to generate figures in figures/ from data in paper_models/.
│
└── envs/
    └── Conda environment specification files.

[ASCII formatting generated using ChatGPT]
```

&nbsp;


## Citation 

This code was written by [Shrihan Agarwal](https://github.com/ShrihanSolo).

```tex
@article{agarwal2024, 
    author = {Shrihan Agarwal, Aleksandra Ciprijanovic, Brian Nord}, 
    title = {Domain-adaptive neural network prediction with
    uncertainty quantification for strong gravitational lens
    analysis}, 
    journal = {Accepted to the Machine Learning for the Physical Sciences workshop at Neurips 2024}, 
    year = {2024}
}
```

&nbsp;

### Acknowledgement 
This project is a part of the [DeepSkiesLab](https://deepskieslab.com). We greatly appreciate advice and contributions from Jason Poh, Paxson Swierc, Megan Zhao, and Becky Nevin; this work would be impossible without building on their earlier discoveries. We used the [Fermilab Elastic Analysis Facility (EAF)](https://eafjupyter.readthedocs.io/) for computational and storage purposes in this project. This project  used data from both the Dark Energy Survey and Dark Energy CAM Legacy Survey DR10 to generate realistic data; we thank the collaborations for making their catalogs accessible.
