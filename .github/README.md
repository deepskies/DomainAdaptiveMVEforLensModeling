## Domain Adaptation and Uncertainty Quantification for Gravitational Lens Modeling

![status](https://img.shields.io/badge/License-MIT-lightgrey)

---

This project combines Domain Adaptation (DA) with neural network (NN) Uncertainty Quantification (UQ) in the context of strong gravitational lens parameter prediction. We hope that this work helps take a step towards more accurate applications of deep learning models to real observed datasets, especially when the latter have limited labels. We predict the Einstein radius $\theta_\mathrm{E}$ from simulated multi-band images of strong gravitational lenses. Generally, to our knowledge, this is the first work in which domain adaptation and uncertainty quantification are combined, including for regression on an astrophysics dataset.

### UQ: Mean-variance Estimation (MVE)

For UQ, we use a mean-variance estimation (MVE) NN to predict the Einstein radius $\theta_\mathrm{E}$ and its aleatoric uncertainty $\sigma_\mathrm{al}$. 
Scientific analysis requires an estimate of uncertainty on measurements. We adopt an approach known as mean-variance estimation, which seeks to estimate the variance and control regression by minimizing the beta negative log-likelihood loss.


### Unsupervised Domain Adaptation (UDA)

Applying deep learning in science fields like astronomy can be difficult. When models trained on simulated data are applied to real data, the models frequently underperform because simulations rarely perfectly capture the full complexity of real data. Enter domain adaptation (DA), a framework for 

In this work, we use unsupervised DA (UDA), where the target data The DA technique used in this work use Maximum Mean Discrepancy (MMD) Loss to train a network to being embeddings of labeled "source" data gravitational lenses in line with unlabeled "target" gravitational lenses. With source and target datasets made similar, training on source datasets can be used with greater fidelity on target datasets.
Unuspervised DA aligns an unlabelled "target" dataset with a labeled "source" dataset, so that predictions can be performed on both with accuracy. That target domain has a domain shift that must be aligned. In our case, we add realistic astrophysical survey-like noise to strong lensing images in the target dataset, but no noise in the source dataset. 

![plot](../src/training/MVEUDA/figures/isomap_final.png)




#### Coded By: [Shrihan Agarwal](https://github.com/ShrihanSolo)
---


### Datasets
Both source and target datasets are generated using ```deeplenstronomy```. Below, we show a single 3-band image simulated using the no-noise source dataset and DES-like noise target dataset as a comparison.

![plot](../src/training/MVEUDA/figures/source_example.png)

![plot](../src/training/MVEUDA/figures/target_example.png)

The datasets with these images, as well as the training labels, can be downloaded from zenodo: https://zenodo.org/records/13647416.

---

### Installation 

#### Clone

Clone the package using:

> git clone https://github.com/deepskies/AdaptiveMVEforLensModeling

into any directory. Then, install the environments.

#### Environments

This works on linux, but has not been tested for mac, windows.
We recommend using conda. Install the environments in `envs/` using conda with the following command:

> conda env create -f training_env.yml.
  
> conda env create -f deeplenstronomy_env.yml

The `training_env.yml` is required for training the Pytorch model, and `deeplenstronomy_env.yml` for simulating strong lensing datasets using `deeplenstronomy`. Note that there is a sky brightness-related bug in the PyPI 0.0.2.3 version of deeplenstronomy, and an update to the latest version will be required for reproduction of results.

---

### Repository Structure

The repository structure is below. 

```
AdaptiveMVEforLensModeling/
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
---

### Reproducing Paper Results

#### Acquiring The Dataset

* __Option A: Generate the Dataset__
    * Navigate to `src/sim/notebooks/`.
    * Generate a source/target data pair in the `src/data/` directory:
        * Run `gen_sim.py` on `src/sim/config/source_config.yaml` and `target_config.yaml`.
    * A source and target data folder should be present in `src/data/`.
  
* __Option B: Download the Dataset__
    * Zip files of the dataset are available at https://zenodo.org/records/13647416.
    * The source and target data downloaded should be added to the `src/data/` directory.
        * Place the folders `mb_paper_source_final` and `mb_paper_target_final` into the `src/data/` directory.

#### Running Training

* __MVE-Only__
    * Navigate to `src/training/MVEonly/MVE_noDA_RunA.ipynb` (or Run B, C, D, E)
    * Adjust filepaths to the dataset if necessary.
    * Activate the `neural` conda environment.
    * Run training by running the notebook.
    * New runs by a user will be stored in the adjacent `models/` directories.
    
* __MVE-UDA__
    * Follows an identical procedure to above, in `src/training/MVEUDA/`.

#### Visualizing Paper Results

* To generate the results in the paper use the notebook `src/training/MVEUDA/ModelVizPaper.ipynb`.
    * Final figures from this notebook are stored in `src/training/MVEUDA/figures/`. 
    * Saved PyTorch models of the runs are provided in `src/training/MVE*/paper_models/`.



<br>
  
<div style="display: flex; justify-content: space-between;">
  <img src="../src/training/MVEUDA/figures/residual.png" alt="Residual Plot" style="width: 70%;"/>
  <img src="../src/training/MVEUDA/figures/resid_legend.png" alt="Residual Legend" style="width: 25%;"/>
</div>

---

### Citation 

```
@article{key , 
    author = {Shrihan Agarwal, Aleksandra Ciprijanovic, Brian Nord}, 
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
This project is a part of the DeepSkies group. We greatly appreciate advice and contributions from Jason Poh, Paxson Swierc, Megan Zhao and Becky Nevin -- this work would be impossible without building on their earlier discoveries. We used the Fermilab Elastic Analysis Facility (EAF) for computational and storage purposes in this project. Additionally, this project has used data from both the Dark Energy Survey and Dark Energy CAM Legacy Survey DR10 to generate realistic data - we thank the collaborations for making their catalogs accessible.
