# Channel-Specific Prototype Network

Intrinsically explainable neural networks using channel specific prototypes. Code accompanient for Narayanan A and Bergen KJ, Prototype-based Explainable Neural Networks with Channel-specific Reasoning for Geospatial Learning Tasks. (submitted, 2026)

### Create a Python environment 

Create a python environment from the environment.yml file
```
conda env create -f environment.yml
```

### Creating the `data/` Folder

1. **Download preprocessed datasets and saved models**  
   Download the `data/` folder from the associated Zenodo repository (https://doi.org/10.5281/zenodo.18425035). This folder contains:
   - The Synthetic MNIST dataset  
   - The processed MJO dataset used in the study
   - `models/` folder with saved models and prototype information for each case study located in their respective subdirectories

2. **Download EuroSAT**  
   Download the EuroSAT dataset into the `data/` folder using TorchGeo:  
   https://torchgeo.readthedocs.io/en/latest/api/datasets.html#torchgeo.datasets.EuroSAT

3. **(Optional) Generate Synthetic MNIST**  
   To generate the Synthetic MNIST dataset from scratch, run:
   ```bash
   python generate_syntheticMNIST.py

4. **(Optional) Process MJO data from raw .nc files **  
   To process the MJO data, download raw OLR, U200 and U850 .nc files from https://zenodo.org/records/3968896 (B Toms 2020) into the data/ folder and run:
   ```bash
   python preprocess_data.py

## Loading and Running the Notebooks

Each case study includes a Python notebook that loads a trained ML model and reproduces the figures presented in the paper.

### Case Studies

- **Case Study 1:** Synthetic MNIST Classification  
- **Case Study 2:** Madden–Julian Oscillation (MJO) Phase Classification  
- **Case Study 3:** EuroSAT Land-Use Classification  

### Additional Notebooks

- Notebook for the MJO case study with an added noise channel, as described in the paper  
- Notebook for the MNIST_nc case study, which loads a prototype network with joint prototypes across channels (No Channel-Specific Prototypes)

### Running the Notebooks

To reproduce the paper figures:

1. Open `model.py` and uncomment the appropriate `exp_var_dict` corresponding to the desired case study  
   - For the MNIST_nc experiment, modify `model_nc.py` and use `load_mnist_nc.ipynb`

2. Launch and run the corresponding notebook to generate the figures

## Zenodo DOI: [![DOI](https://zenodo.org/badge/1144590605.svg)](https://doi.org/10.5281/zenodo.18434306)

## Authors

Anushka Narayanan, Brown University
anushka_narayanan@brown.edu

## Acknowledgments

Data used to replicate findings in this study is freely available. Data used in the MJO classification
task can be found at https://zenodo.org/records/3968896 (B Toms 2020). Data used in the land use classification task is obtained
from TorchGeo package at Stewart AJ, Robinson C, Corley IA, Ortiz A, Lavista Ferres JM and Banerjee A (2025) TorchGeo: Deep Learning With Geospatial Data. ACM Transactions on Spatial Algorithms and Systems 11(4), 1–28 (https://torchgeo.readthedocs.io/en/latest/api/datasets.html#torchgeo.datasets.EuroSAT). 

Code segments from Barnes EA et. al (2022) doi: 10.1175/AIES-D-22-0001.1 and Chen C et al (2019) doi: 10.48550/arXiv.1806.10574 are modified to include channel-specific prototypes
