# Gene expression-receptor density associations
This repository contains code and data created in support of my project, ["Correspondence between gene expression and neurotransmitter receptor and transporter density in the human cortex"](https://www.sciencedirect.com/science/article/pii/S1053811922007923?via%3Dihub), published in NeuroImage.
All code was written in Python.

## Main analyses
Code to reproduce figures in the main text can be found in [code/main.py](code/main.py).
Code for supplementary analyses can be found in [code/supplement.py](code/supplement.py).
Running all gene expression-receptor density correlations (PET, autoradiography, microarray, RNAseq, cortex, subcortex, etc...) with spin-tests can be done using [code/correlate_expression_density.py](code/correlate_expression_density.py).

## Data
The [data](data/) folder contains the following files/directories:
- [autoradiography](data/autoradiography/): this contains autoradiography data originally collected by [Zilles & Palomero-Gallagher, 2017](https://www.frontiersin.org/articles/10.3389/fnana.2017.00078/full), and available as `numpy` files (data originally presented [here](https://github.com/AlGoulas/receptor_principles) in support of [this paper](https://www.pnas.org/content/118/3/e2020574118/tab-article-info))
- [expression](data/expression/): this contains microarray and RNAseq gene expression data and differential stability for two parcellation resolutions, originally from the AHBA. Data was processed using the [abagen toolbox](https://github.com/rmarkello/abagen) (see [this paper](https://www.biorxiv.org/content/10.1101/2021.07.08.451635v1)). Gene expression was estimated using multiple different probe selection methods (see [Supplement](hansen2022neuroimage_supplement.pdf) hence the repeats.
- [PET_receptors.csv](data/PET_receptors.csv): 68 cortical Desikan Killiany regions x 18 neurotransmitter receptor and transporter PET-derived densities, orignally collated and used [here](https://github.com/netneurolab/hansen_receptors) for [this](https://www.biorxiv.org/content/10.1101/2021.10.28.466336v1) paper. Note that the order of receptors can be found in [main.py](code/main.py)
- [PET_receptors_scale125.csv](data/PET_receptors_scale125.csv): same thing but 219 regions (I'm not the one that came up with this naming convention)
- [PET_receptors_subcortex.csv](data/PET_receptors_subcortex.csv'): same thing but for 15 subcortical regions.
- [panther_ontologies](panther_ontologies/) are lists of genes involved in the protein pathway of each neurotransmitter (e.g. "acetylcholine"), derived from the (PANTHER Classification System)(http://pantherdb.org/panther/globalSearch.jsp?). 

## Results
The [results](results/) folder contains:
- [AUTcorrs_microarray.csv](results/AUTcorrs_microarray.csv) and [AUTcorrs_rnaseq.csv](results/AUTcorrs_rnaseq.csv): Spearman $r$ and $p_\text{spin}$ for each autoradiography-derived receptor-gene pair using microarray and RNAseq gene expression data.
- [autorad_data.npy](results/autorad_data.npy): generated region x receptor matrix of autoradiography data, converted from the original 44 regions to the Desikan Killiany parcellation. Receptor names can be found in [main.py](code/main.py).
- [PETcorrs_scale033_microarray.csv](results/PETcorrs_scale033_microarray.csv), [PETcorrs_scale033_rnaseq.csv](results/PETcorrs_scale033_rnaseq.csv), and [PETcorrs_scale125_microarray.csv](results/PETcorrs_scale125_microarray.csv): Spearman $r$ and $p_\text{spin}$ for each PET-derived receptor-gene pair using microarray and RNAseq gene expression data, under two parcellations (Cammoun-033, regionally equivalent to Desikan Killiany, and Cammoun-125, a subdivision of Cammoun-033).
- [layercorrs.npz](results/layercorrs.npz): my saved output of the autoradiography expression-density correlations within three laminar layers.
- [panther.csv](results/panther.csv): the PANTHER Classification output (see [Supplement](hansen2022neuroimage_supplement.pdf)).
