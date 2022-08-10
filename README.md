# Gene expression-receptor density associations
This repository contains code and data created in support of my project, "Correspondence between gene expression and neurotransmitter receptor and transporter density in the human cortex", now up on [bioRxiv](https://www.biorxiv.org/content/10.1101/2021.11.30.469876v1).
All code was written in Python.

## Main analyses
All of the analyses can be found in [code/main.py](code/main.py), which is organized by figures.

## Data
The [data](data/) folder contains the following files/directories:
- [autoradiography](data/autoradiography/): this contains autoradiography data originally collected by [Zilles & Palomero-Gallagher, 2017](https://www.frontiersin.org/articles/10.3389/fnana.2017.00078/full), and converted to `numpy` files by Al Goulas (data originally presented [here](https://github.com/AlGoulas/receptor_principles) in support of [this paper](https://www.pnas.org/content/118/3/e2020574118/tab-article-info))
- [expression](data/expression/): this contains microarray gene expression data and differential stability for two parcellation resolutions, originally from the AHBA. Data was processed using the [abagen toolbox](https://github.com/rmarkello/abagen) (see [this paper](https://www.biorxiv.org/content/10.1101/2021.07.08.451635v1))
- [PET_receptors.csv](data/PET_receptors.csv): 68 cortical Desikan Killiany regions x 18 neurotransmitter receptor and transporter PET-derived densities, orignally collated and used [here](https://github.com/netneurolab/hansen_receptors) for [this](https://www.biorxiv.org/content/10.1101/2021.10.28.466336v1) paper. Note that the order of receptors can be found in [main.py](code/main.py)
- [PET_receptors.csv](data/PET_receptors_scale125.csv): same thing but 219 regions (I'm not the one that came up with this naming convention)
- [mesulam_mapping.csv](data/mesulam_mapping.csv): Mesulam classes of laminar differentiation for the 219-region parcellation ("scale 125")

## Figures
The [figures](figures/) folder contains `.eps` files of the figures used to make the figures in the manuscript.
I use Adobe Illustrator to make my figures.

## Results
The [results](results/) folder contains the non-figure outputs from [main.py](code/main.py).
- [autinfo.csv](results/autinfo.csv): Pearson's $r$ and $p_\text{spin}$ for each autoradiography-derived receptor-gene pair.
- [autorad_data.npy](results/autorad_data.npy): generated region x receptor matrix of autoradiography data, converted from the original 44 regions to the Desikan Killiany parcellation. Receptor names can be found in [main.py](code/main.py).
- [petinfo.csv](results/petinfo.csv): Pearson's $r$ and $p_\text{spin}$ for each PET-derived receptor-gene pair.
