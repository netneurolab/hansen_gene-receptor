"""
script for reproducing main results,
after running `correlate_expression_density.py`
"""

import numpy as np
from scipy.stats import zscore, spearmanr
from netneurotools import datasets, stats, utils
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def brodmann2dk(M, duplicate, mapping):
    """
    Converts 44 autoradiography regions to 33 Desikan Killiany.

    Parameters
    ----------
    M : (44, 15) np array
        Autoradiography receptor densities from Zilles & Palomero-Gallagher.
    duplicate : (1, 8) array
        Autoradiography regions to duplicate.
    mapping : (1, 52) np array
        DK indices mapped to Brodmann/Jubrain regions including duplicates.

    Returns
    -------
    convertedM : (33, 15) np array
        Autoradiography densities mapped to 33 DK regions (insula excluded).

    """

    rep = np.ones((M.shape[0], ), dtype=int)  # duplicate regions
    rep[duplicate] = 2
    M = np.repeat(M, rep, 0)

    # convert to dk
    n_dknodes = max(mapping) + 1  # number of nodes in dk atlas (left hem only)

    u = np.unique(mapping)
    convertedM = np.zeros((n_dknodes, M.shape[1]))
    for i in range(len(u)):
        if sum(mapping == u[i]) > 1:
            convertedM[u[i], :] = np.mean(M[mapping == u[i], :], axis=0)
        else:
            convertedM[u[i], :] = M[mapping == u[i], :]

    return convertedM


def get_perm_p(emp, null, twotailed=True):
    if twotailed:
        return (1 + sum(abs(null - np.nanmean(null))
                        > abs(emp - np.nanmean(null)))) / (len(null) + 1)
    else:
        return (1 + sum(null > emp)) / (len(null) + 1)


def corr_spin(x, y, spins, nspins, twotailed=True):

    # convert x and y to arrays to avoid dataframe index bugs
    x = np.array(x)
    y = np.array(y)

    rho, _ = spearmanr(x, y)
    null = np.zeros((nspins,))

    if len(x) == spins.shape[0] - 1:  # if insula is missing
        x = np.append(x, np.nan)
        y = np.append(y, np.nan)

    # null correlation
    for i in range(nspins):
        tmp = y[spins[:, i]]
        # convert to dataframe in case insula is missing
        # for better handling of nans
        df = pd.DataFrame(np.stack((x, tmp)).T, columns=['x', 'y'])
        null[i] = df["x"].corr(df["y"], method='spearman')

    pval = get_perm_p(rho, null, twotailed)

    return rho, pval, null


def get_boot_ci(x, y, nboot=1000):
    bootstat = np.zeros((nboot, ))
    for i in range(nboot):
        bootsamp = np.random.choice(len(x), size=len(x), replace=True)
        bootstat[i] = spearmanr(x[bootsamp], y[bootsamp])[0]
    return np.array([np.percentile(bootstat, 2.5), np.percentile(bootstat, 97.5)])

"""
load data
"""

path = 'C:/Users/justi/OneDrive - McGill University/MisicLab/proj_genes_receptors/\
github/hansen_gene-receptor/'

# PET receptor map
PETrecept = np.genfromtxt(path+'data/PET_receptors.csv', delimiter=',')
PETrecept = zscore(PETrecept)
PETrecept125 = np.genfromtxt(path+'data/PET_receptors_scale125.csv', delimiter=',')
PETrecept125 = zscore(PETrecept125)
PETrecept_subc = np.genfromtxt(path+'data/PET_receptors_subcortex.csv', delimiter=',')
PETrecept_subc = zscore(PETrecept_subc)
receptor_names_p = ["5HT1a", "5HT1b", "5HT2a", "5HT4", "5HT6", "5HTT", "A4B2",
                  "CB1", "D1", "D2", "DAT", "GABAa", "H3", "M1", "mGluR5",
                  "MOR", "NET", "VAChT"]

# get spins
scale = 'scale033'
cammoun = datasets.fetch_cammoun2012()
info = pd.read_csv(cammoun['info'])
leftcortex = info.query('scale == @scale \
                    & structure == "cortex" \
                    & hemisphere == "L"')['id']
leftcortex = np.array(leftcortex) - 1  # python indexing
subcortex = info.query('scale == @scale & structure == "subcortex"')['id']
subcortex = np.array(subcortex) - 1  # python indexing
coords = utils.get_centroids(cammoun[scale], image_space=True)
coords = coords[leftcortex, :]
nspins = 10000
spins = stats.gen_spinsamples(coords, hemiid=np.ones((len(leftcortex),)),
                              n_rotate=nspins, seed=1234)

# get scale125 spins
leftcortex125 = info.query('scale == "scale125" \
                    & structure == "cortex" \
                    & hemisphere == "L"')['id']
leftcortex125 = np.array(leftcortex125) - 1  # python indexing
coords125 = utils.get_centroids(cammoun['scale125'], image_space=True)
coords125 = coords125[leftcortex125, :]
spins125 = stats.gen_spinsamples(coords125, hemiid=np.ones((len(leftcortex125),)),
                                 n_rotate=nspins, seed=1234)

# get gene expression
rnaseq = False
if not(rnaseq):
    expression = pd.read_csv(path+'data/expression/scale033_data.csv')
    expression125 = pd.read_csv(path+'data/expression/scale125_data.csv')
    ds = pd.read_csv(path+'data/expression/scale033_stability.csv')
else:
    expression = pd.read_csv(path+'data/expression/rnaseq_data/scale033_data_rnaseq.csv')
    expression125 = pd.read_csv(path+'data/expression/rnaseq_data/scale125_data_rnaseq.csv')
    ds = pd.read_csv(path+'data/expression/rnaseq_data/scale033_stability_rnaseq.csv')

# fetch gene expression data:
# expression = abagen.get_expression_data(cammoun['scale033'], ibf_threshold=0)
# expression = expression.iloc[leftcortex]

# correlation coefficients
PETcorrs = pd.read_csv(path+'results/PETcorrs_scale033_microarray.csv')
AUTcorrs = pd.read_csv(path+'results/AUTcorrs_microarray.csv')

# plotting set up
plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams['font.size'] = 8.0

"""
make autoradiography receptor matrix
"""

receptdata_s = np.load(path+'data/autoradiography/ReceptData_S.npy')  # supra
receptdata_g = np.load(path+'data/autoradiography/ReceptData_G.npy')  # granular
receptdata_i = np.load(path+'data/autoradiography/ReceptData_I.npy')  # infra
receptor_names_a = np.load(path+'data/autoradiography/ReceptorNames.npy').tolist()

# region indeces associated with more than one dk region
duplicate = [20, 21, 28, 29, 30, 32, 34, 39]

# mapping from 44 brodmann areas + 7 duplicate regions to dk left hem
# manually done, comparing anatomical regions to one another
# originally written in matlab and too lazy to change indices hence the -1
# the index refers to the cammoun scale033 structure name
mapping = np.array([57, 57, 57, 57, 63, 62, 65, 62, 64, 65, 64, 66, 66,
                    66, 66, 66, 74, 74, 70, 71, 72, 73, 67, 68, 69, 52,
                    52, 60, 60, 58, 58, 59, 53, 54, 53, 54, 55, 56, 61,
                    51, 51, 50, 49, 49, 44, 44, 45, 42, 47, 46, 48, 43])
mapping = mapping - min(mapping)  # python indexing

# convert
# note: insula (last idx of cammoun atlas) is missing
receptdata_s = zscore(brodmann2dk(receptdata_s, duplicate, mapping))
receptdata_g = zscore(brodmann2dk(receptdata_g, duplicate, mapping))
receptdata_i = zscore(brodmann2dk(receptdata_i, duplicate, mapping))

# average across layers
# final region x receptor autoradiography receptor dataset
autorad_data = np.mean(np.stack((receptdata_s,
                                 receptdata_g,
                                 receptdata_i), axis=2), axis=2)
np.save(path+'results/autorad_data.npy', autorad_data)

"""
Figure 1: PET receptors
"""

PETgenes = ['HTR1A', 'HTR1B', 'HTR2A', 'HTR4', 'HTR6', 'SLC6A4', 'CHRNA4',
            'CHRNB2', 'CNR1', 'DRD1', 'DRD2', 'SLC6A3', 'GABRA1', 'GABRB2',
            'GABRG2', 'HRH3', 'CHRM1', 'GRM5', 'OPRM1', 'SLC6A2', 'SLC18A3']
PETgenes_recept = ['5HT1a', '5HT1b', '5HT2a', '5HT4', '5HT6', '5HTT',
                   'A4B2', 'A4B2', 'CB1', 'D1', 'D2', 'DAT', 'GABAa',
                   'GABAa', 'GABAa', 'H3', 'M1', 'mGluR5', 'MOR', 'NET',
                   'VAChT']

# cortex

plt.ion()
fig, axs = plt.subplots(5, 5, figsize=(15, 12))
axs = axs.ravel()
for i in range(len(PETgenes)):
    x = PETrecept[34:, receptor_names_p.index(PETgenes_recept[i])]
    try:  # necessary if ibf_threshold != 0 in abagen.get_expression_data()
        y = zscore(expression[PETgenes[i]])
    except KeyError:
        continue
    row = PETcorrs.query('genes == @PETgenes[@i] & receptors == @PETgenes_recept[@i]')
    r = np.squeeze(np.array(row['cortex-rho']))
    p = np.squeeze(np.array(row['cortex-pspin']))
    axs[i].scatter(x, y, s=5)
    axs[i].set_xlabel(PETgenes_recept[i] + ' density')
    axs[i].set_ylabel(PETgenes[i] + ' expression')
    axs[i].set_title(['r=' + str(r)[:5] + ', p= ' + str(p)[:5]])
plt.tight_layout()
plt.savefig(path+'figures/scatter_pet.svg')

"""
Figure 2: autoradiography receptors
"""

AUTgenes = ['GRIA1', 'GRIN1', 'GRIK2', 'GABRA1', 'GABRB2', 'GABRG2',
            'GABRA1', 'GABRG2', 'GABRB2', 'GABBR1', 'GABBR2', 'CHRM1',
            'CHRM2', 'CHRM3', 'CHRNA4', 'CHRNB2', 'ADRA1A', 'ADRA2A',
            'HTR1A', 'HTR2A', 'DRD1']
AUTgenes_recept = ['AMPA', 'NMDA', 'kainate', 'GABAa', 'GABAa', 'GABAa',
                   'GABAa/BZ', 'GABAa/BZ', 'GABAa/BZ', 'GABAb', 'GABAb',
                   'm1', 'm2', 'm3', 'a4b2', 'a4b2', 'a1', 'a2', '5-HT1a',
                   '5-HT2', 'D1']

plt.ion()
fig, axs = plt.subplots(5, 5, figsize=(15, 12))
axs = axs.ravel()
for i in range(len(AUTgenes)):
    print(AUTgenes[i])
    x = autorad_data[:, receptor_names_a.index(AUTgenes_recept[i])]
    try:  # necessary if ibf_threshold != 0 in abagen.get_expression_data()
        y = zscore(expression[AUTgenes[i]][:-1])
    except KeyError:
        continue
    row = AUTcorrs.query('genes == @AUTgenes[@i] & receptors == @AUTgenes_recept[@i]')
    r = np.squeeze(np.array(row['rho']))
    p = np.squeeze(np.array(row['pspin']))
    axs[i].scatter(x, y, s=5)
    axs[i].set_xlabel(AUTgenes_recept[i] + ' density')
    axs[i].set_ylabel(AUTgenes[i] + ' expression')
    axs[i].set_title(['r=' + str(r)[:5] + ', p= ' + str(p)[:5]])
plt.tight_layout()
plt.savefig(path+'figures/scatter_aut.svg')

"""
Figure 3: differential stability
"""

plt.ion()
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
# pet
xarr = []
yarr = []
for i in range(len(PETgenes)):
    x = np.squeeze(np.array(PETcorrs.query('genes == @PETgenes[@i] \
                                            & receptors == @PETgenes_recept[@i]')['cortex-rho']))
    y = ds.query('genes == @PETgenes[@i]')['stability']
    ax1.scatter(x, y, c='b')
    ax1.text(x+0.01, y+0.01, PETgenes[i], fontsize=7)
    xarr.append(x)
    yarr.append(y)
sns.regplot(x=np.array(xarr), y=np.array(yarr),
            scatter=False, ci=False, ax=ax1)
ax1.set_xlabel('gene-receptor correlation')
ax1.set_ylabel('differential stability')
ax1.set_aspect(1.0/ax1.get_data_ratio(), adjustable='box')
r, p = spearmanr(xarr, yarr)
ax1.set_title(['r=' + str(r)[:5] + ', p=' + str(p)[:6]])
# autorad
xarr = []
yarr = []
for i in range(len(AUTgenes)):
    x = np.squeeze(np.array(AUTcorrs.query('genes == @AUTgenes[@i] \
                                            & receptors == @AUTgenes_recept[@i]')['rho']))
    y = ds.query('genes == @AUTgenes[@i]')['stability']
    ax2.scatter(x, y, c='b')
    ax2.text(x+0.01, y+0.01, AUTgenes[i], fontsize=7)
    xarr.append(x)
    yarr.append(y)
sns.regplot(x=np.array(xarr), y=np.array(yarr),
            scatter=False, ci=False, ax=ax2)
ax2.set_xlabel('gene-receptor correlation')
ax2.set_ylabel('differential stability')
ax2.set_aspect(1.0/ax2.get_data_ratio(), adjustable='box')
r, p = spearmanr(xarr, yarr)
ax2.set_title(['r=' + str(r)[:5] + ', p=' + str(p)[:6]])
plt.tight_layout()
plt.savefig(path+'figures/scatter_ds.svg')

"""
Figure 4: functional heirarchy
"""

mesulam = np.genfromtxt(path+'data/mesulam_mapping.csv', delimiter=',')
mesulam = mesulam[-111:].astype(int)
system_corr = np.zeros((4, len(PETgenes)))
for system in range(system_corr.shape[0]):
    for gene in range(len(PETgenes)):
        x = PETrecept125[-111:, receptor_names_p.index(PETgenes_recept[gene])][mesulam==system+1]
        y = zscore(expression125[PETgenes[gene]])[mesulam==system+1]
        system_corr[system, gene], _  = spearmanr(x, y)

# get confidence interval for correlations
boots = np.zeros((len(PETgenes), 2))
for gene in range(len(PETgenes)):
    x = PETrecept125[-111:, receptor_names_p.index(PETgenes_recept[gene])]
    y = zscore(expression125[PETgenes[gene]])
    boots[gene, :] = get_boot_ci(x, y, nboot=1000)

# plot for each receptor
plt.ion()
plt.figure(figsize=(12, 6))
colour = ['#bfd3e6', '#8c96c6', '#8856a7', '#810f7c']
for i in range(system_corr.shape[0]):
    lw = np.array([boots[j, 0] <= system_corr[i, j] <= boots[j, 1] for j in range(boots.shape[0])])
    lw = 2*(1 - lw.astype(int))
    plt.scatter(np.arange(len(PETgenes)), system_corr[i, :], linewidths=lw, color=colour[i])
plt.xticks(range(len(PETgenes)), PETgenes, rotation=90)
plt.tight_layout()
plt.savefig(path+'figures/scatter_hierarchy.eps')

"""
Figure 5: subcortex
"""

subc_label = info.query('scale=="scale033" & structure=="subcortex"')['label']
subc_name, subc_label = np.unique(np.array(subc_label), return_inverse=True)
expression_subc = pd.read_csv(path+'data/expression/scale033_data_subcortex.csv')
plt.ion()
fig, axs = plt.subplots(5, 5, figsize=(15, 12))
axs = axs.ravel()
for i in range(len(PETgenes)):
    x = PETrecept_subc[:, receptor_names_p.index(PETgenes_recept[i])]
    try:
        y = zscore(expression_subc[PETgenes[i]])
    except KeyError:
        continue
    row = PETcorrs.query('genes == @PETgenes[@i] & receptors == @PETgenes_recept[@i]')
    r = np.squeeze(np.array(row['subcortex-rho']))
    p = np.squeeze(np.array(row['subcortex-pval']))
    axs[i].scatter(x, y, s=10, c=subc_label)
    axs[i].set_xlabel(PETgenes_recept[i] + ' density')
    axs[i].set_ylabel(PETgenes[i] + ' expression')
    axs[i].set_title(['r=' + str(r)[:5] + ', p=' + str(p)[:5]])
plt.tight_layout()
plt.savefig(path+'figures/scatter_pet_subc.svg')
