import numpy as np
from scipy.stats import zscore, pearsonr
from statsmodels.stats.multitest import multipletests
from netneurotools import datasets, stats, utils
import pandas as pd
import matplotlib.pyplot as plt
import abagen

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


def corr_spin(x, y, spins, nspins):

    # convert x and y to arrays to avoid dataframe index bugs
    x = np.array(x)
    y = np.array(y)

    rho, _ = pearsonr(x, y)
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
        null[i] = df["x"].corr(df["y"])

    pval = (1 + sum(abs((null - np.mean(null))) >
                    abs((rho - np.mean(null))))) / (nspins + 1)
    return rho, pval


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
                  "MU", "NAT", "VAChT"]

# get spins
cammoun = datasets.fetch_cammoun2012()
info = pd.read_csv(cammoun['info'])
leftcortex = info.query('scale == "scale033" \
                    & structure == "cortex" \
                    & hemisphere == "L"')['id']
leftcortex = np.array(leftcortex) - 1  # python indexing
subcortex = info.query('scale == "scale033" & structure == "subcortex')['id']
subcortex = np.array(subcortex) - 1  # python indexing
coords = utils.get_centroids(cammoun['scale033'], image_space=True)
coords = coords[leftcortex, :]
nspins = 10000
spins = stats.gen_spinsamples(coords, hemiid=np.ones((len(leftcortex),)),
                              n_rotate=nspins, seed=1234)

# get gene expression
expression = abagen.get_expression_data(cammoun['scale033'], ibf_threshold=0)
expression125 = pd.read_csv(path+'data/expression/scale125_data.csv')
ds = pd.read_csv(path+'data/expression/scale033_stability.csv')

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
                   'GABAa', 'GABAa', 'H3', 'M1', 'mGluR5', 'MU', 'NAT',
                   'VAChT']

# cortex

PETcorr = {'rho' : np.zeros((len(PETgenes), )),
           'pspin' : np.zeros((len(PETgenes), ))}

plt.ion()
fig, axs = plt.subplots(5, 5, figsize=(15, 12))
axs = axs.ravel()
for i in range(len(PETgenes)):
    x = PETrecept[34:, receptor_names_p.index(PETgenes_recept[i])]
    try:  # necessary if ibf_threshold != 0 in abagen.get_expression_data()
        y = zscore(expression.iloc[leftcortex][PETgenes[i]])
    except KeyError:
        continue
    PETcorr['rho'][i], PETcorr['pspin'][i] = corr_spin(x, y, spins, nspins)
    axs[i].scatter(x, y, s=5)
    axs[i].set_xlabel(PETgenes_recept[i] + ' density')
    axs[i].set_ylabel(PETgenes[i] + ' expression')
plt.tight_layout()
plt.savefig(path+'figures/scatter_pet.eps')

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

AUTcorr = {'rho' : np.zeros((len(AUTgenes), )),
           'pspin' : np.zeros((len(AUTgenes), ))}

plt.ion()
fig, axs = plt.subplots(5, 5, figsize=(15, 12))
axs = axs.ravel()
for i in range(len(AUTgenes)):
    x = autorad_data[:, receptor_names_a.index(AUTgenes_recept[i])]
    y = zscore(expression[AUTgenes[i]])[:-1]
    AUTcorr['rho'][i], AUTcorr['pspin'][i] = corr_spin(x, y, spins, nspins)
    axs[i].scatter(x, y, s=5)
    axs[i].set_xlabel(AUTgenes_recept[i] + ' density')
    axs[i].set_ylabel(AUTgenes[i] + ' expression')
plt.tight_layout()
plt.savefig(path+'figures/scatter_aut.eps')

"""
Figure 3: differential stability
"""
plt.ion()
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
# pet
for i in range(len(PETcorr['rho'])):
    x = PETcorr['rho'][i]
    y = ds.query('genes == @PETgenes[@i]')['stability']
    ax1.scatter(x, y, c='b')
    ax1.text(x+0.01, y+0.01, PETgenes[i], fontsize=7)
ax1.set_xlabel('gene-receptor correlation')
ax1.set_ylabel('differential stability')
ax1.set_aspect(1.0/ax1.get_data_ratio(), adjustable='box')
# autorad
for i in range(len(AUTcorr['rho'])):
    x = AUTcorr['rho'][i]
    y = ds.query('genes == @AUTgenes[@i]')['stability']
    ax2.scatter(x, y, c='b')
    ax2.text(x+0.01, y+0.01, AUTgenes[i], fontsize=7)
ax2.set_xlabel('gene-receptor correlation')
ax2.set_ylabel('differential stability')
ax2.set_aspect(1.0/ax2.get_data_ratio(), adjustable='box')
plt.tight_layout()
plt.savefig(path+'figures/scatter_ds.eps')

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
        system_corr[system, gene], _  = pearsonr(x, y)

# plot for each receptor
plt.ion()
plt.figure(figsize=(12, 6))
colour = ['#bfd3e6', '#8c96c6', '#8856a7', '#810f7c']
for i in range(system_corr.shape[0]):
    plt.scatter(np.arange(len(PETgenes)), system_corr[i, :], c=colour[i])
plt.xticks(range(len(PETgenes)), PETgenes, rotation=90)
plt.tight_layout()
plt.savefig(path+'figures/scatter_hierarchy.eps')


"""
Supplementary Figure 1: PET Scale 125
"""

# get spins
leftcortex = info.query('scale == "scale125" \
                    & structure == "cortex" \
                    & hemisphere == "L"')['id']
leftcortex = np.array(leftcortex) - 1  # python indexing
coords = utils.get_centroids(cammoun['scale125'], image_space=True)
coords = coords[leftcortex, :]
spins = stats.gen_spinsamples(coords, hemiid=np.ones((len(leftcortex),)),
                              n_rotate=nspins, seed=1234)

PETcorr125 = {'rho' : np.zeros((len(PETgenes), )),
              'pspin' : np.zeros((len(PETgenes), ))}

plt.ion()
fig, axs = plt.subplots(5, 5, figsize=(15, 12))
axs = axs.ravel()
for i in range(len(PETgenes)):
    x = PETrecept125[-111:, receptor_names_p.index(PETgenes_recept[i])]
    y = zscore(expression125[PETgenes[i]])
    PETcorr125['rho'][i], PETcorr125['pspin'][i] = corr_spin(x, y, spins, nspins)
    axs[i].scatter(x, y, s=5)
    axs[i].set_xlabel(PETgenes_recept[i] + ' density')
    axs[i].set_ylabel(PETgenes[i] + ' expression')
plt.tight_layout()
plt.savefig(path+'figures/scatter_pet_scale125.eps')

"""
Supplementary Figure 2: GABAa subunits
"""

GABAAgenes = ['GABRR3', 'GABRD', 'GABRR2', 'GABRR1', 'GABRG1', 'GABRA2',
              'GABRA4', 'GABRB1', 'GABRG3', 'GABRA5', 'GABRB3', 'GABRP',
              'GABRA6','GABRE','GABRA3','GABRQ']

plt.ion()
fig, axs = plt.subplots(4, 5, figsize=(15, 10))
axs = axs.ravel()
for i in range(len(GABAAgenes)):
    x = PETrecept[34:, receptor_names_p.index('GABAa')]
    y = zscore(expression[GABAAgenes[i]])
    axs[i].scatter(x, y, s=5)
    axs[i].set_xlabel('GABAa density')
    axs[i].set_ylabel(GABAAgenes[i] + ' expression')
plt.tight_layout()
plt.savefig(path+'figures/scatter_gabaa.eps')

"""
Supplementary Figure 3: Subcortex
"""

PETcorr_subc = {'rho' : np.zeros((len(PETgenes), )),
                'pspin' : np.zeros((len(PETgenes), ))}

subc_label = info.query('scale=="scale033" & structure=="subcortex"')['label']
subc_name, subc_label = np.unique(np.array(subc_label), return_inverse=True)
plt.ion()
fig, axs = plt.subplots(5, 5, figsize=(15, 12))
axs = axs.ravel()
for i in range(len(PETgenes)):
    x = PETrecept_subc[:, receptor_names_p.index(PETgenes_recept[i])]
    try:
        y = zscore(expression.iloc[subcortex][PETgenes[i]])
    except KeyError:
        continue
    PETcorr_subc['rho'][i], PETcorr_subc['pspin'][i] = pearsonr(x, y)
    axs[i].scatter(x, y, s=10, c=subc_label)
    axs[i].set_xlabel(PETgenes_recept[i] + ' density')
    axs[i].set_ylabel(PETgenes[i] + ' expression')
plt.tight_layout()
plt.savefig(path+'figures/scatter_pet_subc.eps')

"""
Supplemental tables
"""

# PET

petinfo = {}

petinfo['genes'] = ['HTR1A', 'HTR1B', 'HTR2A', 'HTR4', 'HTR6', 'SLC6A4',  # serotonin
                    'CHRNA2', 'CHRNA3', 'CHRNA4', 'CHRNA5', 'CHRNA6', 'CHRNA7',
                    'CHRNA9', 'CHRNA10', 'CHRNB2', 'CHRNB3', 'CHRNB4',  # acetylcholine
                    'CNR1', 'DRD1', 'DRD2', 'SLC6A3',
                    'GABRA1', 'GABRB2', 'GABRG2', 'GABRR3', 'GABRD', 'GABRR2',
                    'GABRR1', 'GABRG1', 'GABRA2', 'GABRA4','GABRB1','GABRG3',
                    'GABRA5', 'GABRB3', 'GABRP', 'GABRA6','GABRE','GABRA3',
                    'GABRQ',  # 19 gabaa subunits!!
                    'HRH3', 'CHRM1', 'GRM5', 'OPRM1', 'SLC6A2', 'SLC18A3']

petinfo['receptors'] = ['5HT1a', '5HT1b', '5HT2a', '5HT4', '5HT6', '5HTT'] \
                       + ['A4B2' for i in range(11)] \
                       + ['CB1', 'D1', 'D2', 'DAT'] \
                       + ['GABAa' for i in range(19)] \
                       + ['H3', 'M1', 'mGluR5', 'MU', 'NAT', 'VAChT']

petinfo['rho'] = np.zeros((len(petinfo['genes']), ))
petinfo['pspin'] = np.zeros((len(petinfo['genes']), ))
petinfo['ci-lower'] = np.zeros((len(petinfo['genes']), ))
petinfo['ci-upper'] = np.zeros((len(petinfo['genes']), ))

def corr(x, y):
    r, _ = pearsonr(x, y)
    return r

for i in range(len(petinfo['genes'])):
    x = PETrecept[34:, receptor_names_p.index(petinfo['receptors'][i])]
    y = zscore(expression[petinfo['genes'][i]])
    petinfo['rho'][i], petinfo['pspin'][i] = corr_spin(x, y, spins, nspins)
    # petinfo['ci-lower'], petinfo['ci-upper'] = bootstrap(data=(x, y),
    #                                                      statistic=corr,
    #                                                      n_resamples=nspins,
    #                                                      method='BCa')

needs_correct = ['GABAa', 'A4B2']
for i in range(len(needs_correct)):
    index = [index for index, value in enumerate(petinfo['receptors']) if value == needs_correct[i]]
    uncorrected_pval = petinfo['pspin'][index]
    _, petinfo['pspin'][index], _, _ = multipletests(pvals=uncorrected_pval, method='fdr_bh')

petinfo = pd.DataFrame.from_dict(petinfo)
petinfo.to_csv(path+'results/petinfo.csv')

# AUTORADIOGRAPHY

autinfo = {}

autinfo['genes'] = ['GRIA1', 'GRIA2', 'GRIA3', 'GRIA4',  # AMPA
                    'GRIN1', 'GRIN2A', 'GRIN2B', 'GRIN2C', 'GRIN2D',
                    'GRIN3A', 'GRIN3B',  # NMDA
                    'GRIK1', 'GRIK2', 'GRIK3', 'GRIK4', 'GRIK5',  # kainate
                    'GABRA1', 'GABRB2', 'GABRG2', 'GABRR3', 'GABRD', 'GABRR2',
                    'GABRR1', 'GABRG1', 'GABRA2', 'GABRA4','GABRB1','GABRG3',
                    'GABRA5', 'GABRB3', 'GABRP', 'GABRA6','GABRE','GABRA3',
                    'GABRQ',  # all 19 gabaa subunits
                    'GABRA1', 'GABRB2', 'GABRG2', 'GABRR3', 'GABRD', 'GABRR2',
                    'GABRR1', 'GABRG1', 'GABRA2', 'GABRA4','GABRB1','GABRG3',
                    'GABRA5', 'GABRB3', 'GABRP', 'GABRA6','GABRE','GABRA3',
                    'GABRQ',  # all 19 gabaa subunits, again
                    'GABBR1', 'GABBR2', 'CHRM1', 'CHRM2', 'CHRM3',
                    'CHRNA2', 'CHRNA3', 'CHRNA4', 'CHRNA5', 'CHRNA6', 'CHRNA7',
                    'CHRNA9', 'CHRNA10', 'CHRNB2', 'CHRNB3', 'CHRNB4',  # acetylcholine
                    'ADRA1A', 'ADRA2A', 'HTR1A', 'HTR2A', 'DRD1']
autinfo['receptors'] = ['AMPA' for i in range(4)] \
                       + ['NMDA' for i in range(7)] \
                       + ['kainate' for i in range(5)] \
                       + ['GABAa' for i in range(19)] \
                       + ['GABAa/BZ' for i in range(19)] \
                       + ['GABAb' for i in range(2)] \
                       + ['m1', 'm2', 'm3'] \
                       + ['a4b2' for i in range(11)] \
                       + ['a1', 'a2', '5-HT1a', '5-HT2', 'D1']

autinfo['rho'] = np.zeros((len(autinfo['genes']), ))
autinfo['pspin'] = np.zeros((len(autinfo['genes']), ))
autinfo['ci-lower'] = np.zeros((len(autinfo['genes']), ))
autinfo['ci-upper'] = np.zeros((len(autinfo['genes']), ))

for i in range(len(autinfo['genes'])):
    x = autorad_data[:, receptor_names_a.index(autinfo['receptors'][i])]
    y = zscore(expression[autinfo['genes'][i]])[:-1]
    autinfo['rho'][i], autinfo['pspin'][i] = corr_spin(x, y, spins, nspins)
    # autinfo['ci-lower'], autinfo['ci-upper'] = bootstrap((x, y),   
    #                                                      pearsonr,
    #                                                      n_resamples=nspins,
    #                                                      method='percentile')

needs_correct = ['AMPA', 'NMDA', 'kainate', 'GABAa', 'GABAa/BZ', 'GABAb', 'a4b2']
for i in range(len(needs_correct)):
    index = [index for index, value in enumerate(autinfo['receptors']) if value == needs_correct[i]]
    uncorrected_pval = autinfo['pspin'][index]
    _, autinfo['pspin'][index], _, _ = multipletests(pvals=uncorrected_pval, method='fdr_bh')

autinfo = pd.DataFrame.from_dict(autinfo)
autinfo.to_csv(path+'results/autinfo.csv')