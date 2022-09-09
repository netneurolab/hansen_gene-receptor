"""
script for reproducing supplementary results
"""

import numpy as np
from scipy.stats import zscore, spearmanr
from statsmodels.stats.multitest import multipletests
from netneurotools import datasets, stats, utils
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

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

# PET receptor data
PETrecept = np.genfromtxt(path+'data/PET_receptors.csv', delimiter=',')
PETrecept = zscore(PETrecept)
PETrecept125 = np.genfromtxt(path+'data/PET_receptors_scale125.csv', delimiter=',')
PETrecept125 = zscore(PETrecept125)
PETrecept_subc = np.genfromtxt(path+'data/PET_receptors_subcortex.csv', delimiter=',')
PETrecept_subc = zscore(PETrecept_subc)
receptor_names_p = ["5HT1a", "5HT1b", "5HT2a", "5HT4", "5HT6", "5HTT", "A4B2",
                  "CB1", "D1", "D2", "DAT", "GABAa", "H3", "M1", "mGluR5",
                  "MOR", "NET", "VAChT"]

# autoradiography receptor data
receptdata_s = np.load(path+'data/autoradiography/ReceptData_S_scale033.npy')  # supra
receptdata_g = np.load(path+'data/autoradiography/ReceptData_G_scale033.npy')  # granular
receptdata_i = np.load(path+'data/autoradiography/ReceptData_I_scale033.npy')  # infra
receptor_names_a = np.load(path+'data/autoradiography/ReceptorNames.npy').tolist()
autorad_data = np.load(path+'results/autorad_data.npy')

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
expression_ma = pd.read_csv(path+'data/expression/scale033_data.csv')
expression_subc = pd.read_csv(path+'data/expression/scale033_data_subcortex.csv')
expression125 = pd.read_csv(path+'data/expression/scale125_data.csv')
ds = pd.read_csv(path+'data/expression/scale033_stability.csv')

expression_rna = pd.read_csv(path+'data/expression/rnaseq_data/scale033_data_rnaseq.csv')

# correlation coefficients
PETcorrs_scale033_microarray = pd.read_csv(path+'results/PETcorrs_scale033_microarray.csv')
AUTcorrs_microarray = pd.read_csv(path+'results/AUTcorrs_microarray.csv')
PETcorrs_scale033_rnaseq = pd.read_csv(path+'results/PETcorrs_scale033_rnaseq.csv')
AUTcorrs_rnaseq = pd.read_csv(path+'results/AUTcorrs_rnaseq.csv')
PETcorrs_scale125_microarray = pd.read_csv(path+'results/PETcorrs_scale125_microarray.csv')

# receptor/gene names
PETgenes = ['HTR1A', 'HTR1B', 'HTR2A', 'HTR4', 'HTR6', 'SLC6A4', 'CHRNA4',
            'CHRNB2', 'CNR1', 'DRD1', 'DRD2', 'SLC6A3', 'GABRA1', 'GABRB2',
            'GABRG2', 'HRH3', 'CHRM1', 'GRM5', 'OPRM1', 'SLC6A2', 'SLC18A3']
PETgenes_recept = ['5HT1a', '5HT1b', '5HT2a', '5HT4', '5HT6', '5HTT',
                   'A4B2', 'A4B2', 'CB1', 'D1', 'D2', 'DAT', 'GABAa',
                   'GABAa', 'GABAa', 'H3', 'M1', 'mGluR5', 'MOR', 'NET',
                   'VAChT']

AUTgenes = ['GRIA1', 'GRIN1', 'GRIK2', 'GABRA1', 'GABRB2', 'GABRG2',
            'GABRA1', 'GABRG2', 'GABRB2', 'GABBR1', 'GABBR2', 'CHRM1',
            'CHRM2', 'CHRM3', 'CHRNA4', 'CHRNB2', 'ADRA1A', 'ADRA2A',
            'HTR1A', 'HTR2A', 'DRD1']
AUTgenes_recept = ['AMPA', 'NMDA', 'kainate', 'GABAa', 'GABAa', 'GABAa',
                   'GABAa/BZ', 'GABAa/BZ', 'GABAa/BZ', 'GABAb', 'GABAb',
                   'm1', 'm2', 'm3', 'a4b2', 'a4b2', 'a1', 'a2', '5-HT1a',
                   '5-HT2', 'D1']


"""
Supplementary Figure: alternative tracers
"""
# git clone https://github.com/netneurolab/hansen_receptors/

recept_path = '/home/jhansen/gitrepos/hansen_receptors/data/PET_parcellated/scale033/'
cumi101 =  np.genfromtxt(recept_path + '5HT1a_cumi_hc8_beliveau.csv', delimiter=',')
az104 =  np.genfromtxt(recept_path + '5HT1b_az_hc36_beliveau.csv', delimiter=',')
altanserin = np.genfromtxt(recept_path + '5HT2a_alt_hc19_savli.csv', delimiter=',')
fmpep =  np.genfromtxt(recept_path + 'CB1_FMPEPd2_hc22_laurikainen.csv', delimiter=',')
fallypride = np.genfromtxt(recept_path + 'D2_fallypride_hc49_jaworska.csv', delimiter=',')

alt_trac = [cumi101, az104, altanserin, fmpep, fallypride]
alt_trac_gene = ['HTR1A', 'HTR1B', 'HTR2A', 'CNR1', 'DRD2']
alt_trac_rec = ['5HT1A', '5HT1B', '5HT2A', 'CB1', 'D2']

plt.ion()
fig, axs = plt.subplots(1, 5, figsize=(15, 3))
axs = axs.ravel()
for i in range(len(alt_trac)):
    x = zscore(alt_trac[i])[leftcortex]
    try:  # necessary if ibf_threshold != 0 in abagen.get_expression_data()
        y = zscore(expression_ma[alt_trac_gene[i]])
    except KeyError:
        continue
    r, p, _ = corr_spin(x, y, spins, nspins, True)
    axs[i].scatter(x, y, s=5)
    axs[i].set_xlabel(alt_trac_rec[i] + ' density')
    axs[i].set_ylabel(alt_trac_gene[i] + ' expression')
    axs[i].set_title(['r=' + str(r)[:5] + ', p= ' + str(p)[:5]])
plt.tight_layout()
plt.savefig(path+'figures/scatter_alt_tracers.eps')


"""
Supplementary Figure: laminar layers
"""

layercorrs = dict([])
layercorrs['rho'] = np.zeros((3, len(AUTgenes)))
layercorrs['pspin'] = np.zeros((3, len(AUTgenes)))
layer_data = [receptdata_s, receptdata_g, receptdata_i]
for layer in range(len(layer_data)):
    for g in range(len(AUTgenes)):
        print(AUTgenes[g])
        x = layer_data[layer][:, receptor_names_a.index(AUTgenes_recept[g])]
        try:  # necessary if ibf_threshold != 0 in abagen.get_expression_data()
            y = zscore(expression_ma[AUTgenes[g]][:-1])
        except KeyError:
            continue
        layercorrs['rho'][layer, g], layercorrs['pspin'][layer, g], _ = corr_spin(x, y, spins, nspins, False)
np.savez(path+'results/layercorrs.npz',
         rho=layercorrs['rho'],
         pspin=layercorrs['pspin'])

fig, ax = plt.subplots(figsize=(8, 8))
ax.plot(layercorrs['rho'])
# ax.legend(AUTgenes)
ax.scatter(np.where(layercorrs['pspin'] < 0.05)[0],
           layercorrs['rho'][layercorrs['pspin'] < 0.05])
for g in range(len(AUTgenes)):
    ax.text(-0.4, layercorrs['rho'][0, g], AUTgenes[g])
ax.set_xticks(range(3))
ax.set_xticklabels(["supragranular", "granular", "infragranular"])
ax.set_xlim([-0.5, 2.1])
plt.tight_layout()
plt.savefig(path+'figures/plot_laminar_layers.eps')

"""
Supplementary Figure: probe selection method
"""

# probe_select = ["max_variance", "average", "corr_variance", "corr_intensity"]
# for probe in range(len(probe_select)):
#     exp = abagen.get_expression_data(cammoun['scale033'], ibf_threshold=0,
#                                      probe_selection=probe_select[probe], return_donors=True)
#     exp, ds = abagen.correct.keep_stable_genes(exp,
#                                                threshold=0,
#                                                percentile=True,
#                                                return_stability=True)
#     exp = pd.concat(exp).groupby('label').mean()
#     exp.to_csv(path+'data/expression/scale033_data_' + probe_select[probe] + '.csv')
#     ds = {'genes': exp.columns, 'stability': ds}
#     ds = pd.DataFrame(ds)
#     ds.to_csv(path+'data/expression/scale033_stability_' + probe_select[probe] + '.csv')

plt.ion()
fig, axs = plt.subplots(2, 4, figsize=(20, 5))
axs = axs.ravel()
probe_select = ["average", "max_intensity", "max_variance", "pc_loading",
                "corr_variance", "corr_intensity", "rnaseq"]
for probe in range(len(probe_select)):
    exp = pd.read_csv(path+'data/expression/scale033_data_' + probe_select[probe] + '.csv')
    ds = pd.read_csv(path+'data/expression/scale033_stability_' + probe_select[probe] + '.csv')
    exp_den_rho = np.zeros((len(PETgenes)))
    for rec in range(len(PETgenes)):
        exp_den_rho[rec] = spearmanr(PETrecept[34:, receptor_names_p.index(PETgenes_recept[rec])],
                                     zscore(exp.iloc[leftcortex][PETgenes[rec]]))[0]
    xarr = []
    yarr = []
    for i in range(len(exp_den_rho)):
        x = exp_den_rho[i]
        y = ds.query('genes == @PETgenes[@i]')['stability']
        axs[probe].scatter(x, y, c='b')
        axs[probe].text(x+0.01, y+0.01, PETgenes[i], fontsize=7)
        xarr.append(x)
        yarr.append(y)
    sns.regplot(x=np.array(xarr), y=np.array(yarr),
                scatter=False, ci=False, ax=axs[probe])
    r, p = spearmanr(xarr, yarr)
    print('rho = ' + str(r) + ', p = ' + str(p))
    axs[probe].set_xlabel('gene-receptor correlation')
    axs[probe].set_ylabel('differential stability')
    axs[probe].set_title(probe_select[probe])
    axs[probe].set_aspect(1.0/axs[probe].get_data_ratio(), adjustable='box')
plt.tight_layout()
plt.savefig(path+'figures/scatter_probe_selection.svg')

"""
Supplementary Figure: PET Scale033 RNAseq
"""

plt.ion()
fig, axs = plt.subplots(5, 5, figsize=(15, 12))
axs = axs.ravel()
for i in range(len(PETgenes)):
    x = PETrecept[34:, receptor_names_p.index(PETgenes_recept[i])]
    y = zscore(expression_rna[PETgenes[i]])
    row = PETcorrs_scale033_rnaseq.query('\
          genes == @PETgenes[@i] & receptors == @PETgenes_recept[@i]')
    r = np.squeeze(np.array(row['cortex-rho']))
    p = np.squeeze(np.array(row['cortex-pspin']))
    axs[i].scatter(x, y, s=5)
    axs[i].set_xlabel(PETgenes_recept[i] + ' density')
    axs[i].set_ylabel(PETgenes[i] + ' expression')
    axs[i].set_title(['r= ' + str(r)[:5] + ', p= ' + str(p)[:5]])
plt.tight_layout()
plt.savefig(path+'figures/scatter_pet_rnaseq.svg')

"""
Supplementary Figure: AUT RNAseq
"""

plt.ion()
fig, axs = plt.subplots(5, 5, figsize=(15, 12))
axs = axs.ravel()
for i in range(len(AUTgenes)):
    print(AUTgenes[i])
    x = autorad_data[:, receptor_names_a.index(AUTgenes_recept[i])]
    y = zscore(expression_rna[AUTgenes[i]][:-1])
    row = AUTcorrs_rnaseq.query('genes == @AUTgenes[@i] & receptors == @AUTgenes_recept[@i]')
    r = np.squeeze(np.array(row['rho']))
    p = np.squeeze(np.array(row['pspin']))
    axs[i].scatter(x, y, s=5)
    axs[i].set_xlabel(AUTgenes_recept[i] + ' density')
    axs[i].set_ylabel(AUTgenes[i] + ' expression')
    axs[i].set_title(['r=' + str(r)[:5] + ', p= ' + str(p)[:5]])
plt.tight_layout()
plt.savefig(path+'figures/scatter_aut_rnaseq.svg')

"""
Supplementary Figure: RNAseq vs microarray
"""

# compare the correlations
fig, axs = plt.subplots(1, 2, figsize=(10, 7))
axs[0].scatter(PETcorrs_scale033_microarray['cortex-rho'], PETcorrs_scale033_rnaseq['cortex-rho'])
axs[0].set_title('PET')
axs[1].scatter(AUTcorrs_microarray['rho'], AUTcorrs_rnaseq['rho'])
axs[1].set_title('autoradiography')
for i in range(2):
    axs[i].set_xlabel('microarray exp-den corr')
    axs[i].set_ylabel('rnaseq exp-den corr')
    axs[i].set_aspect(1.0/axs[0].get_data_ratio(), adjustable='box')
plt.savefig(path+'figures/scatter_ma_vs_rnaseq.eps')

"""
Supplementary Figure: GABAa subunits
"""

GABAAgenes = ['GABRR3', 'GABRD', 'GABRR2', 'GABRR1', 'GABRG1', 'GABRA2',
              'GABRA4', 'GABRB1', 'GABRG3', 'GABRA5', 'GABRB3', 'GABRP',
              'GABRA6','GABRE','GABRA3','GABRQ']

plt.ion()
fig, axs = plt.subplots(4, 5, figsize=(15, 10))
axs = axs.ravel()
x = PETrecept[34:, receptor_names_p.index('GABAa')]
for i in range(len(GABAAgenes)):
    y = zscore(expression_ma[GABAAgenes[i]])
    row = PETcorrs_scale033_microarray.query('\
          genes == @GABAAgenes[@i] & receptors == "GABAa"')
    r = np.squeeze(np.array(row['cortex-rho']))
    p = np.squeeze(np.array(row['cortex-pspin']))
    axs[i].scatter(x, y, s=5)
    axs[i].set_xlabel('GABAa density')
    axs[i].set_ylabel(GABAAgenes[i] + ' expression')
    axs[i].set_title(['r= ' + str(r)[:5] + ', p= ' + str(p)[:5]])
plt.tight_layout()
plt.savefig(path+'figures/scatter_gabaa.svg')

"""
Supplementary Figure: PET Scale 125
"""

plt.ion()
fig, axs = plt.subplots(5, 5, figsize=(15, 12))
axs = axs.ravel()
for i in range(len(PETgenes)):
    x = PETrecept125[-111:, receptor_names_p.index(PETgenes_recept[i])]
    y = zscore(expression125[PETgenes[i]])
    row = PETcorrs_scale125_microarray.query('\
          genes == @PETgenes[@i] & receptors == @PETgenes_recept[@i]')
    r = np.squeeze(np.array(row['cortex-rho']))
    p = np.squeeze(np.array(row['cortex-pspin']))
    axs[i].scatter(x, y, s=5)
    axs[i].set_xlabel(PETgenes_recept[i] + ' density')
    axs[i].set_ylabel(PETgenes[i] + ' expression')
    axs[i].set_title(['r= ' + str(r)[:5] + ', p= ' + str(p)[:5]])
plt.tight_layout()
plt.savefig(path+'figures/scatter_pet_scale125.svg')

"""
Supplementary Figure: PET vs autoradiography
"""

# overlap: 5-ht1a, 5-ht2, a4b2, D1, GABAa, M1
pet_idx = [0, 2, 6, 8, 11, 13]
aut_idx = [12, 13, 9, 14, 3, 6]

fig, axs = plt.subplots(1, len(pet_idx), figsize=(17, 3))
axs = axs.ravel()
fig2, axs2 = plt.subplots()  # compare DS with pet-aut corr

PETAUTcorr = {'rho' : np.zeros((len(pet_idx), )),
              'pspin' : np.zeros((len(pet_idx), )),
              'null' : np.zeros((len(pet_idx), nspins))}
for i in range(len(pet_idx)):
    print(i)
    axs[i].scatter(PETrecept[34:-1, pet_idx[i]], autorad_data[:, aut_idx[i]])
    axs[i].set_xlabel(receptor_names_p[pet_idx[i]] + ' PET density')
    axs[i].set_ylabel(receptor_names_a[aut_idx[i]] + ' autorad density')
    PETAUTcorr['rho'][i], PETAUTcorr['pspin'][i], PETAUTcorr['null'][i, :] = \
        corr_spin(PETrecept[34:-1, pet_idx[i]], autorad_data[:, aut_idx[i]], spins, nspins, False)
    print(receptor_names_p[pet_idx[i]] + ': r = ' +
          str(PETAUTcorr['rho'][i])[:4] + ', pspin = ' +
          str(PETAUTcorr['pspin'][i])[:5])
    axs[i].set_title('r = ' + str(PETAUTcorr['rho'][i])[:4])
    axs[i].set_aspect(1.0/axs[i].get_data_ratio(), adjustable='box')

    g = PETgenes[PETgenes_recept.index(receptor_names_p[pet_idx[i]])]
    x = ds.query('genes == @g')['stability']
    axs2.scatter(x, PETAUTcorr['rho'][i], c='b')
    axs2.text(x+0.01, PETAUTcorr['rho'][i]+0.01, receptor_names_p[pet_idx[i]], fontsize=7)
    axs2.set_aspect(1.0/axs2.get_data_ratio(), adjustable='box')
axs2.set_xlabel('differential stability')
axs2.set_ylabel('PET-autorad correlation')
plt.tight_layout()
plt.savefig(path+'figures/scatter_petvsaut_density.eps')

"""
Supplementary Figures: SC neighbourhood analysis
"""

sc_bin = np.load('/home/jhansen/projects/hansen_crossdisorder_vulnerability/data/hcp/sc_binary.npy')
sc_wei = np.load('/home/jhansen/projects/hansen_crossdisorder_vulnerability/data/hcp/sc_weighted.npy')
sc = sc_wei[34:, 34:]
# correlate density with mean mRNA of connected brain regions

PETcorr_nn = {'rho' : np.zeros((len(PETgenes), )),
              'pspin' : np.zeros((len(PETgenes), )),
              'null' : np.zeros((len(PETgenes), nspins))}

plt.ion()
fig, axs = plt.subplots(5, 5, figsize=(15, 12))
axs = axs.ravel()
for i in range(len(PETgenes)):
    print(i)
    x = PETrecept[34:, receptor_names_p.index(PETgenes_recept[i])]
    try:  # necessary if ibf_threshold != 0 in abagen.get_expression_data()
        g = zscore(expression[PETgenes[i]])
        y = np.zeros((len(x), ))
        for j in range(len(y)):
            y[j] = np.mean(g[sc[j, :] != 0]
                           * sc[sc[j, :] != 0, j])
    except KeyError:
        continue
    PETcorr_nn['rho'][i], PETcorr_nn['pspin'][i], PETcorr_nn['null'][i, :] \
        = corr_spin(x, y, spins, nspins)
    axs[i].scatter(x, y, s=5)
    axs[i].set_xlabel(PETgenes_recept[i] + ' density')
    axs[i].set_ylabel(PETgenes[i] + ' neighbour expression')
    axs[i].set_title(['r=' + str(PETcorr_nn['rho'][i])[:5] + ', p= ' + str(PETcorr_nn['pspin'][i])[:5]])
plt.tight_layout()
plt.savefig(path+'figures/scatter_nncorr_pet_eachrecept.eps')

AUTcorr_nn = {'rho' : np.zeros((len(AUTgenes), )),
              'pspin' : np.zeros((len(AUTgenes), )),
              'null' : np.zeros((len(AUTgenes), nspins))}

plt.ion()
fig, axs = plt.subplots(5, 5, figsize=(15, 12))
axs = axs.ravel()
for i in range(len(AUTgenes)):
    print(i)
    x = autorad_data[:, receptor_names_a.index(AUTgenes_recept[i])]
    try:  # necessary if ibf_threshold != 0 in abagen.get_expression_data()
        g = zscore(expression[AUTgenes[i]][:-1])
        y = np.zeros((len(x), ))
        for j in range(len(y)):
            y[j] = np.mean(g[sc[j, :-1] != 0]
                           * sc[:-1, :][sc[j, :-1] != 0, j])
    except KeyError:
        continue
    AUTcorr_nn['rho'][i], AUTcorr_nn['pspin'][i], AUTcorr_nn['null'][i, :] \
        = corr_spin(x, y, spins, nspins)
    axs[i].scatter(x, y, s=5)
    axs[i].set_xlabel(AUTgenes_recept[i] + ' density')
    axs[i].set_ylabel(AUTgenes[i] + ' expression')
    axs[i].set_title(['r=' + str(AUTcorr_nn['rho'][i])[:5] + ', p= ' + str(AUTcorr_nn['pspin'][i])[:5]])
plt.tight_layout()
plt.savefig(path+'figures/scatter_nncorr_aut_eachrecept.eps')

fig, ax = plt.subplots()
xarr = []
for i in range(len(PETgenes)):
    x = np.squeeze(np.array(PETcorrs_scale033_microarray.query('\
        genes == @PETgenes[@i] & receptors == @PETgenes_recept[@i]')['cortex-rho']))
    y = PETcorr_nn['rho'][i]
    xarr.append(x)
    ax.scatter(x, y, c='b')
    ax.text(x+0.01, y+0.01, PETgenes[i])
r, p = spearmanr(xarr, PETcorr_nn['rho'])
ax.set_xlabel('exp-den corr')
ax.set_ylabel('neighbour exp-den corr')
ax.set_title(['r=' + str(r)[:5] + ', p=' + str(p)[:6]])
plt.tight_layout()
plt.savefig(path+'figures/scatter_nncorr_pet.eps')

fig, ax = plt.subplots()
xarr = []
for i in range(len(AUTgenes)):
    x = np.squeeze(np.array(AUTcorrs_microarray.query('\
        genes == @AUTgenes[@i] & receptors == @AUTgenes_recept[@i]')['rho']))
    y = AUTcorr_nn['rho'][i]
    xarr.append(x)
    ax.scatter(x, y, c='b')
    ax.text(x+0.01, y+0.01, AUTgenes[i])
r, p = spearmanr(xarr, AUTcorr_nn['rho'])
ax.set_xlabel('exp-den corr')
ax.set_ylabel('neighbour exp-den corr')
ax.set_title(['r=' + str(r)[:5] + ', p=' + str(p)[:6]])
plt.tight_layout()
plt.savefig(path+'figures/scatter_nncorr_aut.eps')

"""
Supplementary Figure : RSN & VE network classification
"""

rsn_mapping = info.query('scale == "scale125" \
                         & structure == "cortex" \
                         & hemisphere == "L"')['yeo_7']
rsn_labels, rsn_mapping = np.unique(rsn_mapping, return_inverse=True)
rsn_idx = np.array([0, 2, 3, 5, 1, 4, 6])  # from dmn --> vis

ve_mapping = np.genfromtxt('/home/jhansen/data/ve_mapping_scale125.csv',
                           delimiter=',', dtype=int)[-111:] - 1
ve_labels = np.array(pd.read_csv('/home/jhansen/data/ve_mapping_names.csv')['labels'])
ve_idx = np.array([6, 5, 2, 1, 3, 4, 0])  # from insular --> motor

# get expression-density correlation within system
system, system_idx = ve_mapping, ve_idx  # or ve_mapping, ve_idx
system_corr = np.zeros((np.max(system)+1, len(PETgenes)))
for sys in range(system_corr.shape[0]):
    for gene in range(len(PETgenes)):
        x = PETrecept125[-111:, receptor_names_p.index(PETgenes_recept[gene])][system==sys]
        y = zscore(expression125[PETgenes[gene]])[system==sys]
        system_corr[sys, gene], _  = spearmanr(x, y)

# get confidence interval for correlations
boots = np.zeros((len(PETgenes), 2))
for gene in range(len(PETgenes)):
    x = PETrecept125[-111:, receptor_names_p.index(PETgenes_recept[gene])]
    y = zscore(expression125[PETgenes[gene]])
    boots[gene, :] = get_boot_ci(x, y, nboot=1000)

# plot for each receptor
plt.ion()
plt.figure(figsize=(12, 6))
cmap = plt.get_cmap('BuPu', np.max(system)+3)
for i in range(system_corr.shape[0]):
    lw = np.array([boots[j, 0] <= system_corr[i, j] <= boots[j, 1] for j in range(boots.shape[0])])
    lw = 2*(1 - lw.astype(int))
    plt.scatter(np.arange(len(PETgenes)), system_corr[i, :], linewidths=lw, color=cmap(system_idx[i]+2))
plt.xticks(range(len(PETgenes)), PETgenes, rotation=90)
plt.tight_layout()
plt.savefig(path+'figures/scatter_hierarchy_ve.eps')  # or _ve


"""
Supplementary Figure: whole-brain
"""

PETrecept_wb = np.concatenate((PETrecept[34:, :], PETrecept_subc), axis=0)
PETcorr_wb = {'rho' : np.zeros((len(PETgenes), )),
              'pspin' : np.zeros((len(PETgenes), ))}

fig, axs = plt.subplots(5, 5, figsize=(15, 12))
axs = axs.ravel()
for i in range(len(PETgenes)):
    x = PETrecept_wb[:, receptor_names_p.index(PETgenes_recept[i])]
    try:
        y1 = zscore(expression_ma[PETgenes[i]])
        y2 = zscore(expression_subc[PETgenes[i]])
        y = np.concatenate((y1, y2), axis=0)
    except KeyError:
        continue
    PETcorr_wb['rho'][i], PETcorr_wb['pspin'][i] = spearmanr(x, y)
    axs[i].scatter(x, y, s=5)
    axs[i].set_xlabel(PETgenes_recept[i] + ' density')
    axs[i].set_ylabel(PETgenes[i] + ' expression')
    axs[i].set_title('r=' + str(PETcorr_wb['rho'][i])[:5])
plt.tight_layout()
plt.savefig(path+'figures/scatter_pet_wb.svg')

fig, ax = plt.subplots()
xarr = []
for i in range(len(PETgenes)):
    x = np.squeeze(np.array(PETcorrs_scale033_microarray.query('\
        genes == @PETgenes[@i] & receptors == @PETgenes_recept[@i]')['cortex-rho']))
    y = PETcorr_wb['rho'][i]
    xarr.append(x)
    ax.scatter(x, y, c='b')
    ax.text(x+0.01, y+0.01, PETgenes[i])
ax.plot([-0.2, 0.82], [-0.2, 0.82])
ax.set_xlabel('cortex-only exp-den corr')
ax.set_ylabel('whole brain exp-den corr')
r, p = spearmanr(xarr, PETcorr_wb['rho'])
ax.set_title(['r=' + str(r)[:5] + ', p=' + str(p)[:6]])
plt.tight_layout()
plt.savefig(path+'figures/scatter_cortex_v_wholebrain.eps')

"""
Supplementary Figure : subcortical differential stability
"""

fig, ax = plt.subplots()
xarr = []
yarr = []
for i in range(len(PETgenes)):
    x = np.squeeze(np.array(PETcorrs_scale033_microarray.query('\
        genes == @PETgenes[@i] & receptors == @PETgenes_recept[@i]')['subcortex-rho']))
    y = ds.query('genes == @PETgenes[@i]')['stability']
    ax.scatter(x, y, c='b')
    ax.text(x+0.01, y+0.01, PETgenes[i], fontsize=7)
    xarr.append(x)
    yarr.append(y)
sns.regplot(x=np.array(xarr), y=np.array(yarr),
            scatter=False, ci=False, ax=ax)
r, p = spearmanr(xarr, yarr)
ax.set_title(['r = ' + str(r)[:5] + ', p=' + str(p)[:6]])
ax.set_xlabel('gene-receptor correlation')
ax.set_ylabel('differential stability')
ax.set_aspect(1.0/ax.get_data_ratio(), adjustable='box')
plt.savefig(path+'figures/scatter_ds_subc.svg')

"""
Panther protein pathways
"""

# download gene list (.txt) for only Gene ID and Gene Name/Symbol for "<neurotransmitter> receptor"
panthergenelists = ["serotonin", "acetylcholine", "cannabinoid", "dopamine",
                    "gaba", "histamine", "glutamate", "opioid", "noradrenaline"]
pantherreceptors =[['5HT1a', '5HT1b', '5HT2a', '5HT4', '5HT6', '5HTT'],
                   ['A4B2', 'M1', 'VAChT'],
                   ['CB1'],
                   ['D1', 'D2', 'DAT'],
                   ['GABAa'],
                   ['H3'],
                   ['mGluR5'],
                   ['MOR'],
                   ['NET']]

pantherinfo = {'receptor' : [],
               '# AHBA genes' : np.zeros((len(receptor_names_p), )),
               '# significant' : np.zeros((len(receptor_names_p), )),
               'gene names' : []}

counter = 0
for searchterm in range(len(panthergenelists)):
    panther = pd.read_csv(path + 'data/panther_ontologies/panthergenelist_keyword-search_' +
                          panthergenelists[searchterm] + '.txt', sep='\t', lineterminator='\n', header=None)
    genelist = [panther.iloc[j].loc[1].split(';')[1] for j in range(panther.shape[0])]
    for r in pantherreceptors[searchterm]:
        print(r)
        # pantherinfo['receptor'].append(r)
        # pval = []
        rho = []
        x = PETrecept[34:, receptor_names_p.index(r)]  # protein density
        ahbagenelist = []
        for g in genelist:
            try:
                y = zscore(expression_ma[g])
            except KeyError:
                continue
            # pval.append(corr_spin(x, y, spins, nspins, False)[1])
            rho.append(spearmanr(x, y)[0])
            ahbagenelist.append(g)
        pval = multipletests(pvals=pval, method='fdr_bh')[1]
        # pantherinfo['# AHBA genes'][counter] = len(ahbagenelist)
        # pantherinfo['# significant'][counter] = np.sum(pval < 0.05)
        # pantherinfo['gene names'].append(np.array(ahbagenelist)[np.where(pval < 0.05)[0]])
        pantherinfo['# r>0.6'][counter] = np.sum(np.array(rho) > 0.6)
        counter += 1
        
pd.DataFrame.from_dict(pantherinfo).to_csv(path+'results/panther.csv')