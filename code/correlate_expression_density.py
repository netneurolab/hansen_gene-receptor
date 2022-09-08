"""
this script runs:
1) scale033 PET microarray expression-density correlations, with subcortex
2) scale033 autoradiography microarray expression-density correlatoins
3) scale125 PET microarray expression-density correlations
4) scale033 PET RNAseq expression-density correlations
5) scale033 autoradiography RNAseq expression-density correlations
"""

import numpy as np
import pandas as pd
from scipy.stats import zscore, spearmanr
from statsmodels.stats.multitest import multipletests
from netneurotools import datasets, utils, stats


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


def get_perm_p(emp, null, twotailed=True):
    if twotailed:
        return (1 + sum(abs(null - np.nanmean(null))
                        > abs(emp - np.nanmean(null)))) / (len(null) + 1)
    else:
        return (1 + sum(null > emp)) / (len(null) + 1)



"""
set-up
"""

path = 'C:/Users/justi/OneDrive - McGill University/MisicLab/proj_genes_receptors/\
github/hansen_gene-receptor/'

# PET receptors
PETrecept = np.genfromtxt(path+'data/PET_receptors.csv', delimiter=',')
PETrecept = zscore(PETrecept)
PETrecept125 = np.genfromtxt(path+'data/PET_receptors_scale125.csv', delimiter=',')
PETrecept125 = zscore(PETrecept125)
PETrecept_subc = np.genfromtxt(path+'data/PET_receptors_subcortex.csv', delimiter=',')
PETrecept_subc = zscore(PETrecept_subc)
receptor_names_p = ["5HT1a", "5HT1b", "5HT2a", "5HT4", "5HT6", "5HTT", "A4B2",
                  "CB1", "D1", "D2", "DAT", "GABAa", "H3", "M1", "mGluR5",
                  "MOR", "NET", "VAChT"]

# autoradiography receptors
receptor_names_a = np.load(path+'data/autoradiography/ReceptorNames.npy').tolist()
autorad_data = np.load(path+'results/autorad_data.npy')

# get scale033 spins
cammoun = datasets.fetch_cammoun2012()
info = pd.read_csv(cammoun['info'])
leftcortex = info.query('scale == "scale033" \
                    & structure == "cortex" \
                    & hemisphere == "L"')['id']
leftcortex = np.array(leftcortex) - 1  # python indexing
subcortex = info.query('scale == @scale & structure == "subcortex"')['id']
subcortex = np.array(subcortex) - 1  # python indexing
coords = utils.get_centroids(cammoun['scale033'], image_space=True)
coords = coords[leftcortex, :]
nspins = 10000
spins033 = stats.gen_spinsamples(coords, hemiid=np.ones((len(leftcortex),)),
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
expression_ma = pd.read_csv(path+'data/expression/scale033_data.csv')  # microarray
expression125 = pd.read_csv(path+'data/expression/scale125_data.csv')
expression_subc = pd.read_csv(path+'data/expression/scale033_data_subcortex.csv')

expression_rna = pd.read_csv(path+'data/expression/rnaseq_data/scale033_data_rnaseq.csv')

"""
Supplemental tables
"""

gexp_type = ['microarray', 'rnaseq']
scale = ['scale033', 'scale125']
expression = [[expression_ma, expression125], [expression_rna, None]]
recept = [PETrecept, PETrecept125]
spins = [spins033, spins125]

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
                       + ['H3', 'M1', 'mGluR5', 'MOR', 'NET', 'VAChT']

for gexp_i in range(2):  # for microarray and RNAseq gene expression

    for scale_i in range(2):  # for scale033 and scale125

        exp = expression[gexp_i][scale_i]
        rec = recept[scale_i]
        sp = spins[scale_i]

        if scale[scale_i] == 'scale125' & gexp_type[gexp_i] == 'rnaseq':
            continue  # do not run scale125 with RNAseq
        
        petinfo['cortex-rho'] = np.zeros((len(petinfo['genes']), ))
        petinfo['cortex-pspin'] = np.zeros((len(petinfo['genes']), ))

        if gexp_type[gexp_i] == 'microarray' and scale[scale_i] == 'scale033':
            petinfo['subcortex-rho'] = np.zeros((len(petinfo['genes']), ))
            petinfo['subcortex-pval'] = np.zeros((len(petinfo['genes']), ))

        print("Running: " + scale[scale_i] + ", " + gexp_type[gexp_i] + " gene expression")

        for i in range(len(petinfo['genes'])):  # for each gene-receptor pair

            print(petinfo['genes'][i])

            if scale[scale_i] == 'scale033':
                x = PETrecept[34:, receptor_names_p.index(petinfo['receptors'][i])]
            elif scale[scale_i] == 'scale125':
                x = PETrecept125[-111:, receptor_names_p.index(petinfo['receptors'][i])]

            try:  # necessary if ibf_threshold != 0 in abagen.get_expression_data()
                y = zscore(exp[petinfo['genes'][i]])
            except KeyError:
                continue
            
            # one-tailed spin test spearman correlation
            petinfo['cortex-rho'][i], petinfo['cortex-pspin'][i], _ = corr_spin(x, y, sp, nspins, False)

            if gexp_type[gexp_i] == 'microarray' and scale[scale_i] == 'scale033':
                petinfo['subcortex-rho'][i], petinfo['subcortex-pval'][i] = \
                    spearmanr(PETrecept_subc[:, receptor_names_p.index(petinfo['receptors'][i])],
                              zscore(expression_subc[petinfo['genes'][i]]), alternative='greater')

        # run multiple comparisons correction
        needs_correct = ['GABAa', 'A4B2']
        for i in range(len(needs_correct)):
            index = [index for index, value in enumerate(petinfo['receptors']) if value == needs_correct[i]]
            uncorrected_pval = petinfo['cortex-pspin'][index]
            petinfo['cortex-pspin'][index] = multipletests(pvals=uncorrected_pval, method='fdr_bh')[1]

            if gexp_type[gexp_i] == 'microarray' and scale[scale_i] == 'scale033':
                uncorrected_pval = petinfo['subcortex-pval'][index]
                petinfo['subcortex-pval'][index]= multipletests(pvals=uncorrected_pval, method='fdr_bh')[1]

        # save out
        pd.DataFrame.from_dict(petinfo).to_csv(path+'results/PETcorrs_' + scale[scale_i] + '_' + gexp_type[gexp_i] + '.csv')
        
        # reset
        del petinfo['cortex-rho'], petinfo['cortex-pspin']
        if gexp_type[gexp_i] == 'microarray' and scale[scale_i] == 'scale033':
            del petinfo['subcortex-rho'], petinfo['subcortex-pval']



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

for gexp_i in range(2):  # for microarray and RNAseq gene expression

    exp = expression[gexp_i][0]

    autinfo['rho'] = np.zeros((len(autinfo['genes']), ))
    autinfo['pspin'] = np.zeros((len(autinfo['genes']), ))

    for i in range(len(autinfo['genes'])):
        print(autinfo['genes'][i])
        x = autorad_data[:, receptor_names_a.index(autinfo['receptors'][i])]
        try:  # necessary if ibf_threshold != 0 in abagen.get_expression_data()
            y = zscore(exp[autinfo['genes'][i]][:-1])
        except KeyError:
            continue
        autinfo['rho'][i], autinfo['pspin'][i], _ = corr_spin(x, y, spins033, nspins, False)

    needs_correct = ['AMPA', 'NMDA', 'kainate', 'GABAa', 'GABAa/BZ', 'GABAb', 'a4b2']
    for i in range(len(needs_correct)):
        index = [index for index, value in enumerate(autinfo['receptors']) if value == needs_correct[i]]
        uncorrected_pval = autinfo['pspin'][index]
        autinfo['pspin'][index] = multipletests(pvals=uncorrected_pval, method='fdr_bh')[1]

    pd.DataFrame.from_dict(autinfo).to_csv(path+'results/AUTcorrs_' + gexp_type[gexp_i] + '.csv')
    del autinfo['rho'], autinfo['pspin']
