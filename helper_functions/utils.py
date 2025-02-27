import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.stats.multitest as smm

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns

import os
import sys
import re

set2_cols = sns.color_palette("Set2")

def get_clone_table(cls_df, cls_md, thr_state, conditions):
    clones = []
    for i in cls_df.sampleID.unique():
        state = cls_md.loc[i, thr_state]
        vv_df = cls_df[cls_df.sampleID==i].copy()
        clones.append([i,
                    state,
                    vv_df[vv_df.Clonality == 'clonal [early]'].shape[0],
                    vv_df[vv_df.Clonality == 'clonal [NA]'].shape[0],
                    vv_df[vv_df.Clonality == 'clonal [late]'].shape[0],
                    vv_df[vv_df.Clonality == 'subclonal'].shape[0]])

    clones_df = pd.DataFrame(clones, columns=['sampleID', 'State', 'clonal [early]',
                                            'clonal [NA]', 'clonal [late]', 'subclonal'])
    clones_df = clones_df.fillna(0)

    clones_df.columns = ['sampleID','State','Early Clonal','Clonal','Late Clonal','Subclonal']
    clones_df = clones_df.set_index('sampleID')

    clones_df_sum = clones_df.groupby('State').sum()
    clones_df_sum = clones_df_sum.loc[[x for x in clones_df_sum.index if x in conditions]]
    clones_df_sum = clones_df_sum.div(clones_df_sum.sum(axis=1), axis=0)
    return clones_df_sum


def get_type_numbers(plat_df):
    plat_df.Tumor = [x.replace('*', "") for x in plat_df.Tumor]
    plat_df.Tumor = [x.replace('**', "") for x in plat_df.Tumor]

    type_sig = pd.concat([plat_df[plat_df.Platin_sig=='N'].Tumor.value_counts(),
                          plat_df[plat_df.Platin_sig=='Y'].Tumor.value_counts()], axis=1)
    type_sig.columns = ['Sig-', 'Sig+']
    type_sig = type_sig.fillna(0)
    type_sig['Tumor'] = type_sig.index.tolist()

    type_sig_mlt = pd.melt(type_sig, id_vars=['Tumor'], value_vars=['Sig-', 'Sig+'])
    type_sig_mlt.columns = ['Tumor', 'Sig', 'Count']
    type_sig_mlt['Color'] = [set2_cols[2] if x=='Sig-' else set2_cols[3] for x in type_sig_mlt.Sig]
    type_sig_mlt.Count = type_sig_mlt.Count.astype(int)
    return type_sig_mlt


def sigs_melt(sigs_df):
    sigs_df['samples'] = sigs_df.index.tolist()
    sigs_df_mlt = pd.melt(sigs_df, id_vars=['samples'], value_vars=sigs_df.columns.tolist())
    sigs_df_mlt.columns = ['Samples', 'Signatures', 'Mutations']
    return sigs_df_mlt


def get_drivers_table(drivers_df, md_df, sample_col, burden=None):
    '''
    Convert the drivers df into a table to be plotted as barchart.
    1. Load drivers df
    KZM611_drivers_all = pd.read_csv(os.path.join(root_dir, 'source_data/KZM611_drivers_CosCGI.tsv'), sep='\t')
    2. Run this function
    drivers_counts_low = get_drivers_table(KZM611_drivers_all, kzm611_md, burden='low')

    Result:
    	            Zero	    One	        Two+
    All	            0.455806	0.292894	0.251300
    Primary-Naive	0.484848	0.268398	0.246753
    Primary-Treated	0.544444	0.277778	0.177778
    Advanced-Naive	0.666667	0.166667	0.166667
    Advanced-Treated	0.385246	0.327869	0.286885
    '''
    if burden == 'low':
        md_df = md_df[md_df.Burden==burden].copy()
        drivers_df = drivers_df[drivers_df[sample_col].isin(md_df.index.tolist())]
    n_samples = md_df.shape[0]

    zero_dri = [x for x in md_df.index if x not in drivers_df[sample_col].unique().tolist()]
    one_dri = drivers_df[sample_col].value_counts()[drivers_df[sample_col].value_counts()==1].index.tolist()
    two_dri = drivers_df[sample_col].value_counts()[drivers_df[sample_col].value_counts()>1].index.tolist()
    allist = [len(zero_dri)/n_samples, len(one_dri)/n_samples, len(two_dri)/n_samples]

    zd = md_df.loc[zero_dri].Thr_State.value_counts()/md_df.Thr_State.value_counts()
    od = md_df.loc[one_dri].Thr_State.value_counts()/md_df.Thr_State.value_counts()
    td = md_df.loc[two_dri].Thr_State.value_counts()/md_df.Thr_State.value_counts()

    ddf = pd.concat([zd, od, td], axis=1)
    ddf.columns = ['Zero', 'One', 'Two+']
    ddf = pd.concat([ddf, pd.Series(allist, index=['Zero', 'One', 'Two+']).to_frame().T], axis=0)
    ddf = ddf.rename(index={0: 'All'})
    return ddf.loc[['All', 'Primary-Naive', 'Primary-Treated', 'Advanced-Naive', 'Advanced-Treated']]


def get_drivers_mat(drivers, md_df, sample_col, burden=None):
    '''
    Convert the drivers df into a table to be plotted as a heatmap.
    1. Load drivers df
    KZM611_drivers_all = pd.read_csv(os.path.join(root_dir, 'source_data/KZM611_drivers_CosCGI.tsv'), sep='\t')
    2. Run this function
    low_genes_df = get_driver_mat(KZM611_drivers_all, kzm611_md, burden='low')

    Result:
    	Primary-Naive	Primary-Treated	Advanced-Naive	Advanced-Treated	Total	Percentage
    TP53	28	8	0	27	63	11.0
    KMT2C	9	5	0	18	32	6.0
    H3F3A	20	0	0	1	21	4.0
    '''
    low_samples = md_df[md_df.Burden=='low'].index.tolist()
    if burden == 'low':
        drivers = drivers[drivers[sample_col].isin(low_samples)]
    
    print(drivers.shape)

    all_genes = drivers.Hugo_Symbol.unique().tolist()
    genes_mat = np.zeros([len(all_genes), 4], dtype=int)

    for i in range(len(all_genes)):
        g = all_genes[i]
        genes_mat[i, 0] = drivers.loc[(drivers[sample_col].isin(md_df[md_df.Thr_State=='Primary-Naive'].index.tolist())) & 
                                             (drivers.Hugo_Symbol == g)].shape[0]
        genes_mat[i, 1] = drivers.loc[(drivers[sample_col].isin(md_df[md_df.Thr_State=='Primary-Treated'].index.tolist())) & 
                                             (drivers.Hugo_Symbol == g)].shape[0]
        genes_mat[i, 2] = drivers.loc[(drivers[sample_col].isin(md_df[md_df.Thr_State=='Advanced-Naive'].index.tolist())) & 
                                             (drivers.Hugo_Symbol == g)].shape[0]
        genes_mat[i, 3] = drivers.loc[(drivers[sample_col].isin(md_df[md_df.Thr_State=='Advanced-Treated'].index.tolist())) & 
                                             (drivers.Hugo_Symbol == g)].shape[0]

    genes_df = pd.DataFrame(genes_mat)
    genes_df.index = all_genes
    genes_df.columns = ['Primary-Naive', 'Primary-Treated', 'Advanced-Naive', 'Advanced-Treated']
    genes_df['Total'] = genes_df.sum(axis=1)
    if burden == 'low':
        genes_df['Percentage'] = genes_df.Total / len(low_samples)
    else:
        genes_df['Percentage'] = genes_df.Total / len(md_df)
    genes_df['Percentage'] = genes_df['Percentage'].apply(lambda x: round(x, 2)) * 100
    genes_df = genes_df.sort_values('Total', ascending=False)
    return genes_df


def get_drivers_type_state(dri_df, md_df, burden=None):
    dri_df['Type'] = [md_df.loc[x, 'Type'] for x in dri_df.Tumor_Sample_Barcode]

    if burden == 'low':
        md_df = md_df[md_df.Burden==burden].copy()
        dri_df = dri_df[dri_df.Tumor_Sample_Barcode.isin(md_df.index.tolist())]
    
    conditions = ['All', 'Primary-Naive', 'Primary-Treated', 'Advanced-Naive', 'Advanced-Treated']
    types = md_df.Type.value_counts()[md_df.Type.value_counts()>=10].index.tolist()

    rel_dict = {}
    abs_dict = {}
    rel_df = pd.DataFrame()
    abs_df = pd.DataFrame()

    for catype in types:
        drivers_df = dri_df[dri_df.Type==catype]
        sub_md = md_df[md_df.Type==catype]

        zero_dri = [x for x in sub_md.index if x not in drivers_df.Tumor_Sample_Barcode.unique().tolist()]
        one_dri = drivers_df.Tumor_Sample_Barcode.value_counts()[drivers_df.Tumor_Sample_Barcode.value_counts()==1].index.tolist()
        two_dri = drivers_df.Tumor_Sample_Barcode.value_counts()[drivers_df.Tumor_Sample_Barcode.value_counts()>1].index.tolist()

        zero_rel = []
        ones_rel = []
        twop_rel = []
        zero_abs = []
        ones_abs = []
        twop_abs = []

        zero_rel.append(len(zero_dri) / sub_md.shape[0])
        ones_rel.append(len(one_dri) / sub_md.shape[0])
        twop_rel.append(len(two_dri) / sub_md.shape[0])

        zero_abs.append(len(zero_dri))
        ones_abs.append(len(one_dri))
        twop_abs.append(len(two_dri))

        for cond in conditions[1:]:
            zero_abs.append(sub_md.loc[zero_dri].Thr_State.tolist().count(cond))
            ones_abs.append(sub_md.loc[one_dri].Thr_State.tolist().count(cond))
            twop_abs.append(sub_md.loc[two_dri].Thr_State.tolist().count(cond))

            if sub_md.Thr_State.tolist().count(cond) == 0:
                zero_rel.append(0)
                ones_rel.append(0)
                twop_rel.append(0)
            else:
                zero_rel.append(sub_md.loc[zero_dri].Thr_State.tolist().count(cond) / sub_md.Thr_State.tolist().count(cond))
                ones_rel.append(sub_md.loc[one_dri].Thr_State.tolist().count(cond) / sub_md.Thr_State.tolist().count(cond))
                twop_rel.append(sub_md.loc[two_dri].Thr_State.tolist().count(cond) / sub_md.Thr_State.tolist().count(cond))

        rel_counts = pd.DataFrame({'Zero': zero_rel, 'One': ones_rel, 'Two+': twop_rel}, index=conditions)
        abs_counts = pd.DataFrame({'Zero': zero_abs, 'One': ones_abs, 'Two+': twop_abs}, index=conditions)

        rel_counts['Type'] = catype
        abs_counts['Type'] = catype

        rel_df = pd.concat([rel_df, rel_counts], axis=0)
        abs_df = pd.concat([abs_df, abs_counts], axis=0)

        rel_dict[catype] = rel_counts
        abs_dict[catype] = abs_counts

    rel_df = rel_df.rename_axis('Condition').reset_index()
    abs_df = abs_df.rename_axis('Condition').reset_index()
    return rel_df, abs_df


def enrich_therapy_generic(sigs_df, md_df, min_val=0, pval=0.05, verbose=False, outfile=None, plot=True, test_type='fisher', althyp='two-sided'):
    sigs_md = sigs_df.copy()
    sigs = sigs_md.columns.tolist()
    therapies = md_df.columns.tolist()

    sigs_md = pd.concat([sigs_md, md_df], axis=1)
    sigs_md = sigs_md.dropna()

    sigs_md_enrich_df = pd.DataFrame(np.zeros((len(therapies), len(sigs))))
    sigs_md_enrich_df.columns = sigs
    sigs_md_enrich_df.index = therapies
    sigs_md_enrich_pv = sigs_md_enrich_df.copy()
    sigs_md_enrich_pv = sigs_md_enrich_pv + 1

    for s in sigs:
        sig_print = 1

        for t in therapies:
            '''if t == 'Therapy':
                kzm_Pos = sigs_md[sigs_md[t]=='Post-Therapy']
                kzm_Neg = sigs_md[sigs_md[t]=='Pre-Therapy']
            else:'''
            kzm_Pos = sigs_md[sigs_md[t]=='Y']
            kzm_Neg = sigs_md[sigs_md[t]=='N']

            '''if s == 'SBS17b':
                print(f"{t}: {kzm_Neg.shape[0]}, {kzm_Pos.shape[0]}")
                print(f"{len(kzm_Pos[kzm_Pos[s]>min_val])} -- {len(kzm_Pos[kzm_Pos[s]<=min_val])}")
                print(f"{len(kzm_Neg[kzm_Neg[s]>min_val])} -- {len(kzm_Neg[kzm_Neg[s]<=min_val])}")'''
            
            if kzm_Pos.shape[0]<10 or kzm_Neg.shape[0]<10:
                continue

            if test_type == 'fisher':
                odr, pv = stats.fisher_exact([[len(kzm_Pos[kzm_Pos[s]>min_val]), len(kzm_Pos[kzm_Pos[s]<=min_val])],
                                            [len(kzm_Neg[kzm_Neg[s]>min_val]), len(kzm_Neg[kzm_Neg[s]<=min_val])]],
                                            alternative=althyp)
            elif test_type == 'barnard':
                odr, pv = stats.barnard_exact([[len(kzm_Pos[kzm_Pos[s]>min_val]), len(kzm_Pos[kzm_Pos[s]<=min_val])],
                                            [len(kzm_Neg[kzm_Neg[s]>min_val]), len(kzm_Neg[kzm_Neg[s]<=min_val])]],
                                            alternative=althyp)
            elif test_type == 'boschloo':
                res = stats.boschloo_exact([[len(kzm_Pos[kzm_Pos[s]>min_val]), len(kzm_Pos[kzm_Pos[s]<=min_val])],
                                            [len(kzm_Neg[kzm_Neg[s]>min_val]), len(kzm_Neg[kzm_Neg[s]<=min_val])]],
                                            alternative=althyp)
                odr = res.statistic
                pv = res.pvalue
            if verbose and pv < pval:
                if sig_print:
                    print("\n{}".format(s))
                    sig_print = 0
                print("{}: {:0.3f}, {:0.3f}".format(t, odr, pv))
            sigs_md_enrich_df.loc[t, s] = odr
            sigs_md_enrich_pv.loc[t, s] = pv

    
    if plot:
        plot_enrich(sigs_md_enrich_df, sigs_md_enrich_pv, pval, outfile=None, short=verbose)
    return sigs_md_enrich_df, sigs_md_enrich_pv


def enrich_therapy_generic_v2(sigs_df, md_df, min_val=0, pval=0.05, verbose=False, outfile=None, plot=True, test_type='fisher', althyp='two-sided', correction=False):
    sigs_md = sigs_df.copy()
    sigs = sigs_md.columns.tolist()
    therapies = md_df.columns.tolist()

    sigs_md = pd.concat([sigs_md, md_df], axis=1)
    sigs_md = sigs_md.dropna()

    sigs_md_enrich_df = pd.DataFrame(np.zeros((len(therapies), len(sigs))))
    sigs_md_enrich_df.columns = sigs
    sigs_md_enrich_df.index = therapies
    sigs_md_enrich_pv = sigs_md_enrich_df.copy()
    sigs_md_enrich_pv = sigs_md_enrich_pv + 1
    sigs_md_enrich_adjp = sigs_md_enrich_pv.copy()

    for s in sigs:
        sig_print = 1

        for t in therapies:
            '''if t == 'Therapy':
                kzm_Pos = sigs_md[sigs_md[t]=='Post-Therapy']
                kzm_Neg = sigs_md[sigs_md[t]=='Pre-Therapy']
            else:'''
            kzm_Pos = sigs_md[sigs_md[t]=='Y']
            kzm_Neg = sigs_md[sigs_md[t]=='N']

            '''if s == 'SBS17b':
                print(f"{t}: {kzm_Neg.shape[0]}, {kzm_Pos.shape[0]}")
                print(f"{len(kzm_Pos[kzm_Pos[s]>min_val])} -- {len(kzm_Pos[kzm_Pos[s]<=min_val])}")
                print(f"{len(kzm_Neg[kzm_Neg[s]>min_val])} -- {len(kzm_Neg[kzm_Neg[s]<=min_val])}")'''
            
            if kzm_Pos.shape[0]<10 or kzm_Neg.shape[0]<10:
                continue

            if test_type == 'fisher':
                odr, pv = stats.fisher_exact([[len(kzm_Pos[kzm_Pos[s]>min_val]), len(kzm_Pos[kzm_Pos[s]<=min_val])],
                                            [len(kzm_Neg[kzm_Neg[s]>min_val]), len(kzm_Neg[kzm_Neg[s]<=min_val])]],
                                            alternative=althyp)
            elif test_type == 'barnard':
                odr, pv = stats.barnard_exact([[len(kzm_Pos[kzm_Pos[s]>min_val]), len(kzm_Pos[kzm_Pos[s]<=min_val])],
                                            [len(kzm_Neg[kzm_Neg[s]>min_val]), len(kzm_Neg[kzm_Neg[s]<=min_val])]],
                                            alternative=althyp)
            elif test_type == 'boschloo':
                res = stats.boschloo_exact([[len(kzm_Pos[kzm_Pos[s]>min_val]), len(kzm_Pos[kzm_Pos[s]<=min_val])],
                                            [len(kzm_Neg[kzm_Neg[s]>min_val]), len(kzm_Neg[kzm_Neg[s]<=min_val])]],
                                            alternative=althyp)
                odr = res.statistic
                pv = res.pvalue
            if verbose and pv < pval:
                if sig_print:
                    print("\n{}".format(s))
                    sig_print = 0
                print("{}: {:0.3f}, {:0.3f}".format(t, odr, pv))
            sigs_md_enrich_df.loc[t, s] = odr
            sigs_md_enrich_pv.loc[t, s] = pv

        sigs_md_enrich_adjp.loc[:, s] = smm.multipletests(sigs_md_enrich_pv.loc[:, s].tolist(), alpha=0.05, method='fdr_tsbh')[1]

    return sigs_md_enrich_df, sigs_md_enrich_pv, sigs_md_enrich_adjp


def get_contingency_table(sigs_df, md_df, sig, drug, min_val=0):
    sigs_md = sigs_df.copy()
    sigs = sigs_md.columns.tolist()
    therapies = md_df.columns.tolist()

    sigs_md = pd.concat([sigs_md, md_df], axis=1)
    sigs_md = sigs_md.dropna()

    sigs_md_enrich_df = pd.DataFrame(np.zeros((len(therapies), len(sigs))))
    sigs_md_enrich_df.columns = sigs
    sigs_md_enrich_df.index = therapies
    sigs_md_enrich_pv = sigs_md_enrich_df.copy()
    sigs_md_enrich_pv = sigs_md_enrich_pv + 1
    sigs_md_enrich_adjp = sigs_md_enrich_pv.copy()

    kzm_Pos = sigs_md[sigs_md[drug]=='Y']
    kzm_Neg = sigs_md[sigs_md[drug]=='N']

    cont_table = pd.DataFrame([[len(kzm_Pos[kzm_Pos[sig]>min_val]), len(kzm_Pos[kzm_Pos[sig]<=min_val])],
                                [len(kzm_Neg[kzm_Neg[sig]>min_val]), len(kzm_Neg[kzm_Neg[sig]<=min_val])]])
    cont_table.columns = ['Sig+', 'Sig-']
    cont_table.index = ['Treat+', 'Treat-']
    return cont_table

def get_enr_df(df, pv, test_type='fisher', odr=3, pval=0.05):
    mask = pv.copy()
    mask[:] = np.where(mask<pval,0,1)
    pv[:] = np.where(mask==1, 1, pv[:])
    df[:] = np.where(mask==1, 0, df[:])

    to_remove = [x for x in pv.columns if pv.sum(axis=0)[x]==pv.shape[0]]
    df = df.drop(to_remove, axis=1)
    pv = pv.drop(to_remove, axis=1)
    
    if test_type=='fisher':
        df[df<odr] = 0
    elif test_type=='boschloo':
        df[df<0.05] = 0
    sigs_2remove = [x for x in df.columns if df[x].sum()==0]
    df = df.drop(sigs_2remove, axis=1)
    pv = pv.drop(sigs_2remove, axis=1)

    drugs_2remove = [x for x in df.index if df.loc[x].sum()==0]
    df = df.drop(drugs_2remove, axis=0)
    pv = pv.drop(drugs_2remove, axis=0)

    df['Drug'] = df.index.tolist()
    df_mlt = pd.melt(df, id_vars=['Drug'], value_vars=df.columns[0:-1])
    df_mlt.columns = ['Drug', 'Sig', 'Odds_Ratio']

    pv['Drug'] = pv.index.tolist()
    pv_mlt = pd.melt(pv, id_vars=['Drug'], value_vars=pv.columns[0:-1])
    pv_mlt.columns = ['Drug', 'Sig', 'P-value']

    dfpv = pd.concat([df_mlt, pv_mlt['P-value']], axis=1)
    return dfpv