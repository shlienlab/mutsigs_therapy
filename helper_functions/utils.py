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
    """
    Generates a summary table of clonality distribution across samples, grouped by their state.

    This function processes a dataframe of clonality data (`cls_df`) and metadata (`cls_md`) to calculate
    the number of occurrences of various clonality states ('clonal [early]', 'clonal [NA]', 'clonal [late]',
    and 'subclonal') for each unique sample. The resulting table is grouped by a specified state and normalized.

    Parameters:
    -----------
    cls_df : pandas.DataFrame
        A dataframe containing clonality information for each sample. The dataframe should have the columns
        'sampleID', 'Clonality', and other relevant data.
        
    cls_md : pandas.DataFrame
        A metadata dataframe containing the states of the samples. The dataframe should have a column
        matching the `thr_state` parameter with state information for each `sampleID`.

    thr_state : str
        The name of the column in `cls_md` containing the states of the samples. Each sample's state will 
        be retrieved from this column.

    conditions : list
        A list of states to filter and include in the final summary table. Only rows corresponding to 
        these states will be included in the final output.

    Returns:
    --------
    pandas.DataFrame
        A dataframe containing the normalized count of clonality types ('Early Clonal', 'Clonal', 
        'Late Clonal', 'Subclonal') for each state in `conditions`. The dataframe is grouped by state and 
        the values are normalized to sum to 1 for each state.

    Notes:
    ------
    - The clonality types are categorized as:
        - 'Early Clonal' for 'clonal [early]'
        - 'Clonal' for 'clonal [NA]'
        - 'Late Clonal' for 'clonal [late]'
        - 'Subclonal' for 'subclonal'
    - The table is normalized by dividing each state's counts by the sum of counts within that state.
    - Missing clonality types are filled with 0.

    Example Usage:
    --------------
    clones_summary = get_clone_table(cls_df, cls_md, 'State', ['Condition1', 'Condition2'])
    print(clones_summary)
    """
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
    """
    Generates a summary of tumor types classified by platinum sensitivity.

    This function processes a dataframe containing tumor and platinum sensitivity information (`plat_df`) 
    to count the number of occurrences of each tumor type ('Tumor') for both platinum-sensitive (Sig+) 
    and platinum-insensitive (Sig-) groups. It returns a melted dataframe that summarizes these counts 
    along with associated colors for plotting.

    Parameters:
    -----------
    plat_df : pandas.DataFrame
        A dataframe containing tumor information and platinum sensitivity status. The dataframe should have 
        the following columns:
        - 'Tumor': Tumor type or identifier.
        - 'Platin_sig': Platinum sensitivity status, where 'Y' indicates platinum sensitivity ('Sig+') 
          and 'N' indicates platinum resistance ('Sig-').

    Returns:
    --------
    pandas.DataFrame
        A dataframe containing the counts of each tumor type for platinum-sensitive ('Sig+') 
        and platinum-insensitive ('Sig-') groups. The dataframe includes:
        - 'Tumor': Tumor type.
        - 'Sig': Sensitivity status ('Sig-' or 'Sig+').
        - 'Count': Count of each tumor type for each sensitivity status.
        - 'Color': Color associated with each sensitivity status for plotting.

    Notes:
    ------
    - The tumor types in the 'Tumor' column are cleaned by removing asterisks (`*` and `**`).
    - The dataframe is reshaped to a long format with columns for tumor type, sensitivity status, 
      count, and color for visualization.
    - The `set2_cols` color palette is used to assign colors to the 'Sig-' (platinum-insensitive) and 
      'Sig+' (platinum-sensitive) groups.

    Example Usage:
    --------------
    type_summary = get_type_numbers(plat_df)
    print(type_summary)
    """
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
    """
    Melts a signature dataframe into a long format for easier analysis.

    This function transforms a dataframe of mutation signatures (`sigs_df`) from a wide format, where 
    each column represents a mutation signature, to a long format, where each row represents a specific 
    sample and its corresponding mutation counts for each signature.

    Parameters:
    -----------
    sigs_df : pandas.DataFrame
        A dataframe containing mutation signature data. The rows represent samples, and the columns represent 
        different mutation signatures. The index of the dataframe should correspond to sample identifiers.

    Returns:
    --------
    pandas.DataFrame
        A melted dataframe with the following columns:
        - 'Samples': The sample identifier (from the index of the input dataframe).
        - 'Signatures': The names of mutation signatures (original column names in `sigs_df`).
        - 'Mutations': The mutation counts or values corresponding to each sample-signature pair.

    Notes:
    ------
    - The function uses `pd.melt()` to convert the wide dataframe into a long format.
    - The input dataframe should have its sample identifiers as the index.
    
    Example Usage:
    --------------
    melted_df = sigs_melt(sigs_df)
    print(melted_df)
    """
    sigs_df['samples'] = sigs_df.index.tolist()
    sigs_df_mlt = pd.melt(sigs_df, id_vars=['samples'], value_vars=sigs_df.columns.tolist())
    sigs_df_mlt.columns = ['Samples', 'Signatures', 'Mutations']
    return sigs_df_mlt




def get_drivers_table(drivers_df, md_df, sample_col, burden=None):
    """
    Generates a summary table of the distribution of driver mutations across different sample groups.

    This function creates a table showing the frequency of driver mutations for different sample groups 
    based on their tumor burden and treatment status. The table shows the proportion of samples with zero, 
    one, or multiple driver mutations in each group.

    Parameters:
    -----------
    drivers_df : pandas.DataFrame
        A dataframe containing information about driver mutations. Each row corresponds to a sample, 
        and the columns include sample identifiers and driver mutation details.

    md_df : pandas.DataFrame
        A dataframe containing metadata for the samples, including tumor burden and treatment state.
        The index of this dataframe should match the sample identifiers in `drivers_df`.

    sample_col : str
        The column in `drivers_df` that contains sample identifiers. This column will be used to match 
        samples between `drivers_df` and `md_df`.

    burden : str, optional, default=None
        The tumor burden category to filter the samples by. If specified, only samples with the given tumor burden 
        (e.g., 'low') are included in the analysis.

    Returns:
    --------
    pandas.DataFrame
        A summary dataframe with the following columns:
        - 'Zero': Proportion of samples in each group with zero driver mutations.
        - 'One': Proportion of samples in each group with one driver mutation.
        - 'Two+': Proportion of samples in each group with two or more driver mutations.
        The rows represent different sample groups (e.g., 'Primary-Naive', 'Primary-Treated', etc.).

    Notes:
    ------
    - The input `md_df` must contain a column 'Thr_State' that describes the treatment state for each sample.
    - The function calculates the distribution of driver mutations across samples with different treatment states.
    - The `burden` argument allows for filtering based on tumor burden (e.g., 'low').

    Example Usage:
    --------------
    result = get_drivers_table(drivers_df, md_df, sample_col='sampleID', burden='low')
    print(result)
    """
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
    """
    Generates a matrix of driver mutation counts across different sample groups and calculates their percentages.

    This function creates a matrix where each row corresponds to a gene, and each column represents the 
    number of driver mutations in a specific sample group (e.g., Primary-Naive, Primary-Treated, etc.). 
    The matrix also includes a 'Total' column with the total number of driver mutations for each gene across 
    all sample groups, as well as a 'Percentage' column showing the percentage of samples with mutations for 
    each gene.

    Parameters:
    -----------
    drivers : pandas.DataFrame
        A dataframe containing information about driver mutations, where each row corresponds to a driver mutation 
        and includes the sample identifier (`sample_col`) and the gene (`Hugo_Symbol`).

    md_df : pandas.DataFrame
        A dataframe containing metadata for the samples, including their treatment state (`Thr_State`) and tumor burden (`Burden`).

    sample_col : str
        The column in `drivers` that contains sample identifiers. This column is used to match samples between `drivers` and `md_df`.

    burden : str, optional, default=None
        The tumor burden category to filter samples by. If specified as 'low', only samples with low tumor burden will be included 
        in the analysis.

    Returns:
    --------
    pandas.DataFrame
        A dataframe where each row corresponds to a gene (`Hugo_Symbol`) and each column corresponds to a sample group 
        (e.g., 'Primary-Naive', 'Primary-Treated', 'Advanced-Naive', 'Advanced-Treated'). 
        The columns contain the count of driver mutations for each gene in the respective sample group, 
        along with 'Total' (sum of all counts) and 'Percentage' (percentage of samples with mutations for each gene).

    Notes:
    ------
    - The `md_df` dataframe must contain a column 'Thr_State' indicating the treatment state of each sample 
      and a column 'Burden' indicating the tumor burden category ('low' or otherwise).
    - The function calculates mutation counts for genes within different sample groups defined by treatment states. 
    - The `burden` parameter allows for filtering based on tumor burden (e.g., only 'low' burden samples).

    Example Usage:
    --------------
    result = get_drivers_mat(drivers, md_df, sample_col='sampleID', burden='low')
    """
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
    """
    Computes the relative and absolute counts of driver mutations across different tumor types and treatment states.

    This function processes driver mutation data to calculate the number of driver mutations (both relative and absolute) 
    for each tumor type across different treatment conditions ('Primary-Naive', 'Primary-Treated', 'Advanced-Naive', 
    'Advanced-Treated'). It also computes the proportion of samples with no driver mutations, one driver mutation, or two 
    or more driver mutations in each treatment condition.

    Parameters:
    -----------
    dri_df : pandas.DataFrame
        A dataframe containing driver mutation data, with columns including `Tumor_Sample_Barcode` and `Type`, where 
        `Type` represents the tumor type of the samples.

    md_df : pandas.DataFrame
        A metadata dataframe containing sample information, including `Type` (tumor type), `Thr_State` (treatment state), 
        and `Burden` (tumor burden).

    burden : str, optional, default=None
        The tumor burden category to filter samples by. If specified as 'low', only samples with low tumor burden are 
        included in the analysis. Otherwise, all samples are considered.

    Returns:
    --------
    tuple of pandas.DataFrame
        Two dataframes are returned:
        1. `rel_df` - A dataframe containing relative counts (proportions) of samples with 0, 1, or 2+ driver mutations 
           across different tumor types and treatment conditions.
        2. `abs_df` - A dataframe containing absolute counts of samples with 0, 1, or 2+ driver mutations across 
           the same conditions.

    Notes:
    ------
    - The function computes counts for each tumor type (`Type`) based on the `Type` column in both `dri_df` and `md_df`.
    - The `burden` parameter allows filtering based on tumor burden, which can be set to 'low'.
    - The driver mutation counts are categorized by the number of mutations: 'Zero' (no driver mutation), 'One' (one driver mutation),
      and 'Two+' (two or more driver mutations).
    - The function also calculates the relative and absolute counts of mutations for each treatment condition.

    Example Usage:
    --------------
    rel_df, abs_df = get_drivers_type_state(dri_df, md_df, burden='low')
    print(rel_df)
    print(abs_df)
    """
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


def get_sig_counts(sigs_df, kzm611_md, sig_id):
    """
    Counts the occurrences of specific mutation signatures (SBS, DBS, or ID) in different tumor groups based on their 
    hue classification in the provided metadata.

    This function filters and counts the mutation signatures in a subset of samples grouped by their hue classification 
    (Low, High, and optionally Mid for ID signature). It returns a count of occurrences of mutations for each 
    signature type (SBS, DBS, or ID) across these groups.

    Parameters:
    -----------
    sigs_df : pandas.DataFrame
        A DataFrame containing mutation signatures with samples as rows and mutation signatures as columns. 
        Each column represents a mutation signature, and each row represents a sample.

    kzm611_md : pandas.DataFrame
        A metadata DataFrame containing hue classification for each sample (i.e., 'sbs_hue', 'dbs_hue', 'id_hue' 
        columns), which categorize samples into groups such as 'Low', 'High', and 'Mid' based on certain conditions.

    sig_id : str
        The type of mutation signature to count. Can be one of the following:
        - 'SBS' : Single Base Substitution
        - 'DBS' : Double Base Substitution
        - 'ID'  : Insertion/Deletion

    Returns:
    --------
    pandas.DataFrame
        A DataFrame containing the count of mutation occurrences for the specified signature type ('Low', 'High', 
        and 'Mid' for ID signatures) across the corresponding hue categories in the metadata.

    Raises:
    -------
    ValueError
        If `sig_id` is not one of 'SBS', 'DBS', or 'ID', a message will be printed and the function will return `None`.

    Notes:
    ------
    - The function filters the samples by their hue classification (e.g., 'Low', 'High', 'Mid') based on the given 
      signature type (`sig_id`), and calculates the total occurrences of the signature mutations in each category.
    - For 'ID' signature type, there are three hue categories: 'Low', 'Mid', and 'High'. For 'SBS' and 'DBS', 
      only 'Low' and 'High' categories are considered.
    - The function fills missing values with 0 before returning the counts.

    Example Usage:
    --------------
    sig_counts = get_sig_counts(sigs_df, kzm611_md, 'SBS')
    print(sig_counts)
    """
    sigs_subd = sigs_df[[x for x in sigs_df.columns if x.startswith(sig_id)]].copy()
    if sig_id == 'SBS':
        sig_counts = pd.concat([sigs_subd.loc[kzm611_md[kzm611_md.sbs_hue=='Low'].index.tolist()].astype(bool).sum(axis=0),
                                sigs_subd.loc[kzm611_md[kzm611_md.sbs_hue=='High'].index.tolist()].astype(bool).sum(axis=0)], axis=1)
        sig_counts.columns = ['Low', 'High']
    elif sig_id == 'DBS':
        sig_counts = pd.concat([sigs_subd.loc[kzm611_md[kzm611_md.dbs_hue=='Low'].index.tolist()].astype(bool).sum(axis=0),
                                sigs_subd.loc[kzm611_md[kzm611_md.sbs_hue=='High'].index.tolist()].astype(bool).sum(axis=0)], axis=1)
        sig_counts.columns = ['Low', 'High']
    elif sig_id == 'ID':
        sig_counts = pd.concat([sigs_subd.loc[kzm611_md[kzm611_md.id_hue=='Low'].index.tolist()].astype(bool).sum(axis=0),
                                sigs_subd.loc[kzm611_md[kzm611_md.id_hue=='Mid'].index.tolist()].astype(bool).sum(axis=0),
                                sigs_subd.loc[kzm611_md[kzm611_md.id_hue=='High'].index.tolist()].astype(bool).sum(axis=0)], axis=1)
        sig_counts.columns = ['Low', 'Mid', 'High']
    else:
        print('One of SBS, DBS or ID sigs must be selected!')
        return

    sig_counts = sig_counts.fillna(0)
    for col in sig_counts.columns:
        sig_counts[col] = sig_counts[col].astype(int)
    #sig_counts['Sig'] = sig_counts.index.tolist()
    return sig_counts