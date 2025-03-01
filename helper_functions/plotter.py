"""
===========================================
MS: Unique patterns of mutations in childhood cancer highlight chemotherapy’s disease-defining role at relapse
Author: Mehdi Layeghifard
Email: mlayeghi@gmail.com
Date Created: February 28, 2025
Version: 0.1

Description:
This script contains several functions to plot mutational signatures in genomic data.

Usage:
These function are called from within the provided Notebooks.

License:
MIT License - You are free to use, modify, and distribute this code with appropriate attribution.
===========================================
"""

## Data processing imports
import pandas as pd
from collections import OrderedDict

## Plotting imports
import matplotlib.pyplot as plt
import matplotlib.colors as mc
import matplotlib.patches as patches
import matplotlib.patches as mpatches
import matplotlib.axis as axis
import seaborn as sns
import networkx as nx

## Stats imports
import numpy as np
import math
from scipy import stats


GENOME_SIZE = 2897.310462

A_col = "#3288BD"
B_col = "#D53E4F"


def get_mut_dict(mat_df):
    """
    Converts a mutation frequency matrix into a nested dictionary categorized by mutation types.

    Parameters:
    -----------
    mat_df : pd.DataFrame
        A dataframe where:
        - Rows represent samples.
        - Columns represent mutation contexts (e.g., 'A[C>A]T').
        - Values represent mutation counts or frequencies.

    Returns:
    --------
    mutations : dict
        A nested dictionary structured as:
        ```
        {
            sample1: {
                'C>A': {mutation_context1: count, mutation_context2: count, ...},
                'C>G': {...},
                'C>T': {...},
                'T>A': {...},
                'T>C': {...},
                'T>G': {...}
            },
            sample2: { ... }
        }
        ```
        - The first level keys are sample names.
        - The second level keys are mutation types (`C>A`, `C>G`, etc.).
        - The values are ordered dictionaries mapping mutation contexts to their respective counts.

    Notes:
    ------
    - Mutation types are extracted from the 3rd to 5th character of each column name (`col[2:5]`).
    - The function assumes percentage-based mutation counts (`float` values), but can be adjusted for integer counts.
    - The mutation data for each sample is organized into an `OrderedDict` to maintain the order of mutations.
    """
    mutations = dict()
    percentage = True

    for sample in mat_df.index:
        mutations[sample] = {'C>A':OrderedDict(), 'C>G':OrderedDict(), 'C>T':OrderedDict(),
                            'T>A':OrderedDict(), 'T>C':OrderedDict(), 'T>G':OrderedDict()}
        
    for col in mat_df.columns:
        mut_type = col[2:5]
        for sample in mat_df.index:
            if percentage:
                mutCount = float(mat_df.loc[sample, col])
            else:
                mutCount = int(mat_df.loc[sample, col])
            mutations[sample][mut_type][col] = mutCount

    return mutations



## Adapted from https://github.com/AlexandrovLab/SigProfilerPlotting
def plotTMB_therapy(inputDF, pv, scale, color_dict, order=[], Yrange = "adapt", cutoff = 0, output = "TMB_plot.png",
            redbar = "median", xaxis = "Samples (n)", yaxis = "Somatic Mutations per Megabase",
            ascend = True, leftm = 1, rightm = 0.3, topm = 1.4, bottomm = 1):
    """
    Generates a Tumor Mutational Burden (TMB) bar plot stratified by therapy type.

    Parameters:
    -----------
    inputDF : pd.DataFrame
        A dataframe containing mutation burden data with the following structure:
        - A column indicating therapy types.
        - A column containing mutation burden values.
    pv : float or None
        The p-value for statistical significance annotations.
        If None, a permutation-based t-test is performed.
    scale : str or int
        The scaling method for mutation burden normalization:
        - `"exome"`: Uses a default value of 55.
        - `"genome"`: Uses `GENOME_SIZE` (must be predefined).
        - `int`: A custom numerical scaling factor.
    color_dict : dict
        A dictionary mapping therapy types to colors for bar plot visualization.
    order : list, optional (default=[])
        A predefined ordering of therapy types for plot arrangement.
    Yrange : str or list, optional (default="adapt")
        Determines the y-axis range:
        - `"adapt"`: Automatically adapts based on data.
        - `"cancer"`: Sets y-axis limits to `10^-3` to `10^3`.
        - `list`: A custom range [min, max] where values are powers of 10.
    cutoff : float, optional (default=0)
        Minimum mutation burden threshold for samples to be included in the plot.
    output : str, optional (default="TMB_plot.png")
        Filename for saving the generated plot.
    redbar : str, optional (default="median")
        Determines the summary statistic for red reference bars:
        - `"mean"`: Uses the mean mutation burden per group.
        - `"median"`: Uses the median mutation burden per group.
    xaxis : str, optional (default="Samples (n)")
        Label for the x-axis.
    yaxis : str, optional (default="Somatic Mutations per Megabase")
        Label for the y-axis.
    ascend : bool, optional (default=True)
        If True, therapy types are sorted in ascending order based on mutation burden.
    leftm, rightm, topm, bottomm : float, optional
        Margins for adjusting plot spacing.

    Returns:
    --------
    None
        The function generates and displays a bar plot with statistical annotations.

    Notes:
    ------
    - The function normalizes mutation burden by the specified `scale`.
    - If `pv` is not provided, a permutation-based t-test is performed to compare therapy groups.
    - Therapy groups are plotted as colored bars, with dots representing individual samples.
    - A significance annotation is displayed if `pv < 0.05`:
        - `"*"` for `0.01 ≤ pv < 0.05`
        - `"**"` for `0.001 ≤ pv < 0.01`
        - `"***"` for `pv < 0.001`
    - The final plot includes:
        - Therapy groups on the x-axis.
        - Tumor Mutational Burden on the y-axis (log10 scale).
        - A red reference bar indicating the mean or median TMB per group.
        - Individual sample data points overlaid on the bars.
    """
    if type(scale) == int:
        scale = scale
    elif scale == "genome":
        scale  = GENOME_SIZE
    elif scale == "exome":
        scale = 55
    else:
        print("Please input valid scale values: \"exome\", \"genome\" or a numeric value")
        return
    if not pv:
        rvs1 = inputDF[inputDF.Thr_State2=='Advanced\nTreated'].iloc[:, -1].tolist()
        rvs2 = inputDF[inputDF.Thr_State2=='Primary\nNaive'].iloc[:, -1].tolist()

        _, pv = stats.ttest_ind(rvs1, rvs2, permutations=10000, random_state=42)

    inputDF.columns = ['Types', 'Mut_burden']
    df=inputDF[inputDF["Mut_burden"] > cutoff].copy()
    df['log10BURDENpMB'] = df.apply(lambda row: np.log10(row.Mut_burden/scale), axis = 1)
    groups = df.groupby(["Types"])
    if redbar == "mean":
        redbars = groups.mean()["log10BURDENpMB"].sort_values(ascending=ascend)
        names = groups.mean()["log10BURDENpMB"].sort_values(ascending=ascend).index
    elif redbar == "median":
        if len(order) == 0:
            redbars = groups.median()["log10BURDENpMB"].sort_values(ascending=ascend)
            names = groups.median()["log10BURDENpMB"].sort_values(ascending=ascend).index
        else:
            redbars = groups.median()["log10BURDENpMB"]
            redbars = redbars.loc[order]
            names = order
    else:
        print("ERROR: redbar parameter must be either mean or median")
        return
    print(redbars)
    counts = groups.count()["log10BURDENpMB"][names]
    ngroups = groups.ngroups
    #second row of bottom label
    input_groups = inputDF.groupby(["Types"])
    input_counts = input_groups.count()["Mut_burden"][names]
    list1 = counts.to_list()
    list2 = input_counts.to_list()
    str1 = ''
    list3 = prepend(list1, str1)
    str2 = '\n'
    list4 = prepend(list2, str2)
    result = [None]*(len(list3)+len(list4))
    result[::2] = list3
    result[1::2] = list4
    tick_labels = result
    new_labels = [ ''.join(x) for x in zip(tick_labels[0::2], tick_labels[1::2]) ]
    #new_labels = list3
    if Yrange == "adapt":
        ymax = math.ceil(df['log10BURDENpMB'].max())
        ymin = math.floor(df['log10BURDENpMB'].min())
    elif Yrange == "cancer":
        ymax = 3
        ymin = -3
    elif type(Yrange) == list:
        print("Yrange is a list")
        ymax = int(math.log10(Yrange[1]))
        ymin = int(math.log10(Yrange[0]))
    else:
        print("ERROR:Please input valid scale values: \"adapt\", \"cancer\" or a list of two power of 10 numbers")
        return
    #plotting
    if ngroups < 7:
        rightm = rightm + 0.4 * (7 - ngroups)
    if len(names[0])>13:
        leftm = leftm + 0.09 * (len(names[0]) - 13)
        topm = topm + 0.080 * (len(names[0]) - 13)
    fig_width = leftm + rightm + 0.4 * ngroups
    fig_length = 4
    fig, ax = plt.subplots(figsize=(fig_width/2, 4))
    if cutoff < 0:
        print("ERROR: cutoff value is less than 0")
        return
    plt.xlim(0,2*ngroups)
    plt.ylim(ymin,ymax)
    yticks_loc = range(ymin,ymax+1,1)
    plt.yticks(yticks_loc,list(map((lambda x: 10**x), list(yticks_loc)))) 
    plt.xticks(np.arange(1, 2*ngroups+1, step = 2), list1, fontsize=14) 
    plt.tick_params(axis = 'both', which = 'both', length = 0)
    plt.hlines(yticks_loc,0,2*ngroups,colors = 'black',linestyles = "dashed",linewidth = 0.5,zorder = 1)

    bar_y = ymax-1
    if pv < 0.001:
        plt.plot([3,3,5,5], [bar_y, bar_y+.1, bar_y+.1, bar_y], lw=1.5, c='k')
        plt.text(4, bar_y+.2, "***", ha='center', va='bottom', color='k')
    elif pv < 0.01:
        plt.plot([3,3,5,5], [bar_y, bar_y+.1, bar_y+.1, bar_y], lw=1.5, c='k')
        plt.text(4, bar_y+.2, "**", ha='center', va='bottom', color='k')
    elif pv < 0.05:
        plt.plot([3,3,5,5], [bar_y, bar_y+.1, bar_y+.1, bar_y], lw=1.5, c='k')
        plt.text(4, bar_y+.2, "*", ha='center', va='bottom', color='k')
    
    for i in range(len(names)):
        rectangle = mpatches.Rectangle([(i)*2, ymin], 2, ymax-ymin, color=color_dict[names[i]], zorder = 0)
        ax.add_patch(rectangle)


    for i in range(0,ngroups,1):
        X_start = i*2+0.2
        X_end = i*2+2-0.2
        y_values = groups.get_group(names[i])["log10BURDENpMB"].sort_values(ascending = True).values.tolist()
        x_values = list(np.linspace(start = X_start, stop = X_end, num = counts[i]))
        plt.scatter(x_values,y_values,color = "darkgrey",s=1.5)
        plt.hlines(redbars[i], X_start, X_end, colors='darkred', zorder=2)
        plt.text(X_start, redbars[i]+0.1, ("%.3f" % 10**redbars[i]), color='darkred', fontsize=12)
        
    plt.ylabel(yaxis, fontsize=14)
    plt.xlabel(xaxis, fontsize=14)
    axes2 = ax.twiny()
    plt.tick_params(axis = 'both', which = 'both',length = 0)
    plt.xticks(np.arange(1, 2*ngroups+1, step = 2),names,rotation = -90,ha = 'right', fontsize=14)




## Adapted from https://github.com/AlexandrovLab/SigProfilerPlotting
def plotTMB_type(inputDF, pval_dict, scale, order=[], Yrange = "adapt", cutoff = 0, output = "TMB_plot.png",
            redbar = "median", yaxis = "Somatic Mutations per Megabase",
            ascend = True, leftm = 1, rightm = 0.3, topm = 1.4, bottomm = 1):
    """
    This function generates a plot for Tumor Mutation Burden (TMB) based on input data. It visualizes the mutation burden
    for different mutation types, and allows customization of various plot parameters such as scale, cutoff, axis labels, 
    and statistical bars.

    Parameters:
    - inputDF (pandas.DataFrame): A DataFrame containing the mutation types and their corresponding mutation burden values.
    - pval_dict (dict): A dictionary containing p-values for each mutation type, which will be displayed on the plot.
    - scale (str or int): A scale value to normalize mutation burden. Options: 'genome', 'exome', or a numeric value.
    - order (list): An optional list specifying the order of the mutation types.
    - Yrange (str or list): Range for the Y-axis. Options: "adapt", "cancer", or a list of two numbers defining a custom range.
    - cutoff (int): A threshold to filter out mutation burden values below this threshold.
    - output (str): File name for saving the generated plot (default is "TMB_plot.png").
    - redbar (str): The type of statistical bar to display ('mean' or 'median').
    - yaxis (str): Label for the Y-axis (default is "Somatic Mutations per Megabase").
    - ascend (bool): Whether to sort mutation types in ascending order (default is True).
    - leftm (float): Left margin for the plot (default is 1).
    - rightm (float): Right margin for the plot (default is 0.3).
    - topm (float): Top margin for the plot (default is 1.4).
    - bottomm (float): Bottom margin for the plot (default is 1).

    Returns:
    - Saves a TMB plot as an image file (default is "TMB_plot.png").

    The function groups mutations by their type, calculates log-transformed mutation burden values, and plots a scatter plot
    with vertical red lines indicating the mean or median value of the mutation burden for each mutation type. It also 
    includes annotations for p-values and adjusts plot appearance based on the number of mutation types and the desired range 
    for the Y-axis.
    """
    pret_color = 'lightsteelblue'
    post_color = 'lightcoral'
    if type(scale) == int:
        scale = scale
    elif scale == "genome":
        scale  = 2897.310462
    elif scale == "exome":
        scale = 55
    else:
        print("Please input valid scale values: \"exome\", \"genome\" or a numeric value")
        return

    inputDF.columns = ['Types', 'Mut_burden']
    df=inputDF[inputDF["Mut_burden"] > cutoff]
    df['log10BURDENpMB'] = df.apply(lambda row: np.log10(row.Mut_burden/scale), axis = 1)
    groups = df.groupby(["Types"])
    if redbar == "mean":
        redbars = groups.mean()["log10BURDENpMB"].sort_values(ascending=ascend)
        names = groups.mean()["log10BURDENpMB"].sort_values(ascending=ascend).index
    elif redbar == "median":
        if len(order) == 0:
            redbars = groups.median()["log10BURDENpMB"].sort_values(ascending=ascend)
            names = groups.median()["log10BURDENpMB"].sort_values(ascending=ascend).index
        else:
            redbars = groups.median()["log10BURDENpMB"]
            redbars = redbars.loc[order]
            names = order
    else:
        print("ERROR: redbar parameter must be either mean or median")
        return
    counts = groups.count()["log10BURDENpMB"][names]
    ngroups = groups.ngroups
    #second row of bottom label
    input_groups = inputDF.groupby(["Types"])
    input_counts = input_groups.count()["Mut_burden"][names]
    list1 = counts.to_list()
    list2 = input_counts.to_list()
    str1 = ''
    list3 = prepend(list1, str1)
    str2 = '\n'
    list4 = prepend(list2, str2)
    result = [None]*(len(list3)+len(list4))
    result[::2] = list3
    result[1::2] = list4
    tick_labels = result
    new_labels = list3
    if Yrange == "adapt":
        ymax = math.ceil(df['log10BURDENpMB'].max())
        ymin = math.floor(df['log10BURDENpMB'].min())
    elif Yrange == "cancer":
        ymax = 3
        ymin = -3
    elif type(Yrange) == list:
        print("Yrange is a list")
        ymax = int(math.log10(Yrange[1]))
        ymin = int(math.log10(Yrange[0]))
    else:
        print("ERROR:Please input valid scale values: \"adapt\", \"cancer\" or a list of two power of 10 numbers")
        return
    #plotting
    if ngroups < 7:
        rightm = rightm + 0.4 * (7 - ngroups)
    if len(names[0])>13:
        leftm = leftm + 0.09 * (len(names[0]) - 13)
        topm = topm + 0.080 * (len(names[0]) - 13)
    fig_width = leftm + rightm + 0.4 * ngroups
    fig_length = topm + bottomm + (ymax - ymin) * 0.7
    fig, ax = plt.subplots(figsize=(fig_width, fig_length))
    if cutoff < 0:
        print("ERROR: cutoff value is less than 0")
        return
    plt.xlim(0,2*ngroups)
    plt.ylim(ymin,ymax)
    yticks_loc = range(ymin,ymax+1,1)
    plt.yticks(yticks_loc,list(map((lambda x: 10**x), list(yticks_loc)))) 
    plt.xticks(np.arange(1, 2*ngroups+1, step = 2),new_labels) 
    plt.tick_params(axis = 'both', which = 'both',length = 0)
    plt.hlines(yticks_loc,0,2*ngroups,colors = 'black',linestyles = "dashed",linewidth = 0.5,zorder = 1)
    for i in range(0,ngroups,2):
        greystart = [(i)*2,ymin]
        rectangle = mpatches.Rectangle(greystart, 2, ymax-ymin, color = 'aliceblue',zorder = 0)
        ax.add_patch(rectangle)
    for i in range(1,ngroups,2):
        greystart = [(i)*2,ymin]
        rectangle = mpatches.Rectangle(greystart, 2, ymax-ymin, color = 'mistyrose',zorder = 0)
        ax.add_patch(rectangle)
    for i in range(0,ngroups,1):
        X_start = i*2+0.2
        X_end = i*2+2-0.2
        y_values = groups.get_group(names[i])["log10BURDENpMB"].sort_values(ascending = True).values.tolist()
        x_values = list(np.linspace(start = X_start, stop = X_end, num = counts[i]))
        plt.scatter(x_values,y_values,color = "black",s=1.5)
        plt.hlines(redbars[i], X_start, X_end, colors='red', zorder=2)
    for i in range(1,ngroups,2):
        x_line = i*2+2
        plt.axvline(x_line, color='darkgray')
    plt.ylabel(yaxis)
    plt.tick_params(axis = 'both', which = 'both',length = 0)
    new_names = list(dict.fromkeys([x.split('::')[0] for x in names]))

    for i, j in enumerate(np.arange(2, ax.get_xlim()[1], step = 4)):
        ax.text(j, 2.1, new_names[i], horizontalalignment='center')

    for i, n in enumerate(new_names):
        x1 = i*4 + 1
        x2 = i*4 + 3
        pv = pval_dict[n]
        if pv < 0.05:
            plt.text(x1+1, 1.2, f"p={pv:.2}", ha='center', va='bottom', color='k')
            print(n)
        

    fig.subplots_adjust(top = ((ymax - ymin) * 0.7 + bottomm) / fig_length, bottom = bottomm / fig_length, left = leftm / fig_width, right=1 - rightm / fig_width)


## Adapted from https://github.com/AlexandrovLab/SigProfilerPlotting
def plotTMB_generic(inputDF, scale, order=[], Yrange = "adapt", cutoff = 0,
            redbar = "median", yaxis = "Somatic Mutations per Megabase",
            ascend = True, leftm = 1, rightm = 0.3, topm = 1.4, bottomm = 1):
    """
    Plots a graph of Tumor Mutational Burden (TMB) for a given dataset. The plot visualizes the distribution 
    of somatic mutations per megabase across different mutation types, with an optional red bar indicating 
    either the mean or median value for each mutation type.

    Parameters:
    - inputDF (DataFrame): A pandas DataFrame containing mutation type ('Types') and corresponding mutation burden ('Mut_burden').
    - scale (int, str): The scaling factor for mutational burden. Can be a numeric value or one of the strings:
      "genome" (2897.310462) or "exome" (55).
    - order (list, optional): A list specifying the order of mutation types to display in the plot.
    - Yrange (str, list, optional): Defines the y-axis range. Options are "adapt", "cancer", or a list with two power of 10 values.
    - cutoff (int, optional): A threshold below which mutation burden values will be excluded from the plot. Default is 0.
    - output (str, optional): Filename for saving the plot. Default is "TMB_plot.png".
    - redbar (str, optional): Determines whether the red bar represents the "mean" or "median" value for each mutation type.
    - yaxis (str, optional): Label for the y-axis. Default is "Somatic Mutations per Megabase".
    - ascend (bool, optional): If True, the mutation types are ordered in ascending order of mutational burden. Default is True.
    - leftm (float, optional): Left margin for the plot. Default is 1.
    - rightm (float, optional): Right margin for the plot. Default is 0.3.
    - topm (float, optional): Top margin for the plot. Default is 1.4.
    - bottomm (float, optional): Bottom margin for the plot. Default is 1.

    Returns:
    - None

    Notes:
    - If `scale` is a string, it must be either "genome" or "exome", which correspond to predefined scaling values.
    - If `Yrange` is a list, it should contain two values representing the lower and upper limits for the y-axis range.
    - The function will display the p-value for each mutation type if the corresponding p-value is less than 0.05.
    """
    if type(scale) == int:
        scale = scale
    elif scale == "genome":
        scale  = 2897.310462
    elif scale == "exome":
        scale = 55
    else:
        print("Please input valid scale values: \"exome\", \"genome\" or a numeric value")
        return

    inputDF.columns = ['Types', 'Mut_burden']
    df=inputDF[inputDF["Mut_burden"] > cutoff].copy()
    df['log10BURDENpMB'] = df.apply(lambda row: np.log10(row.Mut_burden/scale), axis = 1)
    groups = df.groupby(["Types"])
    if redbar == "mean":
        redbars = groups.mean()["log10BURDENpMB"].sort_values(ascending=ascend)
        names = groups.mean()["log10BURDENpMB"].sort_values(ascending=ascend).index
    elif redbar == "median":
        if len(order) == 0:
            redbars = groups.median()["log10BURDENpMB"].sort_values(ascending=ascend)
            names = groups.median()["log10BURDENpMB"].sort_values(ascending=ascend).index
        else:
            redbars = groups.median()["log10BURDENpMB"]
            redbars = redbars.loc[order]
            names = order
    else:
        print("ERROR: redbar parameter must be either mean or median")
        return
    counts = groups.count()["log10BURDENpMB"][names]
    ngroups = groups.ngroups
    #second row of bottom label
    input_groups = inputDF.groupby(["Types"])
    input_counts = input_groups.count()["Mut_burden"][names]
    list1 = counts.to_list()
    list2 = input_counts.to_list()
    str1 = ''
    list3 = prepend(list1, str1)
    str2 = '\n'
    list4 = prepend(list2, str2)
    result = [None]*(len(list3)+len(list4))
    result[::2] = list3
    result[1::2] = list4
    tick_labels = result
    new_labels = [ ''.join(x) for x in zip(tick_labels[0::2], tick_labels[1::2]) ]
    #new_labels = list3
    if Yrange == "adapt":
        ymax = math.ceil(df['log10BURDENpMB'].max())
        ymin = math.floor(df['log10BURDENpMB'].min())
    elif Yrange == "cancer":
        ymax = 3
        ymin = -3
    elif type(Yrange) == list:
        print("Yrange is a list")
        ymax = int(math.log10(Yrange[1]))
        ymin = int(math.log10(Yrange[0]))
    else:
        print("ERROR:Please input valid scale values: \"adapt\", \"cancer\" or a list of two power of 10 numbers")
        return
    #plotting
    if ngroups < 7:
        rightm = rightm + 0.4 * (7 - ngroups)
    if len(names[0])>13:
        leftm = leftm + 0.09 * (len(names[0]) - 13)
        topm = topm + 0.080 * (len(names[0]) - 13)
    fig_width = leftm + rightm + 0.4 * ngroups
    fig_length = 6
    fig, ax = plt.subplots(figsize=(fig_width, fig_length))
    if cutoff < 0:
        print("ERROR: cutoff value is less than 0")
        return
    plt.xlim(0,2*ngroups)
    plt.ylim(ymin,ymax)
    yticks_loc = range(ymin,ymax+1,1)
    plt.yticks(yticks_loc,list(map((lambda x: 10**x), list(yticks_loc)))) 
    plt.xticks(np.arange(1, 2*ngroups+1, step = 2),new_labels) 
    plt.tick_params(axis = 'both', which = 'both',length = 0)
    plt.hlines(yticks_loc,0,2*ngroups,colors = 'black',linestyles = "dashed",linewidth = 0.5,zorder = 1)
    for i in range(0,ngroups,2):
        greystart = [(i)*2,ymin]
        rectangle = mpatches.Rectangle(greystart, 2, ymax-ymin, color = "lightgrey",zorder = 0)
        ax.add_patch(rectangle)
    for i in range(0,ngroups,1):
        X_start = i*2+0.2
        X_end = i*2+2-0.2
        y_values = groups.get_group(names[i])["log10BURDENpMB"].sort_values(ascending = True).values.tolist()
        x_values = list(np.linspace(start = X_start, stop = X_end, num = counts[i]))
        plt.scatter(x_values,y_values,color = "black",s=1.5)
        plt.hlines(redbars[i], X_start, X_end, colors='red', zorder=2)
        plt.text((leftm + 0.2 + i * 0.4) / fig_width , 0.85 / fig_length , "___",  horizontalalignment='center',transform=plt.gcf().transFigure)
    plt.ylabel(yaxis)
    axes2 = ax.twiny()
    plt.tick_params(axis = 'both', which = 'both',length = 0)
    plt.xticks(np.arange(1, 2*ngroups+1, step = 2),names,rotation = -90,ha = 'right')
    fig.subplots_adjust(top = ((ymax - ymin) * 0.7 + bottomm) / fig_length, bottom = bottomm / fig_length, left = leftm / fig_width, right=1 - rightm / fig_width)




def prepend(list, str): 
    str += '{0}'
    list = [str.format(i) for i in list] 
    return(list)


## Adapted from https://github.com/AlexandrovLab/SigProfilerPlotting
def plotTMB_SBS(inputDF, kzm611_sbs_rel, scale, order=[], Yrange = "adapt", cutoff = 0,
            redbar = "median", yaxis = "Somatic Mutations per Megabase",
            ascend = True, leftm = 1, rightm = 0.3, topm = 1.4, bottomm = 1):
    """
    Plots a graph of Tumor Mutational Burden (TMB) and the SBS (Single Base Substitution) relative frequencies 
    for a given dataset. The plot displays the distribution of somatic mutations per megabase across different 
    mutation types, with an optional red bar representing either the mean or median mutational burden for each 
    mutation type. Additionally, it overlays a bar plot showing the relative frequencies of different mutation 
    types from the SBS dataset.

    Parameters:
    - inputDF (DataFrame): A pandas DataFrame containing mutation type ('Types') and corresponding mutation burden ('Mut_burden').
    - kzm611_sbs_rel (DataFrame): A pandas DataFrame containing the SBS relative frequencies.
    - scale (int, str): The scaling factor for mutational burden. Can be a numeric value or one of the strings:
      "genome" (2897.310462) or "exome" (55).
    - order (list, optional): A list specifying the order of mutation types to display in the plot.
    - Yrange (str, list, optional): Defines the y-axis range. Options are "adapt", "cancer", or a list with two power of 10 values.
    - cutoff (int, optional): A threshold below which mutation burden values will be excluded from the plot. Default is 0.
    - redbar (str, optional): Determines whether the red bar represents the "mean" or "median" value for each mutation type.
    - yaxis (str, optional): Label for the y-axis. Default is "Somatic Mutations per Megabase".
    - ascend (bool, optional): If True, the mutation types are ordered in ascending order of mutational burden. Default is True.
    - leftm (float, optional): Left margin for the plot. Default is 1.
    - rightm (float, optional): Right margin for the plot. Default is 0.3.
    - topm (float, optional): Top margin for the plot. Default is 1.4.
    - bottomm (float, optional): Bottom margin for the plot. Default is 1.

    Returns:
    - None

    Notes:
    - If `scale` is a string, it must be either "genome" or "exome", which correspond to predefined scaling values.
    - If `Yrange` is a list, it should contain two values representing the lower and upper limits for the y-axis range.
    - The function saves the plot as an image in the format defined by the `output` parameter, defaulting to "TMB_plot.png".
    - The second axis (`ax2`) visualizes the mutational burden, while the first axis (`ax1`) shows the SBS relative frequencies.
    - The function also overlays annotations and horizontal lines for visual clarity on the SBS plot.

    Example usage:
    plotTMB_SBS(inputDF, kzm611_sbs_rel, scale="genome", redbar="median", cutoff=0, Yrange="adapt")
    """
    if type(scale) == int:
        scale = scale
    elif scale == "genome":
        scale  = 2897.310462
    elif scale == "exome":
        scale = 55
    else:
        print("Please input valid scale values: \"exome\", \"genome\" or a numeric value")
        return

    inputDF.columns = ['Types', 'Mut_burden']
    df=inputDF[inputDF["Mut_burden"] > cutoff].copy()
    df['log10BURDENpMB'] = df.apply(lambda row: np.log10(row.Mut_burden/scale), axis = 1)
    groups = df.groupby(["Types"])
    if redbar == "mean":
        redbars = groups.mean()["log10BURDENpMB"].sort_values(ascending=ascend)
        names = groups.mean()["log10BURDENpMB"].sort_values(ascending=ascend).index
    elif redbar == "median":
        if len(order) == 0:
            redbars = groups.median()["log10BURDENpMB"].sort_values(ascending=ascend)
            names = groups.median()["log10BURDENpMB"].sort_values(ascending=ascend).index
        else:
            redbars = groups.median()["log10BURDENpMB"]
            redbars = redbars.loc[order]
            names = order
    else:
        print("ERROR: redbar parameter must be either mean or median")
        return
    counts = groups.count()["log10BURDENpMB"][names]
    ngroups = groups.ngroups
    #second row of bottom label
    input_groups = inputDF.groupby(["Types"])
    input_counts = input_groups.count()["Mut_burden"][names]
    list1 = counts.to_list()
    list2 = input_counts.to_list()
    str1 = ''
    list3 = prepend(list1, str1)
    str2 = '\n'
    list4 = prepend(list2, str2)
    result = [None]*(len(list3)+len(list4))
    result[::2] = list3
    result[1::2] = list4
    tick_labels = result
    new_labels = [ ''.join(x) for x in zip(tick_labels[0::2], tick_labels[1::2]) ]
    #new_labels = list3
    if Yrange == "adapt":
        ymax = math.ceil(df['log10BURDENpMB'].max())
        ymin = math.floor(df['log10BURDENpMB'].min())
    elif Yrange == "cancer":
        ymax = 3
        ymin = -3
    elif type(Yrange) == list:
        print("Yrange is a list")
        ymax = int(math.log10(Yrange[1]))
        ymin = int(math.log10(Yrange[0]))
    else:
        print("ERROR:Please input valid scale values: \"adapt\", \"cancer\" or a list of two power of 10 numbers")
        return
    #plotting
    if ngroups < 7:
        rightm = rightm + 0.4 * (7 - ngroups)
    if len(names[0])>13:
        leftm = leftm + 0.09 * (len(names[0]) - 13)
        topm = topm + 0.080 * (len(names[0]) - 13)
    fig_width = leftm + rightm + 0.4 * ngroups
    fig_length = 6
    fig, (ax0, ax2) = plt.subplots(2,1, figsize=(fig_width, fig_length), gridspec_kw={'height_ratios': [1, 2]}, facecolor="#f4f0eb")

    ax1 = kzm611_sbs_rel.plot(kind="bar", stacked=True, width=0.95, color=[A_col, B_col], ax=ax0)
    ax1.set_xlabel('')
    ax1.set_yticks([])
    ax1.set_ylabel('')
    ax1.spines['right'].set_visible(False)
    ax1.spines['top'].set_visible(False)
    ax1.spines['left'].set_visible(False)
    ax1.spines['bottom'].set_visible(False)
    ax1.tick_params(axis="x", bottom=False, top=True, labelbottom=False, labeltop=True)
    # Rotate and align top ticklabels
    plt.setp([tick.label2 for tick in ax1.xaxis.get_major_ticks()], rotation=90,
            ha="left", va="center",rotation_mode="anchor")
    ax1.legend(loc='upper right', bbox_to_anchor=(1,2.7), ncol=15, fontsize=12, facecolor="#f4f0eb")

    ax1.set_facecolor('#f4f0eb')

    ann_y = 1.9
    ax1.text(0.4, ann_y, 'Clock-like', fontsize=12, rotation=90)
    ax1.text(2.4, ann_y, 'APOBEC', fontsize=12, rotation=90)
    ax1.text(4.4, ann_y, 'UV', fontsize=12, rotation=90)
    ax1.text(6.4, ann_y, 'POLE', fontsize=12, rotation=90)
    ax1.text(7.8, ann_y, 'POLE/MMRD', fontsize=12, rotation=90)
    ax1.text(8.8, ann_y, 'POLD1/MMRD', fontsize=12, rotation=90)
    ax1.text(9.8, ann_y, 'Temo', fontsize=12, rotation=90)
    ax1.text(10.8, ann_y, '5-FU', fontsize=12, rotation=90)
    ax1.text(12.4, ann_y, 'Platinum', fontsize=12, rotation=90)
    ax1.text(13.8, ann_y, 'Thiopurine', fontsize=12, rotation=90)
    ax1.text(16.4, ann_y, 'Novel\nTherapy', fontsize=12, rotation=90)
    ax1.text(19.8, ann_y, 'AID', fontsize=12, rotation=90)
    ax1.text(21.2, ann_y, 'MMRD/MSI', fontsize=12, rotation=90)
    ax1.text(22.8, ann_y, 'ROS', fontsize=12, rotation=90)
    ax1.text(26, ann_y, 'Unknown', fontsize=12, rotation=90)
    ax1.text(31, ann_y, 'Novel', fontsize=12, rotation=90)
    ax1.text(37, ann_y, 'Artifacts', fontsize=12, rotation=90)

    line_y = 1.7
    trans = ax1.get_xaxis_transform()
    ax1.plot([-.5,1.4],[line_y,line_y], color="k", transform=trans, clip_on=False)
    ax1.plot([1.6, 3.4],[line_y,line_y], color="k", transform=trans, clip_on=False)
    ax1.plot([3.6, 5.4],[line_y,line_y], color="k", transform=trans, clip_on=False)
    ax1.plot([5.6, 7.4],[line_y,line_y], color="k", transform=trans, clip_on=False)
    ax1.plot([7.6, 8.4],[line_y,line_y], color="k", transform=trans, clip_on=False)
    ax1.plot([8.6, 9.4],[line_y,line_y], color="k", transform=trans, clip_on=False)
    ax1.plot([9.6, 10.4],[line_y,line_y], color="k", transform=trans, clip_on=False)
    ax1.plot([10.6, 11.4],[line_y,line_y], color="k", transform=trans, clip_on=False)
    ax1.plot([11.6, 13.4],[line_y,line_y], color="k", transform=trans, clip_on=False)
    ax1.plot([13.6, 14.4],[line_y,line_y], color="k", transform=trans, clip_on=False)
    ax1.plot([14.6, 19.4],[line_y,line_y], color="k", transform=trans, clip_on=False)
    ax1.plot([19.6, 20.4],[line_y,line_y], color="k", transform=trans, clip_on=False)
    ax1.plot([20.6, 22.4],[line_y,line_y], color="k", transform=trans, clip_on=False)
    ax1.plot([22.6, 23.4],[line_y,line_y], color="k", transform=trans, clip_on=False)
    ax1.plot([23.6, 28.4],[line_y,line_y], color="k", transform=trans, clip_on=False)
    ax1.plot([28.6, 32.4],[line_y,line_y], color="k", transform=trans, clip_on=False)
    ax1.plot([32.6, 39.4],[line_y,line_y], color="k", transform=trans, clip_on=False)

    ax1.plot([9.6, 19.4],[1,1], color="red", transform=trans, clip_on=False)
    ax1.plot([9.6, 19.4],[2.5,2.5], color="red", transform=trans, clip_on=False)
    ax1.plot([9.6, 9.6],[1,2.5], color="red", transform=trans, clip_on=False)
    ax1.plot([19.4, 19.4],[1,2.5], color="red", transform=trans, clip_on=False)
    ax1.text(12.4, 2.7, 'Therapy Signatures', color="red", fontsize=12)

    ax2.set_xlim(0,2*ngroups)
    ax2.set_ylim(ymin,ymax)
    yticks_loc = range(ymin,ymax+1,1)
    ax2.set_yticks(yticks_loc,list(map((lambda x: 10**x), list(yticks_loc))))
    ax2.set_xticks(np.arange(1, 2*ngroups+1, step = 2), new_labels) 
    ax2.tick_params(axis = 'both', which = 'both',length = 0)
    ax2.hlines(yticks_loc,0,2*ngroups,colors = 'black',linestyles = "dashed",linewidth = 0.5,zorder = 1)

    for i in range(0,ngroups):
        if ((i+1) % 2) == 0:
            rectangle = mpatches.Rectangle([(i)*2, ymin], 2, ymax-ymin, color="#f4f0eb", zorder = 0)
            ax2.add_patch(rectangle)
        else:
            rectangle = mpatches.Rectangle([(i)*2, ymin], 2, ymax-ymin, color='lightgray', zorder = 0)
            ax2.add_patch(rectangle)
            
    for i in range(0,ngroups,1):
        X_start = i*2+0.2
        X_end = i*2+2-0.2
        y_values = groups.get_group(names[i])["log10BURDENpMB"].sort_values(ascending = True).values.tolist()
        x_values = list(np.linspace(start = X_start, stop = X_end, num = counts[i]))
        ax2.scatter(x_values,y_values,color = "black",s=1.5)
        ax2.hlines(redbars[i], X_start, X_end, colors='red', zorder=2)
        ax2.text((leftm + 0.2 + i * 0.4) / fig_width , 0.85 / fig_length , "___",  horizontalalignment='center',transform=plt.gcf().transFigure)
    ax2.set_ylabel(yaxis)
    fig.subplots_adjust(top = ((ymax - ymin) * 0.7 + bottomm) / fig_length, bottom = bottomm / fig_length, left = leftm / fig_width, right=1 - rightm / fig_width)



## Adapted from https://github.com/AlexandrovLab/SigProfilerPlotting
def plotTMB_DBS(inputDF, kzm611_sbs_rel, scale, order=[], Yrange = "adapt", cutoff = 0,
            redbar = "median", yaxis = "Somatic Mutations per Megabase",
            ascend = True, leftm = 1, rightm = 0.3, topm = 1.4, bottomm = 1):
    """
    Same as plotTMB_SBS
    """
    if type(scale) == int:
        scale = scale
    elif scale == "genome":
        scale  = 2897.310462
    elif scale == "exome":
        scale = 55
    else:
        print("Please input valid scale values: \"exome\", \"genome\" or a numeric value")
        return

    inputDF.columns = ['Types', 'Mut_burden']
    df=inputDF[inputDF["Mut_burden"] > cutoff].copy()
    df['log10BURDENpMB'] = df.apply(lambda row: np.log10(row.Mut_burden/scale), axis = 1)
    groups = df.groupby(["Types"])
    if redbar == "mean":
        redbars = groups.mean()["log10BURDENpMB"].sort_values(ascending=ascend)
        names = groups.mean()["log10BURDENpMB"].sort_values(ascending=ascend).index
    elif redbar == "median":
        if len(order) == 0:
            redbars = groups.median()["log10BURDENpMB"].sort_values(ascending=ascend)
            names = groups.median()["log10BURDENpMB"].sort_values(ascending=ascend).index
        else:
            redbars = groups.median()["log10BURDENpMB"]
            redbars = redbars.loc[order]
            names = order
    else:
        print("ERROR: redbar parameter must be either mean or median")
        return
    counts = groups.count()["log10BURDENpMB"][names]
    ngroups = groups.ngroups
    #second row of bottom label
    input_groups = inputDF.groupby(["Types"])
    input_counts = input_groups.count()["Mut_burden"][names]
    list1 = counts.to_list()
    list2 = input_counts.to_list()
    str1 = ''
    list3 = prepend(list1, str1)
    str2 = '\n'
    list4 = prepend(list2, str2)
    result = [None]*(len(list3)+len(list4))
    result[::2] = list3
    result[1::2] = list4
    tick_labels = result
    new_labels = [ ''.join(x) for x in zip(tick_labels[0::2], tick_labels[1::2]) ]
    if Yrange == "adapt":
        ymax = math.ceil(df['log10BURDENpMB'].max())
        ymin = math.floor(df['log10BURDENpMB'].min())
    elif Yrange == "cancer":
        ymax = 3
        ymin = -3
    elif type(Yrange) == list:
        print("Yrange is a list")
        ymax = int(math.log10(Yrange[1]))
        ymin = int(math.log10(Yrange[0]))
    else:
        print("ERROR:Please input valid scale values: \"adapt\", \"cancer\" or a list of two power of 10 numbers")
        return
    #plotting
    if ngroups < 7:
        rightm = rightm + 0.4 * (7 - ngroups)
    if len(names[0])>13:
        leftm = leftm + 0.09 * (len(names[0]) - 13)
        topm = topm + 0.080 * (len(names[0]) - 13)
    fig_width = leftm + rightm + 0.4 * ngroups
    fig_length = 6
    fig, (ax0, ax2) = plt.subplots(2,1, figsize=(fig_width, fig_length), gridspec_kw={'height_ratios': [1, 2]}, facecolor="#f4f0eb")
    
    ax1 = kzm611_sbs_rel.plot(kind="bar", stacked=True, width=0.95, color=[A_col, B_col], ax=ax0)
    ax1.set_xlabel('')
    ax1.set_yticks([])
    ax1.set_ylabel('')
    ax1.spines['right'].set_visible(False)
    ax1.spines['top'].set_visible(False)
    ax1.spines['left'].set_visible(False)
    ax1.spines['bottom'].set_visible(False)
    ax1.tick_params(axis="x", bottom=False, top=True, labelbottom=False, labeltop=True)
    # Rotate and align top ticklabels
    plt.setp([tick.label2 for tick in ax1.xaxis.get_major_ticks()], rotation=90,
            ha="left", va="center",rotation_mode="anchor")
    ax1.legend(loc='upper right', bbox_to_anchor=(1,3.25), ncol=15, fontsize=12, facecolor="#f4f0eb")

    ann_y = 2.15
    ax1.text(0, ann_y, 'UV', fontsize=12, rotation=90)
    ax1.text(1, ann_y, 'Tobacco, etc.', fontsize=12, rotation=90)
    ax1.text(2, ann_y, 'Platinum', fontsize=12, rotation=90)
    ax1.text(3.5, ann_y, 'MMRD', fontsize=12, rotation=90)
    ax1.text(6, ann_y, 'Novel', fontsize=12, rotation=90)
    ax1.text(9, ann_y, 'Unknown', fontsize=12, rotation=90)

    line_y = 1.9
    trans = ax1.get_xaxis_transform()
    ax1.plot([-.5, .4],[line_y,line_y], color="k", transform=trans, clip_on=False)
    ax1.plot([.6, 1.4],[line_y,line_y], color="k", transform=trans, clip_on=False)
    ax1.plot([1.6, 2.4],[line_y,line_y], color="k", transform=trans, clip_on=False)
    ax1.plot([2.6, 4.4],[line_y,line_y], color="k", transform=trans, clip_on=False)
    ax1.plot([4.6, 7.4],[line_y,line_y], color="k", transform=trans, clip_on=False)
    ax1.plot([7.6, 11.4],[line_y,line_y], color="k", transform=trans, clip_on=False)

    ax1.set_facecolor("#f4f0eb")

    if cutoff < 0:
        print("ERROR: cutoff value is less than 0")
        return
    ax2.set_xlim(0,2*ngroups)
    #print(len(names[0]))
    ax2.set_ylim(ymin,ymax)
    yticks_loc = range(ymin,ymax+1,1)
    ax2.set_yticks(yticks_loc,list(map((lambda x: 10**x), list(yticks_loc))))
    ax2.set_xticks(np.arange(1, 2*ngroups+1, step = 2), new_labels) 
    ax2.tick_params(axis = 'both', which = 'both',length = 0)
    ax2.hlines(yticks_loc,0,2*ngroups,colors = 'black',linestyles = "dashed",linewidth = 0.5,zorder = 1)
    for i in range(0,ngroups):
        if ((i+1) % 2) == 0:
            rectangle = mpatches.Rectangle([(i)*2, ymin], 2, ymax-ymin, color="#f4f0eb", zorder = 0)
            ax2.add_patch(rectangle)
        else:
            rectangle = mpatches.Rectangle([(i)*2, ymin], 2, ymax-ymin, color='lightgray', zorder = 0)
            ax2.add_patch(rectangle)
    for i in range(0,ngroups,1):
        X_start = i*2+0.2
        X_end = i*2+2-0.2
        y_values = groups.get_group(names[i])["log10BURDENpMB"].sort_values(ascending = True).values.tolist()
        x_values = list(np.linspace(start = X_start, stop = X_end, num = counts[i]))
        ax2.scatter(x_values,y_values,color = "black",s=1.5)
        ax2.hlines(redbars[i], X_start, X_end, colors='red', zorder=2)
        ax2.text((leftm + 0.2 + i * 0.4) / fig_width , 0.85 / fig_length , "___",  horizontalalignment='center',transform=plt.gcf().transFigure)
    ax2.set_ylabel(yaxis)
    fig.subplots_adjust(top = ((ymax - ymin) * 0.7 + bottomm) / fig_length, bottom = bottomm / fig_length, left = leftm / fig_width, right=1 - rightm / fig_width)




## Adapted from https://github.com/AlexandrovLab/SigProfilerPlotting
def plotTMB_ID(inputDF, kzm611_sigs_rel, scale, order=[], Yrange = "adapt", cutoff = 0,
            redbar = "median", yaxis = "Somatic Mutations per Megabase",
            ascend = True, leftm = 1, rightm = 0.3, topm = 1.4, bottomm = 1):
    """
    Same as plotTMB_SBS
    """
    if type(scale) == int:
        scale = scale
    elif scale == "genome":
        scale  = 2897.310462
    elif scale == "exome":
        scale = 55
    else:
        print("Please input valid scale values: \"exome\", \"genome\" or a numeric value")
        return

    inputDF.columns = ['Types', 'Mut_burden']
    df=inputDF[inputDF["Mut_burden"] > cutoff].copy()
    df['log10BURDENpMB'] = df.apply(lambda row: np.log10(row.Mut_burden/scale), axis = 1)
    groups = df.groupby(["Types"])
    if redbar == "mean":
        redbars = groups.mean()["log10BURDENpMB"].sort_values(ascending=ascend)
        names = groups.mean()["log10BURDENpMB"].sort_values(ascending=ascend).index
    elif redbar == "median":
        if len(order) == 0:
            redbars = groups.median()["log10BURDENpMB"].sort_values(ascending=ascend)
            names = groups.median()["log10BURDENpMB"].sort_values(ascending=ascend).index
        else:
            redbars = groups.median()["log10BURDENpMB"]
            redbars = redbars.loc[order]
            names = order
    else:
        print("ERROR: redbar parameter must be either mean or median")
        return
    counts = groups.count()["log10BURDENpMB"][names]
    ngroups = groups.ngroups
    #second row of bottom label
    input_groups = inputDF.groupby(["Types"])
    input_counts = input_groups.count()["Mut_burden"][names]
    list1 = counts.to_list()
    list2 = input_counts.to_list()
    str1 = ''
    list3 = prepend(list1, str1)
    str2 = '\n'
    list4 = prepend(list2, str2)
    result = [None]*(len(list3)+len(list4))
    result[::2] = list3
    result[1::2] = list4
    tick_labels = result
    new_labels = [ ''.join(x) for x in zip(tick_labels[0::2], tick_labels[1::2]) ]
    if Yrange == "adapt":
        ymax = math.ceil(df['log10BURDENpMB'].max())
        ymin = math.floor(df['log10BURDENpMB'].min())
    elif Yrange == "cancer":
        ymax = 3
        ymin = -3
    elif type(Yrange) == list:
        print("Yrange is a list")
        ymax = int(math.log10(Yrange[1]))
        ymin = int(math.log10(Yrange[0]))
    else:
        print("ERROR:Please input valid scale values: \"adapt\", \"cancer\" or a list of two power of 10 numbers")
        return
    #plotting
    if ngroups < 7:
        rightm = rightm + 0.4 * (7 - ngroups)
    if len(names[0])>13:
        leftm = leftm + 0.09 * (len(names[0]) - 13)
        topm = topm + 0.080 * (len(names[0]) - 13)
    fig_width = leftm + rightm + 0.4 * ngroups
    fig_length = 6
    fig, (ax0, ax2) = plt.subplots(2,1, figsize=(fig_width, fig_length), gridspec_kw={'height_ratios': [1, 2]}, facecolor="#f4f0eb")
    
    ax1 = kzm611_sigs_rel.plot(kind="bar", stacked=True, width=0.95, color=[A_col, B_col], ax=ax0)
    ax1.set_xlabel('')
    ax1.set_yticks([])
    ax1.set_ylabel('')
    ax1.spines['right'].set_visible(False)
    ax1.spines['top'].set_visible(False)
    ax1.spines['left'].set_visible(False)
    ax1.spines['bottom'].set_visible(False)
    ax1.tick_params(axis="x", bottom=False, top=True, labelbottom=False, labeltop=True)
    # Rotate and align top ticklabels
    plt.setp([tick.label2 for tick in ax1.xaxis.get_major_ticks()], rotation=90,
            ha="left", va="center",rotation_mode="anchor")
    ax1.legend(loc='upper right', bbox_to_anchor=(1,2.75), ncol=15, fontsize=12, facecolor="#f4f0eb")

    ann_y = 1.85
    ax1.text(0.4, ann_y, 'MMRD/MSI', fontsize=12, rotation=90)
    ax1.text(2, ann_y, 'Tobacco', fontsize=12, rotation=90)
    ax1.text(3, ann_y, 'Radiation', fontsize=12, rotation=90)
    ax1.text(4, ann_y, 'Top2A', fontsize=12, rotation=90)
    ax1.text(8, ann_y, 'Novel', fontsize=12, rotation=90)
    ax1.text(14, ann_y, 'Unknown', fontsize=12, rotation=90)

    line_y = 1.65
    trans = ax1.get_xaxis_transform()
    ax1.plot([-.5,1.4],[line_y,line_y], color="k", transform=trans, clip_on=False)
    ax1.plot([1.6, 2.4],[line_y,line_y], color="k", transform=trans, clip_on=False)
    ax1.plot([2.6, 3.4],[line_y,line_y], color="k", transform=trans, clip_on=False)
    ax1.plot([3.6, 4.4],[line_y,line_y], color="k", transform=trans, clip_on=False)
    ax1.plot([4.6, 12.4],[line_y,line_y], color="k", transform=trans, clip_on=False)
    ax1.plot([12.6, 16.4],[line_y,line_y], color="k", transform=trans, clip_on=False)

    ax1.set_facecolor("#f4f0eb")

    if cutoff < 0:
        print("ERROR: cutoff value is less than 0")
        return
    ax2.set_xlim(0,2*ngroups)
    ax2.set_ylim(ymin,ymax)
    yticks_loc = range(ymin,ymax+1,1)
    ax2.set_yticks(yticks_loc,list(map((lambda x: 10**x), list(yticks_loc))))
    ax2.set_xticks(np.arange(1, 2*ngroups+1, step = 2), new_labels) 
    ax2.tick_params(axis = 'both', which = 'both',length = 0)
    ax2.hlines(yticks_loc,0,2*ngroups,colors = 'black',linestyles = "dashed",linewidth = 0.5,zorder = 1)

    for i in range(0,ngroups):
        if ((i+1) % 2) == 0:
            rectangle = mpatches.Rectangle([(i)*2, ymin], 2, ymax-ymin, color="#f4f0eb", zorder = 0)
            ax2.add_patch(rectangle)
        else:
            rectangle = mpatches.Rectangle([(i)*2, ymin], 2, ymax-ymin, color='lightgray', zorder = 0)
            ax2.add_patch(rectangle)
    for i in range(0,ngroups,1):
        X_start = i*2+0.2
        X_end = i*2+2-0.2
        y_values = groups.get_group(names[i])["log10BURDENpMB"].sort_values(ascending = True).values.tolist()
        x_values = list(np.linspace(start = X_start, stop = X_end, num = counts[i]))
        ax2.scatter(x_values,y_values,color = "black",s=1.5)
        ax2.hlines(redbars[i], X_start, X_end, colors='red', zorder=2)
        ax2.text((leftm + 0.2 + i * 0.4) / fig_width , 0.85 / fig_length , "___",  horizontalalignment='center',transform=plt.gcf().transFigure)
    ax2.set_ylabel(yaxis)
    fig.subplots_adjust(top = ((ymax - ymin) * 0.7 + bottomm) / fig_length, bottom = bottomm / fig_length, left = leftm / fig_width, right=1 - rightm / fig_width)




## Adapted from https://github.com/AlexandrovLab/SigProfilerPlotting
def plotTMB_CN(inputDF, kzm611_sigs_rel, scale, order=[], Yrange = "adapt", cutoff = 0,
            redbar = "median", yaxis = "Somatic Mutations per Megabase",
            ascend = True, leftm = 1, rightm = 0.3, topm = 1.4, bottomm = 1):
    """
    Same as plotTMB_SBS
    """
    if type(scale) == int:
        scale = scale
    elif scale == "genome":
        scale  = 2897.310462
    elif scale == "exome":
        scale = 55
    else:
        print("Please input valid scale values: \"exome\", \"genome\" or a numeric value")
        return

    inputDF.columns = ['Types', 'Mut_burden']
    df=inputDF[inputDF["Mut_burden"] > cutoff].copy()
    df['log10BURDENpMB'] = df.apply(lambda row: np.log10(row.Mut_burden/scale), axis = 1)
    groups = df.groupby(["Types"])
    if redbar == "mean":
        redbars = groups.mean()["log10BURDENpMB"].sort_values(ascending=ascend)
        names = groups.mean()["log10BURDENpMB"].sort_values(ascending=ascend).index
    elif redbar == "median":
        if len(order) == 0:
            redbars = groups.median()["log10BURDENpMB"].sort_values(ascending=ascend)
            names = groups.median()["log10BURDENpMB"].sort_values(ascending=ascend).index
        else:
            redbars = groups.median()["log10BURDENpMB"]
            redbars = redbars.loc[order]
            names = order
    else:
        print("ERROR: redbar parameter must be either mean or median")
        return
    counts = groups.count()["log10BURDENpMB"][names]
    ngroups = groups.ngroups
    #second row of bottom label
    input_groups = inputDF.groupby(["Types"])
    input_counts = input_groups.count()["Mut_burden"][names]
    list1 = counts.to_list()
    list2 = input_counts.to_list()
    str1 = ''
    list3 = prepend(list1, str1)
    str2 = '\n'
    list4 = prepend(list2, str2)
    result = [None]*(len(list3)+len(list4))
    result[::2] = list3
    result[1::2] = list4
    tick_labels = result
    new_labels = [ ''.join(x) for x in zip(tick_labels[0::2], tick_labels[1::2]) ]
    if Yrange == "adapt":
        ymax = math.ceil(df['log10BURDENpMB'].max())
        ymin = math.floor(df['log10BURDENpMB'].min())
    elif Yrange == "cancer":
        ymax = 3
        ymin = -3
    elif type(Yrange) == list:
        print("Yrange is a list")
        ymax = int(math.log10(Yrange[1]))
        ymin = int(math.log10(Yrange[0]))
    else:
        print("ERROR:Please input valid scale values: \"adapt\", \"cancer\" or a list of two power of 10 numbers")
        return
    #plotting
    if ngroups < 7:
        rightm = rightm + 0.4 * (7 - ngroups)
    if len(names[0])>13:
        leftm = leftm + 0.09 * (len(names[0]) - 13)
        topm = topm + 0.080 * (len(names[0]) - 13)
    fig_width = leftm + rightm + 0.4 * ngroups
    fig_length = 6
    fig, (ax0, ax2) = plt.subplots(2,1, figsize=(fig_width, fig_length), gridspec_kw={'height_ratios': [1, 2]})

    
    ax1 = kzm611_sigs_rel.plot(kind="bar", stacked=True, width=0.95, color=[A_col, B_col], ax=ax0)
    ax1.set_xlabel('')
    ax1.set_yticks([])
    ax1.set_ylabel('')
    ax1.spines['right'].set_visible(False)
    ax1.spines['top'].set_visible(False)
    ax1.spines['left'].set_visible(False)
    ax1.spines['bottom'].set_visible(False)
    ax1.tick_params(axis="x", bottom=False, top=True, labelbottom=False, labeltop=True)
    # Rotate and align top ticklabels
    plt.setp([tick.label2 for tick in ax1.xaxis.get_major_ticks()], rotation=90,
            ha="left", va="center",rotation_mode="anchor")
    ax1.get_legend().remove()

    ann_y = 1.85
    ax1.text(-0.25, ann_y, 'Tetraploidy', fontsize=12, rotation=90)
    ax1.text(0.75, ann_y, 'Octoploidy', fontsize=12, rotation=90)
    ax1.text(2.75, ann_y, 'Chromothripsis', fontsize=12, rotation=90)
    ax1.text(4.75, ann_y, 'LOH', fontsize=12, rotation=90)
    ax1.text(5.75, ann_y, 'dMMR', fontsize=12, rotation=90)
    ax1.text(8.75, ann_y, 'Novel', fontsize=12, rotation=90)
    ax1.text(12.75, ann_y, 'Unknown', fontsize=12, rotation=90)

    line_y = 1.65
    trans = ax1.get_xaxis_transform()
    ax1.plot([-.5, .4],[line_y,line_y], color="k", transform=trans, clip_on=False)
    ax1.plot([.6, 1.4],[line_y,line_y], color="k", transform=trans, clip_on=False)
    ax1.plot([1.6, 4.4],[line_y,line_y], color="k", transform=trans, clip_on=False)
    ax1.plot([4.6, 5.4],[line_y,line_y], color="k", transform=trans, clip_on=False)
    ax1.plot([5.6, 6.4],[line_y,line_y], color="k", transform=trans, clip_on=False)
    ax1.plot([6.6, 12.4],[line_y,line_y], color="k", transform=trans, clip_on=False)
    ax1.plot([12.6, 13.4],[line_y,line_y], color="k", transform=trans, clip_on=False)

    if cutoff < 0:
        print("ERROR: cutoff value is less than 0")
        return
    ax2.set_xlim(0,2*ngroups)
    #print(len(names[0]))
    ax2.set_ylim(ymin,ymax)
    yticks_loc = range(ymin,ymax+1,1)
    ax2.set_yticks(yticks_loc,list(map((lambda x: 10**x), list(yticks_loc))))
    ax2.set_xticks(np.arange(1, 2*ngroups+1, step = 2), new_labels) 
    ax2.tick_params(axis = 'both', which = 'both',length = 0)
    ax2.hlines(yticks_loc,0,2*ngroups,colors = 'black',linestyles = "dashed",linewidth = 0.5,zorder = 1)

    for i in range(0,ngroups):
        if ((i+1) % 2) == 0:
            rectangle = mpatches.Rectangle([(i)*2, ymin], 2, ymax-ymin, color="white", zorder = 0)
            ax2.add_patch(rectangle)
        else:
            rectangle = mpatches.Rectangle([(i)*2, ymin], 2, ymax-ymin, color='lightgray', zorder = 0)
            ax2.add_patch(rectangle)
    for i in range(0,ngroups,1):
        X_start = i*2+0.2
        X_end = i*2+2-0.2
        #rg = 1.8
        y_values = groups.get_group(names[i])["log10BURDENpMB"].sort_values(ascending = True).values.tolist()
        x_values = list(np.linspace(start = X_start, stop = X_end, num = counts[i]))
        ax2.scatter(x_values,y_values,color = "black",s=1.5)
        ax2.hlines(redbars[i], X_start, X_end, colors='red', zorder=2)
        ax2.text((leftm + 0.2 + i * 0.4) / fig_width , 0.85 / fig_length , "___",  horizontalalignment='center',transform=plt.gcf().transFigure)
    ax2.set_ylabel(yaxis)
    fig.subplots_adjust(top = ((ymax - ymin) * 0.7 + bottomm) / fig_length, bottom = bottomm / fig_length, left = leftm / fig_width, right=1 - rightm / fig_width)




## Adapted from https://github.com/AlexandrovLab/SigProfilerPlotting
def plotTMB_SV(inputDF, kzm611_sigs_rel, scale, order=[], Yrange = "adapt", cutoff = 0,
            redbar = "median", yaxis = "Somatic Mutations per Megabase",
            ascend = True, leftm = 1, rightm = 0.3, topm = 1.4, bottomm = 1):
    """
    Same as plotTMB_SBS
    """
    if type(scale) == int:
        scale = scale
    elif scale == "genome":
        scale  = 2897.310462
    elif scale == "exome":
        scale = 55
    else:
        print("Please input valid scale values: \"exome\", \"genome\" or a numeric value")
        return

    inputDF.columns = ['Types', 'Mut_burden']
    df=inputDF[inputDF["Mut_burden"] > cutoff].copy()
    df['log10BURDENpMB'] = df.apply(lambda row: np.log10(row.Mut_burden/scale), axis = 1)
    groups = df.groupby(["Types"])
    if redbar == "mean":
        redbars = groups.mean()["log10BURDENpMB"].sort_values(ascending=ascend)
        names = groups.mean()["log10BURDENpMB"].sort_values(ascending=ascend).index
    elif redbar == "median":
        if len(order) == 0:
            redbars = groups.median()["log10BURDENpMB"].sort_values(ascending=ascend)
            names = groups.median()["log10BURDENpMB"].sort_values(ascending=ascend).index
        else:
            redbars = groups.median()["log10BURDENpMB"]
            redbars = redbars.loc[order]
            names = order
    else:
        print("ERROR: redbar parameter must be either mean or median")
        return
    counts = groups.count()["log10BURDENpMB"][names]
    ngroups = groups.ngroups
    #second row of bottom label
    input_groups = inputDF.groupby(["Types"])
    input_counts = input_groups.count()["Mut_burden"][names]
    list1 = counts.to_list()
    list2 = input_counts.to_list()
    str1 = ''
    list3 = prepend(list1, str1)
    str2 = '\n'
    list4 = prepend(list2, str2)
    result = [None]*(len(list3)+len(list4))
    result[::2] = list3
    result[1::2] = list4
    tick_labels = result
    new_labels = [ ''.join(x) for x in zip(tick_labels[0::2], tick_labels[1::2]) ]
    if Yrange == "adapt":
        ymax = math.ceil(df['log10BURDENpMB'].max())
        ymin = math.floor(df['log10BURDENpMB'].min())
    elif Yrange == "cancer":
        ymax = 3
        ymin = -3
    elif type(Yrange) == list:
        print("Yrange is a list")
        ymax = int(math.log10(Yrange[1]))
        ymin = int(math.log10(Yrange[0]))
    else:
        print("ERROR:Please input valid scale values: \"adapt\", \"cancer\" or a list of two power of 10 numbers")
        return
    #plotting
    if ngroups < 7:
        rightm = rightm + 0.4 * (7 - ngroups)
    if len(names[0])>13:
        leftm = leftm + 0.09 * (len(names[0]) - 13)
        topm = topm + 0.080 * (len(names[0]) - 13)
    fig_width = leftm + rightm + 0.4 * ngroups
    fig_length = 6
    fig, (ax0, ax2) = plt.subplots(2,1, figsize=(fig_width, fig_length), gridspec_kw={'height_ratios': [1, 2]})

    
    ax1 = kzm611_sigs_rel.plot(kind="bar", stacked=True, width=0.95, color=[A_col, B_col], ax=ax0)
    ax1.set_xlabel('')
    ax1.set_yticks([])
    ax1.set_ylabel('')
    ax1.spines['right'].set_visible(False)
    ax1.spines['top'].set_visible(False)
    ax1.spines['left'].set_visible(False)
    ax1.spines['bottom'].set_visible(False)
    ax1.tick_params(axis="x", bottom=False, top=True, labelbottom=False, labeltop=True)
    # Rotate and align top ticklabels
    plt.setp([tick.label2 for tick in ax1.xaxis.get_major_ticks()], rotation=90,
            ha="left", va="center",rotation_mode="anchor")
    ax1.get_legend().remove()

    ann_y = 1.85
    ax1.text(-0.25, ann_y, 'HRD', fontsize=12, rotation=90)
    ax1.text(2.75, ann_y, 'Novel', fontsize=12, rotation=90)
    ax1.text(8.5, ann_y, 'Unknow', fontsize=12, rotation=90)

    line_y = 1.65
    trans = ax1.get_xaxis_transform()
    ax1.plot([-.5, .4],[line_y,line_y], color="k", transform=trans, clip_on=False)
    ax1.plot([.6, 5.4],[line_y,line_y], color="k", transform=trans, clip_on=False)
    ax1.plot([5.6, 11.4],[line_y,line_y], color="k", transform=trans, clip_on=False)

    if cutoff < 0:
        print("ERROR: cutoff value is less than 0")
        return
    ax2.set_xlim(0,2*ngroups)
    #print(len(names[0]))
    ax2.set_ylim(ymin,ymax)
    yticks_loc = range(ymin,ymax+1,1)
    ax2.set_yticks(yticks_loc,list(map((lambda x: 10**x), list(yticks_loc))))
    ax2.set_xticks(np.arange(1, 2*ngroups+1, step = 2), new_labels) 
    ax2.tick_params(axis = 'both', which = 'both',length = 0)
    ax2.hlines(yticks_loc,0,2*ngroups,colors = 'black',linestyles = "dashed",linewidth = 0.5,zorder = 1)

    for i in range(0,ngroups):
        if ((i+1) % 2) == 0:
            rectangle = mpatches.Rectangle([(i)*2, ymin], 2, ymax-ymin, color="white", zorder = 0)
            ax2.add_patch(rectangle)
        else:
            rectangle = mpatches.Rectangle([(i)*2, ymin], 2, ymax-ymin, color='lightgray', zorder = 0)
            ax2.add_patch(rectangle)
    for i in range(0,ngroups,1):
        X_start = i*2+0.2
        X_end = i*2+2-0.2
        #rg = 1.8
        y_values = groups.get_group(names[i])["log10BURDENpMB"].sort_values(ascending = True).values.tolist()
        x_values = list(np.linspace(start = X_start, stop = X_end, num = counts[i]))
        ax2.scatter(x_values,y_values,color = "black",s=1.5)
        ax2.hlines(redbars[i], X_start, X_end, colors='red', zorder=2)
        ax2.text((leftm + 0.2 + i * 0.4) / fig_width , 0.85 / fig_length , "___",  horizontalalignment='center',transform=plt.gcf().transFigure)
    ax2.set_ylabel(yaxis)
    fig.subplots_adjust(top = ((ymax - ymin) * 0.7 + bottomm) / fig_length, bottom = bottomm / fig_length, left = leftm / fig_width, right=1 - rightm / fig_width)




def get_diff_dict(mat_df, labels):
    """
    Computes the differential mutation values between two specified labels in a mutation matrix.

    This function compares the mutation burden (or other values) between two specified sample labels 
    from a mutation matrix (`mat_df`) and returns a dictionary that represents the differences for each 
    mutation type (e.g., 'C>A', 'C>G', etc.) between the two labels. The result is a dictionary with the 
    mutation types as keys and the mutation burdens for each column as values, distinguishing whether 
    the value increased or decreased between the two labels.

    Parameters:
    - mat_df (DataFrame): A pandas DataFrame containing mutation data where the rows are sample names 
      and columns represent mutations, formatted as 'reference>mutation'.
    - labels (list): A list of two strings, where each string corresponds to a sample label in `mat_df`. 
      These are the labels to compare, and the function computes the difference between them.

    Returns:
    - dict: A dictionary where keys are sample labels (`labels[0]`, `labels[1]`), and the values are 
      dictionaries of mutation types ('C>A', 'C>G', etc.) mapped to their respective mutation burdens. 
      The mutation burdens represent the difference between the two labels, where a positive value 
      indicates an increase in the second label (`labels[1]`), and a negative value indicates a decrease.

    Example:
    mutations = get_diff_dict(mat_df, ['Sample1', 'Sample2'])
    """
    mutations = dict()

    for sample in mat_df.index:
        mutations[sample] = {'C>A':OrderedDict(), 'C>G':OrderedDict(), 'C>T':OrderedDict(),
                            'T>A':OrderedDict(), 'T>C':OrderedDict(), 'T>G':OrderedDict()}
        
    for col in mat_df.columns:
        mut_type = col[2:5]
        mut_mean = float(mat_df.loc[labels[1], col]) - float(mat_df.loc[labels[0], col])
        if mut_mean == 0:
            mutations[labels[0]][mut_type][col] = 0
            mutations[labels[1]][mut_type][col] = 0
        elif mut_mean > 0:
            mutations[labels[0]][mut_type][col] = 0
            mutations[labels[1]][mut_type][col] = mut_mean
        else:
            mutations[labels[0]][mut_type][col] = abs(mut_mean)
            mutations[labels[1]][mut_type][col] = 0

    return mutations




## Adapted from https://github.com/AlexandrovLab/SigProfilerPlotting
def plot_profile_diff(sample1, sample2, name1, name2, ymax):
    """
    Plots the mutation profiles for two samples with differences in mutation counts displayed side by side.

    This function generates a bar plot displaying the mutation profiles of two samples (`sample1` and `sample2`). 
    It visualizes the mutation counts of different mutation types (e.g., 'C>A', 'C>G', etc.) for each sample, 
    along with labeled axes and color-coded mutation types. The function uses a specified maximum y-value (`ymax`) 
    for the y-axis to control the plot’s scale.

    Parameters:
    - sample1 (dict): A dictionary where the keys are mutation types, and the values are dictionaries with sequences as keys 
      and their corresponding mutation counts as values for the first sample.
    - sample2 (dict): A dictionary similar to `sample1` for the second sample to be compared.
    - name1 (str): The label/name to display for the first sample.
    - name2 (str): The label/name to display for the second sample.
    - ymax (float): The maximum value for the y-axis to set the scale for the plot.

    Returns:
    - None: This function generates and displays a plot with mutation profiles for the two samples.

    Example:
    plot_profile_diff(sample1, sample2, 'Sample A', 'Sample B', ymax=100)
    """
    plt.rcParams['axes.linewidth'] = 2
    plot1 = plt.figure(figsize=(43.93,12))
    plt.rc('axes', edgecolor='lightgray')
    panel1 = plot1.add_axes([0.04, 0.491, 0.95, 0.4])
    xlabels = []
    x = 0.4
    colors = [[3/256,189/256,239/256], [1/256,1/256,1/256],[228/256,41/256,38/256], [203/256,202/256,202/256], [162/256,207/256,99/256], [236/256,199/256,197/256]]
    i = 0

    for key in sample1:
        for seq in sample1[key]:
            xlabels.append(seq[0]+seq[2]+seq[6])
            panel1.bar(x, sample1[key][seq],width=0.4,color=colors[i],align='center', zorder=1000)
            x += 1
        i += 1
        
    x = .043
    y3 = .9
    y = ymax #*1.25
    y2 = y+2
    for i in range(0, 6, 1):
        panel1.add_patch(plt.Rectangle((x,y3), .15, .05, facecolor=colors[i], clip_on=False, transform=plt.gcf().transFigure)) 
        x += .159

    yText = y3 + .06
    panel1.text(.1, yText, 'C>A', fontsize=55, fontweight='bold', fontname='Arial', transform=plt.gcf().transFigure)
    panel1.text(.255, yText, 'C>G', fontsize=55, fontweight='bold', fontname='Arial', transform=plt.gcf().transFigure)
    panel1.text(.415, yText, 'C>T', fontsize=55, fontweight='bold', fontname='Arial', transform=plt.gcf().transFigure)
    panel1.text(.575, yText, 'T>A', fontsize=55, fontweight='bold', fontname='Arial', transform=plt.gcf().transFigure)
    panel1.text(.735, yText, 'T>C', fontsize=55, fontweight='bold', fontname='Arial', transform=plt.gcf().transFigure)
    panel1.text(.89, yText, 'T>G', fontsize=55, fontweight='bold', fontname='Arial', transform=plt.gcf().transFigure)

    panel1.text(0.05, 0.8, name1, fontsize=60, weight="bold", color="black", fontname="Arial", transform=plt.gcf().transFigure)

    ytick_offest = y/4 #int(y/4)
    ylabs = [0, ytick_offest, ytick_offest*2, ytick_offest*3, ytick_offest*4]
    ylabels = [f"{round(x*100, 1)}%" for x in [0, ytick_offest, ytick_offest*2, ytick_offest*3, ytick_offest*4]]

    labs = np.arange(0.375,96.375,1)

    panel1.set_xlim([0, 96])
    panel1.set_ylim([0, y])
    panel1.set_yticks(ylabs)
    count = 0
    m = 0

    for i in range (0, 96, 1):
        panel1.text(i/101 + .0415, .42, xlabels[i][0], fontsize=30, color='gray', rotation='vertical', verticalalignment='center', fontname='Courier New', transform=plt.gcf().transFigure)
        panel1.text(i/101 + .0415, .444, xlabels[i][1], fontsize=30, color=colors[m], rotation='vertical', verticalalignment='center', fontname='Courier New', fontweight='bold',transform=plt.gcf().transFigure)
        panel1.text(i/101 + .0415, .471, xlabels[i][2], fontsize=30, color='gray', rotation='vertical', verticalalignment='center', fontname='Courier New', transform=plt.gcf().transFigure)
        count += 1
        if count == 16:
            count = 0
            m += 1

    panel1.set_yticklabels(ylabels, fontsize=30)
    panel1.yaxis.grid(True)
    panel1.grid(which='major', axis='y', color=[0.93,0.93,0.93], zorder=1)
    panel1.set_ylabel('')

    panel1.tick_params(axis='both',which='both',\
                    bottom=False, labelbottom=False,\
                    left=True, labelleft=True,\
                    right=True, labelright=False,\
                    top=False, labeltop=False,\
                    direction='in', length=25, colors='lightgray', width=2)


    [i.set_color("black") for i in panel1.get_yticklabels()]

    ###### 2nd plot
    panel2 = plot1.add_axes([0.04, 0, 0.95, 0.4])
    xlabels = []
    x = 0.4
    i = 0

    for key in sample2:
        for seq in sample2[key]:
            xlabels.append(seq[0]+seq[2]+seq[6])
            panel2.bar(x, sample2[key][seq], width=0.4, color=colors[i], align='center', zorder=1000)
            x += 1
        i += 1


    ylabs = [0, ytick_offest, ytick_offest*2, ytick_offest*3, ytick_offest*4]
    ylabels = [f"{round(x*100, 1)}%" for x in [0, ytick_offest, ytick_offest*2, ytick_offest*3, ytick_offest*4]]

    panel2.set_xlim([0, 96])
    panel2.set_ylim([0, y])
    panel2.set_yticks(ylabs)
    panel2.set_yticklabels(ylabels, fontsize=30)

    panel2.yaxis.grid(True)
    panel2.grid(which='major', axis='y', color=[0.93,0.93,0.93], zorder=1)
    panel2.set_xlabel('')
    panel2.set_ylabel('')

    panel2.text(0.05, 0.32, name2, fontsize=60, weight="bold", color="black", fontname="Arial", transform=plt.gcf().transFigure)

    panel2.tick_params(axis='both',which='both',\
                    bottom=False, labelbottom=False,\
                    left=True, labelleft=True,\
                    right=True, labelright=False,\
                    top=False, labeltop=False,\
                    direction='in', length=25, colors='lightgray', width=2)

    [i.set_color("black") for i in panel2.get_yticklabels()]
    plt.show()



## Adapted from https://github.com/AlexandrovLab/SigProfilerPlotting
def plot_3samples_SBS96(sample1, sample2, sample3, name1, name2, name3, ymax):
    """
    Plots three SBS96 mutation profiles for three samples in a single figure with 
    three vertical bar plots. The mutation counts are represented as bars with different colors 
    for each mutation type. Each subplot represents one sample, with customizable axes, labels, 
    and horizontal lines indicating specific mutation frequency thresholds.

    Parameters:
    -----------
    sample1 : dict
        A dictionary containing the mutation counts for sample 1. The keys represent mutation types 
        (e.g., 'C>A', 'C>T') and the values are dictionaries of sequence positions and their respective counts.
        
    sample2 : dict
        A dictionary containing the mutation counts for sample 2, structured similarly to `sample1`.
        
    sample3 : dict
        A dictionary containing the mutation counts for sample 3, structured similarly to `sample1`.
        
    name1 : str
        The name or label for sample 1 to be displayed above its corresponding plot.
        
    name2 : str
        The name or label for sample 2 to be displayed above its corresponding plot.
        
    name3 : str
        The name or label for sample 3 to be displayed above its corresponding plot.
        
    ymax : float
        The maximum value for the Y-axis, representing the highest mutation frequency across all samples. 
        This value is used to scale the Y-axis uniformly across all plots.

    Returns:
    --------
    None
        This function does not return any value. It generates a plot with three subplots displayed on the screen.
    
    Notes:
    ------
    - Each plot will display mutation types (e.g., 'C>A', 'C>T', etc.) with different colors for easy distinction.
    - Horizontal lines representing 2% and 5% mutation frequency thresholds are added to each plot.
    - The X-axis represents sequence positions (with 96 categories), and the Y-axis represents mutation counts 
      as percentages of the total mutation count in each sample.
    - The plot is rendered using Matplotlib and does not save to a file by default.

    Example Usage:
    --------------
    sample1 = {"C>A": {"AAG": 10, "CGA": 15}, "C>T": {"CAG": 5}}
    sample2 = {"T>G": {"TAA": 20}, "C>A": {"CGA": 7}}
    sample3 = {"T>C": {"TAC": 3}, "C>G": {"CGA": 10}}

    plot_3samples_SBS96(sample1, sample2, sample3, "Sample 1", "Sample 2", "Sample 3", 30)
    """
    plt.rcParams['axes.linewidth'] = 2
    plot1 = plt.figure(figsize=(43.93,18))
    plt.rc('axes', edgecolor='lightgray')
    panel1 = plot1.add_axes([0.04, 0.64, 0.95, 0.25])
    xlabels = []
    x = 0.4
    #ymax = 0
    colors = [[3/256,189/256,239/256], [1/256,1/256,1/256],[228/256,41/256,38/256], [203/256,202/256,202/256], [162/256,207/256,99/256], [236/256,199/256,197/256]]
    i = 0

    for key in sample1:
        for seq in sample1[key]:
            xlabels.append(seq[0]+seq[2]+seq[6])
            panel1.bar(x, sample1[key][seq],width=0.4,color=colors[i],align='center', zorder=1000)
            '''if sample1[key][seq] > ymax:
                    ymax = sample1[key][seq]'''
            x += 1
        i += 1
        
    x = .043
    y3 = .9
    y = ymax #*1.25
    y2 = y+2
    for i in range(0, 6, 1):
        panel1.add_patch(plt.Rectangle((x,y3), .15, .05, facecolor=colors[i], clip_on=False, transform=plt.gcf().transFigure)) 
        x += .159

    yText = y3 + .06
    panel1.text(.1, yText, 'C>A', fontsize=55, fontweight='bold', fontname='Arial', transform=plt.gcf().transFigure)
    panel1.text(.255, yText, 'C>G', fontsize=55, fontweight='bold', fontname='Arial', transform=plt.gcf().transFigure)
    panel1.text(.415, yText, 'C>T', fontsize=55, fontweight='bold', fontname='Arial', transform=plt.gcf().transFigure)
    panel1.text(.575, yText, 'T>A', fontsize=55, fontweight='bold', fontname='Arial', transform=plt.gcf().transFigure)
    panel1.text(.735, yText, 'T>C', fontsize=55, fontweight='bold', fontname='Arial', transform=plt.gcf().transFigure)
    panel1.text(.89, yText, 'T>G', fontsize=55, fontweight='bold', fontname='Arial', transform=plt.gcf().transFigure)

    panel1.text(0.05, 0.85, name1, fontsize=40, weight="bold", color="black", fontname="Arial", transform=plt.gcf().transFigure)

    ytick_offest = y/4 #int(y/4)
    ylabs = [0, ytick_offest, ytick_offest*2, ytick_offest*3, ytick_offest*4]
    ylabels = [f"{round(x*100, 1)}%" for x in [0, ytick_offest, ytick_offest*2, ytick_offest*3, ytick_offest*4]]

    labs = np.arange(0.375,96.375,1)

    panel1.set_xlim([0, 96])
    panel1.set_ylim([0, y])
    #panel1.set_xticks(labs)
    panel1.set_yticks(ylabs)
    count = 0
    m = 0

    panel1.set_yticklabels(ylabels, fontsize=30)
    panel1.yaxis.grid(True)
    panel1.grid(which='major', axis='y', color=[0.93,0.93,0.93], zorder=1)
    #panel1.set_xlabel('')
    panel1.set_ylabel('')

    #plt.ylabel("Mutation Counts", fontsize=35, fontname="Times New Roman", weight = 'bold')

    panel1.tick_params(axis='both',which='both',\
                    bottom=False, labelbottom=False,\
                    left=True, labelleft=True,\
                    right=True, labelright=False,\
                    top=False, labeltop=False,\
                    direction='in', length=25, colors='lightgray', width=2)

    [i.set_color("black") for i in panel1.get_yticklabels()]

    ###### 2nd plot
    panel2 = plot1.add_axes([0.04, 0.35, 0.95, 0.25])
    xlabels = []
    x = 0.4
    #ymax = 0
    i = 0

    for key in sample2:
        for seq in sample2[key]:
            xlabels.append(seq[0]+seq[2]+seq[6])
            panel2.bar(x, sample2[key][seq], width=0.4, color=colors[i], align='center', zorder=1000)
            x += 1
        i += 1

    ylabs = [0, ytick_offest, ytick_offest*2, ytick_offest*3, ytick_offest*4]
    ylabels = [f"{round(x*100, 1)}%" for x in [0, ytick_offest, ytick_offest*2, ytick_offest*3, ytick_offest*4]]

    panel2.set_xlim([0, 96])
    panel2.set_ylim([0, y])
    panel2.set_yticks(ylabs)
    panel2.set_yticklabels(ylabels, fontsize=30)

    panel2.yaxis.grid(True)
    panel2.grid(which='major', axis='y', color=[0.93,0.93,0.93], zorder=1)
    panel2.set_xlabel('')
    panel2.set_ylabel('')
    
    panel2.text(0.05, 0.56, name2, fontsize=40, weight="bold", color="black", fontname="Arial", transform=plt.gcf().transFigure)

    panel2.tick_params(axis='both',which='both',\
                    bottom=False, labelbottom=False,\
                    left=True, labelleft=True,\
                    right=True, labelright=False,\
                    top=False, labeltop=False,\
                    direction='in', length=25, colors='lightgray', width=2)

    [i.set_color("black") for i in panel2.get_yticklabels()]

    ###### 3rd plot
    panel3 = plot1.add_axes([0.04, 0.06, 0.95, 0.25])

    xlabels = []
    x = 0.4
    i = 0

    for key in sample3:
        for seq in sample3[key]:
            xlabels.append(seq[0]+seq[2]+seq[6])
            panel3.bar(x, sample3[key][seq], width=0.4, color=colors[i], align='center', zorder=1000)
            x += 1
        i += 1

    ylabs = [0, ytick_offest, ytick_offest*2, ytick_offest*3, ytick_offest*4]
    ylabels = [f"{round(x*100, 1)}%" for x in [0, ytick_offest, ytick_offest*2, ytick_offest*3, ytick_offest*4]]

    panel3.set_xlim([0, 96])
    panel3.set_ylim([0, y])
    panel3.set_yticks(ylabs)
    panel3.set_yticklabels(ylabels, fontsize=30)

    panel3.yaxis.grid(True)
    panel3.grid(which='major', axis='y', color=[0.93,0.93,0.93], zorder=1)
    panel3.set_xlabel('')
    panel3.set_ylabel('')
    
    panel3.text(0.05, 0.27, name3, fontsize=40, weight="bold", color="black", fontname="Arial", transform=plt.gcf().transFigure)

    panel3.tick_params(axis='both',which='both',\
                    bottom=False, labelbottom=False,\
                    left=True, labelleft=True,\
                    right=True, labelright=False,\
                    top=False, labeltop=False,\
                    direction='in', length=25, colors='lightgray', width=2)

    [i.set_color("black") for i in panel3.get_yticklabels()]

    for i in range (0, 96, 1):
        panel3.text(i/101 + .0415, 0, xlabels[i][0], fontsize=30, color='gray', rotation='vertical', verticalalignment='center', fontname='Courier New', transform=plt.gcf().transFigure)
        panel3.text(i/101 + .0415, .02, xlabels[i][1], fontsize=30, color=colors[m], rotation='vertical', verticalalignment='center', fontname='Courier New', fontweight='bold',transform=plt.gcf().transFigure)
        panel3.text(i/101 + .0415, .04, xlabels[i][2], fontsize=30, color='gray', rotation='vertical', verticalalignment='center', fontname='Courier New', transform=plt.gcf().transFigure)
        count += 1
        if count == 16:
            count = 0
            m += 1

    panel1.axhline(y=0.02,c="blue",linewidth=1,zorder=0)
    panel2.axhline(y=0.02,c="blue",linewidth=1,zorder=0)
    panel3.axhline(y=0.02,c="blue",linewidth=1,zorder=0)
    panel1.text(0.9, .12, '2%', fontsize=30, horizontalalignment='center', transform=panel1.transAxes)
    panel2.text(0.9, .12, '2%', fontsize=30, horizontalalignment='center', transform=panel2.transAxes)
    panel3.text(0.9, .12, '2%', fontsize=30, horizontalalignment='center', transform=panel3.transAxes)


    panel1.axhline(y=0.05,c="green",linewidth=1,zorder=0)
    panel2.axhline(y=0.05,c="green",linewidth=1,zorder=0)
    panel3.axhline(y=0.05,c="green",linewidth=1,zorder=0)
    panel1.text(0.9, .3, '5%', fontsize=30, horizontalalignment='center', transform=panel1.transAxes)
    panel2.text(0.9, .3, '5%', fontsize=30, horizontalalignment='center', transform=panel2.transAxes)
    panel3.text(0.9, .3, '5%', fontsize=30, horizontalalignment='center', transform=panel3.transAxes)

    plt.show()




## Adapted from https://github.com/AlexandrovLab/SigProfilerPlotting
def plotTMB_clustered(inputDF, scale, order=[], Yrange = "adapt", cutoff = 0,
            redbar = "median", yaxis = "Somatic Mutations per Megabase",
            ascend = True, leftm = 1, rightm = 0.3, topm = 1.4, bottomm = 1):
    """
    Plots a clustered bar plot for Tumor Mutational Burden (TMB) using data from a dataframe containing 
    mutation counts for different mutation types. The plot visualizes TMB as log-transformed values 
    with custom scales, grouping, and thresholds. It also provides customization options for plot appearance.

    Parameters:
    -----------
    inputDF : pandas.DataFrame
        A DataFrame with two columns:
        - 'Types': Mutation types (e.g., SBS1, SBS2).
        - 'Mut_burden': The mutation burden (mutation count) for each mutation type.
        
    scale : int, str
        Defines the scale for normalizing mutation burden. Can be:
        - "genome" for genome scale (2897.310462).
        - "exome" for exome scale (55).
        - An integer representing a custom scale value.
        
    order : list, optional, default: []
        A list specifying the order of mutation types for plotting. If empty, mutations are ordered by 
        the median log-transformed mutation burden.
        
    Yrange : str or list, optional, default: "adapt"
        Defines the Y-axis range:
        - "adapt": Automatically adjusts based on the data.
        - "cancer": Fixed Y-axis range for cancer mutation burden (-3 to 3).
        - A list containing two numbers specifying the lower and upper bounds of the Y-axis in log scale.
        
    cutoff : float, optional, default: 0
        A minimum threshold for the mutation burden. Any mutation type with a burden below this cutoff 
        is excluded from the plot.
        
    output : str, optional, default: "TMB_plot.png"
        The file name for saving the generated plot. If not specified, the plot is saved with this default name.
        
    redbar : str, optional, default: "median"
        Determines whether the red reference line represents the 'mean' or 'median' of mutation burdens 
        for each mutation type.
        
    yaxis : str, optional, default: "Somatic Mutations per Megabase"
        Label for the Y-axis.
        
    ascend : bool, optional, default: True
        If True, sorts the mutation types in ascending order based on the mean/median log-transformed mutation burden.
        If False, sorts in descending order.
        
    leftm : float, optional, default: 1
        Left margin of the plot, used for adjusting layout.
        
    rightm : float, optional, default: 0.3
        Right margin of the plot, used for adjusting layout.
        
    topm : float, optional, default: 1.4
        Top margin of the plot, used for adjusting layout.
        
    bottomm : float, optional, default: 1
        Bottom margin of the plot, used for adjusting layout.

    Returns:
    --------
    None
        This function does not return any value. It generates a plot.
    
    Notes:
    ------
    - The plot includes scatter points for each mutation type's log-transformed mutation burden.
    - A horizontal red bar is drawn for each mutation type to indicate its mean or median mutation burden.
    - The plot includes color-coded rectangles for certain mutation types (SBS1, SBS2, etc.).
    - Customizable margins and axis range options are provided to adjust the plot's appearance.

    Example Usage:
    --------------
    import pandas as pd

    # Example data
    data = {'Types': ['SBS1', 'SBS2', 'SBS5', 'SBS13', 'SBS31'],
            'Mut_burden': [200, 150, 500, 600, 300]}
    df = pd.DataFrame(data)

    # Plotting TMB
    plotTMB_clustered(df, scale="genome", Yrange="adapt", cutoff=50, output="TMB_plot_example.png")
    """
    if type(scale) == int:
        scale = scale
    elif scale == "genome":
        scale  = 2897.310462
    elif scale == "exome":
        scale = 55
    else:
        print("Please input valid scale values: \"exome\", \"genome\" or a numeric value")
        return

    inputDF.columns = ['Types', 'Mut_burden']
    df=inputDF[inputDF["Mut_burden"] > cutoff].copy()
    df['log10BURDENpMB'] = df.apply(lambda row: np.log10(row.Mut_burden/scale), axis = 1)
    groups = df.groupby(["Types"])
    if redbar == "mean":
        redbars = groups.mean()["log10BURDENpMB"].sort_values(ascending=ascend)
        names = groups.mean()["log10BURDENpMB"].sort_values(ascending=ascend).index
    elif redbar == "median":
        if len(order) == 0:
            redbars = groups.median()["log10BURDENpMB"].sort_values(ascending=ascend)
            names = groups.median()["log10BURDENpMB"].sort_values(ascending=ascend).index
        else:
            redbars = groups.median()["log10BURDENpMB"]
            redbars = redbars.loc[order]
            names = order
    else:
        print("ERROR: redbar parameter must be either mean or median")
        return
    counts = groups.count()["log10BURDENpMB"][names]
    ngroups = groups.ngroups
    #second row of bottom label
    input_groups = inputDF.groupby(["Types"])
    input_counts = input_groups.count()["Mut_burden"][names]
    list1 = counts.to_list()
    list2 = input_counts.to_list()
    str1 = ''
    list3 = prepend(list1, str1)
    str2 = '\n'
    list4 = prepend(list2, str2)
    result = [None]*(len(list3)+len(list4))
    result[::2] = list3
    result[1::2] = list4
    tick_labels = result
    new_labels = [ ''.join(x) for x in zip(tick_labels[0::2], tick_labels[1::2]) ]
    if Yrange == "adapt":
        ymax = math.ceil(df['log10BURDENpMB'].max())
        ymin = math.floor(df['log10BURDENpMB'].min())
    elif Yrange == "cancer":
        ymax = 3
        ymin = -3
    elif type(Yrange) == list:
        print("Yrange is a list")
        ymax = int(math.log10(Yrange[1]))
        ymin = int(math.log10(Yrange[0]))
    else:
        print("ERROR:Please input valid scale values: \"adapt\", \"cancer\" or a list of two power of 10 numbers")
        return
    #plotting
    if ngroups < 7:
        rightm = rightm + 0.4 * (7 - ngroups)
    if len(names[0])>13:
        leftm = leftm + 0.09 * (len(names[0]) - 13)
        topm = topm + 0.080 * (len(names[0]) - 13)
    fig_width = leftm + rightm + 0.4 * ngroups
    fig_length = 6
    fig, ax = plt.subplots(figsize=(fig_width, fig_length))
    if cutoff < 0:
        print("ERROR: cutoff value is less than 0")
        return
    plt.xlim(0,2*ngroups)
    plt.ylim(ymin,ymax)
    yticks_loc = range(ymin,ymax+1,1)
    plt.yticks(yticks_loc,list(map((lambda x: 10**x), list(yticks_loc)))) 
    plt.xticks(np.arange(1, 2*ngroups+1, step = 2),new_labels) 
    plt.tick_params(axis = 'both', which = 'both',length = 0)
    plt.hlines(yticks_loc,0,2*ngroups,colors = 'black',linestyles = "dashed",linewidth = 0.5,zorder = 1)

    greybar = 0
    for i in range(len(names)):
        if names[i] in ['SBS1', 'SBS5']:
            rectangle = mpatches.Rectangle([(i)*2, ymin], 2, ymax-ymin, color = "lavender",zorder = 0)
            ax.add_patch(rectangle)
        elif names[i] in ['SBS2', 'SBS13']:
            rectangle = mpatches.Rectangle([(i)*2, ymin], 2, ymax-ymin, color = "lightsteelblue",zorder = 0)
            ax.add_patch(rectangle)
        elif names[i] in ['SBS31', 'SBS35']:
            rectangle = mpatches.Rectangle([(i)*2, ymin], 2, ymax-ymin, color = "mistyrose",zorder = 0)
            ax.add_patch(rectangle)
        elif greybar:
            rectangle = mpatches.Rectangle([(i)*2, ymin], 2, ymax-ymin, color = "lightgrey",zorder = 0)
            ax.add_patch(rectangle)
            greybar = 0
        else:
            greybar = 0

    for i in range(0,ngroups,1):
        X_start = i*2+0.2
        X_end = i*2+2-0.2
        y_values = groups.get_group(names[i])["log10BURDENpMB"].sort_values(ascending = True).values.tolist()
        x_values = list(np.linspace(start = X_start, stop = X_end, num = counts[i]))
        plt.scatter(x_values,y_values,color = "black",s=1.5)
        plt.hlines(redbars[i], X_start, X_end, colors='red', zorder=2)
        plt.text((leftm + 0.2 + i * 0.4) / fig_width , 0.85 / fig_length , "___",  horizontalalignment='center',transform=plt.gcf().transFigure)
    plt.ylabel(yaxis)
    axes2 = ax.twiny()
    plt.tick_params(axis = 'both', which = 'both',length = 0)
    plt.xticks(np.arange(1, 2*ngroups+1, step = 2),names,rotation = -90,ha = 'right')
    fig.subplots_adjust(top = ((ymax - ymin) * 0.7 + bottomm) / fig_length, bottom = bottomm / fig_length, left = leftm / fig_width, right=1 - rightm / fig_width)




## Maybe needed
def plot_profile_SBS96(sample1, ymax, sample_name, outname):
    plt.rcParams['axes.linewidth'] = 2
    plot1 = plt.figure(figsize=(43.93,12))
    plt.rc('axes', edgecolor='lightgray')
    panel1 = plot1.add_axes([0.04, 0.491, 0.95, 0.4])
    xlabels = []
    x = 0.4
    #ymax = 0
    colors = [[3/256,189/256,239/256], [1/256,1/256,1/256],[228/256,41/256,38/256], [203/256,202/256,202/256], [162/256,207/256,99/256], [236/256,199/256,197/256]]
    i = 0

    for key in sample1:
        for seq in sample1[key]:
            xlabels.append(seq[0]+seq[2]+seq[6])
            plt.bar(x, sample1[key][seq],width=0.4,color=colors[i],align='center', zorder=1000)
            '''if sample1[key][seq] > ymax:
                    ymax = sample1[key][seq]'''
            x += 1
        i += 1
        
    x = .043
    y3 = .9
    y = ymax #*1.25
    y2 = y+2
    for i in range(0, 6, 1):
        panel1.add_patch(plt.Rectangle((x,y3), .15, .05, facecolor=colors[i], clip_on=False, transform=plt.gcf().transFigure)) 
        x += .159

    yText = y3 + .06
    panel1.text(.1, yText, 'C>A', fontsize=55, fontweight='bold', fontname='Arial', transform=plt.gcf().transFigure)
    panel1.text(.255, yText, 'C>G', fontsize=55, fontweight='bold', fontname='Arial', transform=plt.gcf().transFigure)
    panel1.text(.415, yText, 'C>T', fontsize=55, fontweight='bold', fontname='Arial', transform=plt.gcf().transFigure)
    panel1.text(.575, yText, 'T>A', fontsize=55, fontweight='bold', fontname='Arial', transform=plt.gcf().transFigure)
    panel1.text(.735, yText, 'T>C', fontsize=55, fontweight='bold', fontname='Arial', transform=plt.gcf().transFigure)
    panel1.text(.89, yText, 'T>G', fontsize=55, fontweight='bold', fontname='Arial', transform=plt.gcf().transFigure)

    plt.text(0.85, 0.8, sample_name, fontsize=40, weight="bold", color="black", fontname="Arial", transform=plt.gcf().transFigure)

    ytick_offest = y/4 #int(y/4)
    ylabs = [round(x, 3) for x in [0, ytick_offest, ytick_offest*2, ytick_offest*3, ytick_offest*4]]
    print(ylabs)
    ylabels= [f"{round(x*100, 1)}%" for x in [0, ytick_offest, ytick_offest*2, ytick_offest*3, ytick_offest*4]]

    labs = np.arange(0.375,96.375,1)

    panel1.set_xlim([0, 96])
    panel1.set_ylim([0, y])
    #panel1.set_xticks(labs)
    panel1.set_yticks(ylabs)
    count = 0
    m = 0

    for i in range (0, 96, 1):
        panel1.text(i/101 + .0415, .42, xlabels[i][0], fontsize=30, color='gray', rotation='vertical', verticalalignment='center', fontname='Courier New', transform=plt.gcf().transFigure)
        panel1.text(i/101 + .0415, .444, xlabels[i][1], fontsize=30, color=colors[m], rotation='vertical', verticalalignment='center', fontname='Courier New', fontweight='bold',transform=plt.gcf().transFigure)
        panel1.text(i/101 + .0415, .471, xlabels[i][2], fontsize=30, color='gray', rotation='vertical', verticalalignment='center', fontname='Courier New', transform=plt.gcf().transFigure)
        count += 1
        if count == 16:
            count = 0
            m += 1

    panel1.set_yticklabels(ylabels, fontsize=30)
    plt.gca().yaxis.grid(True)
    plt.gca().grid(which='major', axis='y', color=[0.93,0.93,0.93], zorder=1)
    #panel1.set_xlabel('')
    panel1.set_ylabel('')

    plt.ylabel("Countribution", fontsize=35, fontname="Times New Roman", weight = 'bold')

    panel1.tick_params(axis='both',which='both',\
                    bottom=False, labelbottom=False,\
                    left=True, labelleft=True,\
                    right=True, labelright=False,\
                    top=False, labeltop=False,\
                    direction='in', length=25, colors='lightgray', width=2)


    [i.set_color("black") for i in plt.gca().get_yticklabels()]

    plt.savefig(os.path.join(fig_out, outname+'.SBS96.pdf'), bbox_inches="tight")
    return plot1



def coefs_scatter(coefs, fname, positive_only=True, interactions='singles_only', ylab=''):
    """
    Creates a scatter plot to visualize the relationship between features, outcomes, and their corresponding 
    coefficients and AUROC scores. The plot can be customized to show only positive coefficients, specific 
    interaction terms, and displays annotations based on AUROC scores.

    Parameters:
    -----------
    coefs : pandas.DataFrame
        A DataFrame containing the following columns:
        - 'Outcome': The outcome variable or target.
        - 'Features': The feature names (can include interaction terms).
        - 'Coefficient': The coefficient value for the feature.
        - 'AUC': The AUROC score associated with the feature (used for color mapping).

    fname : str
        The filename where the plot will be saved (including file extension).

    positive_only : bool, optional, default: True
        If True, the plot will only display features with positive coefficients and AUROC scores greater than or equal to 0.65. 
        If False, features with any coefficient value greater than or equal to 0.65 will be displayed regardless of sign.

    interactions : str, optional, default: 'singles_only'
        Specifies which types of features to display:
        - 'singles_only': Only includes single features (no interaction terms).
        - 'x_only': Only includes features with interaction terms (indicated by a colon `:`).
        - 'all': Includes both single features and interaction terms.

    ylab : str, optional, default: ''
        The label for the y-axis. If not provided, the y-axis will remain unlabeled.

    Returns:
    --------
    None
        This function generates and saves a scatter plot to the specified `fname`. It does not return any value.

    Notes:
    ------
    - The size of the scatter points is scaled according to the absolute value of the coefficient.
    - The color of the scatter points represents the AUROC score of the feature.
    - Annotations are added for features with an AUROC score greater than 0.8, displaying the coefficient value.
    - Grid lines are removed for a cleaner presentation, and alternate rows are highlighted with light gray.
    - A legend is added if both positive and negative coefficients are shown.

    Example Usage:
    --------------
    import pandas as pd

    # Example data
    data = {
        'Outcome': ['Outcome1', 'Outcome2', 'Outcome1', 'Outcome3'],
        'Features': ['Feature1', 'Feature2', 'Feature3', 'Feature4'],
        'Coefficient': [0.5, -0.3, 0.7, -0.2],
        'AUC': [0.85, 0.9, 0.65, 0.75]
    }
    df = pd.DataFrame(data)

    # Plotting coefficients scatter plot
    coefs_scatter(df, fname='coefs_scatter_plot.png', positive_only=True, interactions='singles_only', ylab='Feature Coefficients')
    """
    if positive_only:
        data = coefs[(coefs.AUC>=0.65) & (coefs.Coefficient>0)]
    else:
        data = coefs[coefs.AUC>=0.65]

    # Filter out interaction terms if present (assuming interaction terms have a different format)
    # Assuming single terms don't contain ':' (common in interaction terms)
    if interactions=='singles_only':
        data = data[~data['Features'].str.contains(':')]
        fig_h = data['Features'].nunique()/1.2
    elif interactions=='x_only':
        data = data[data['Features'].str.contains(':')]
        fig_h = data['Features'].nunique()/2
    else:
        print('Plotting all interactions.')

    # Map coefficients to colors based on sign
    if positive_only:
        colors = data['Coefficient'].apply(lambda x: 'black' if x > 0 else 'red')
    else:
        colors = data['Coefficient'].apply(lambda x: 'blue' if x > 0 else 'red')

    # Plotting
    fig, ax = plt.subplots(figsize=(24, fig_h))
    scatter = ax.scatter(
        x=data['Outcome'],
        y=data['Features'],
        s=data['Coefficient'].abs() * 1000,  # Scale the coefficient for circle size
        c=data['AUC'],
        cmap='gist_gray',  # Color map based on AUC
        #alpha=0.7,  # Transparency
        zorder=2,
        #edgecolor=colors,  # Color edge based on coefficient sign
        linewidth=2
    )

    # Add annotations for each circle
    for i, row in data.iterrows():
        if row["AUC"] > 0.8:
            ax.annotate(
                f'{row["Coefficient"]:.2f}',  # Annotation text (formatted coefficient)
                (row['Outcome'], row['Features']),  # Position (x, y)
                ha='center',  # Horizontal alignment
                fontsize=11,  # Font size for the annotation
                color='black'  # Annotation text color
            )
        else:
            ax.annotate(
            f'{row["Coefficient"]:.2f}',  # Annotation text (formatted coefficient)
            (row['Outcome'], row['Features']),  # Position (x, y)
            ha='center',  # Horizontal alignment
            fontsize=11,  # Font size for the annotation
            color='white'  # Annotation text color
        )

    # Add color bar
    cbar = plt.colorbar(scatter)
    cbar.set_label('AUROC Score', rotation=270, labelpad=15, fontsize=14)

    # Set new tick labels from min to max
    min_val, max_val = data['AUC'].min(), data['AUC'].max()
    new_ticks = np.linspace(min_val, max_val, num=5)  # Adjust the number of ticks as needed
    cbar.set_ticks(new_ticks)
    cbar.set_ticklabels([f"{tick:.2f}" for tick in new_ticks])  # Format labels

    # Move x-axis labels to the top and rotate them vertically
    ax.tick_params(axis='x', which='both', bottom=False, top=True, labelbottom=False, labeltop=True)
    plt.xticks(rotation=90, verticalalignment='center')

    # Move x-axis labels to the top and rotate them vertically
    ax.tick_params(axis='x', which='both', bottom=False, top=True, labelbottom=False, labeltop=True)
    plt.xticks(rotation=90, verticalalignment='bottom', fontsize=18)
    plt.yticks(fontsize=18)

    # Add labels and title
    ax.set_xlabel('')
    ax.set_ylabel(ylab, fontsize=32)
    ax.set_title('')

    # Add legend for positive and negative coefficients
    if not positive_only:
        legend_elements = [
            plt.Line2D([0], [0], marker='o', color='w', label='Positive Coefficient', markerfacecolor='blue', markersize=10),
            plt.Line2D([0], [0], marker='o', color='w', label='Negative Coefficient', markerfacecolor='red', markersize=10)
        ]
        plt.legend(handles=legend_elements, loc='lower right')

    #plt.grid(True, linestyle='--', alpha=0.5)

    # Remove gridlines
    ax.grid(False)

    # Highlight every other row with a light gray background using Rectangle patches
    for i in range(len(data)):
        if i % 2 == 0:
            rect = patches.Rectangle(
                (-0.5, i - 0.5),  # (x, y) position
                len(data['Outcome']) - 1 + 1,  # width
                1,  # height
                linewidth=1,
                facecolor='lightgray',
                alpha=0.95
            )
            ax.add_patch(rect)


    #axes[0].add_patch(patches.Rectangle((-0.5,9.5),xlen,1,linewidth=1,facecolor="lightgray", alpha=0.95))

    ax.set_xlim(-0.5, data.Outcome.nunique() - 0.5)

    labels = ax.get_xticklabels()
    print(labels[0].get_text())
    for ind, label in enumerate(labels[::-1]):
        lab_name = label.get_text()
        if lab_name.startswith('SBS288') or lab_name.startswith('DBS78') or lab_name.startswith('ID83') or lab_name.startswith('SV32') or lab_name.startswith('CNV48'):
            label.set_color('darkgreen')
        else:
            label.set_color('black')

    plt.tight_layout()
    plt.savefig(fname, bbox_inches="tight")




def plot_coefs(coefs, title=None, leg=None, x_only=True):
    """
    Plots a horizontal bar chart of logistic regression coefficients for features and interaction terms, 
    highlighting their significance. The plot allows customization of the title, legend, and whether to 
    focus on interaction terms.

    Parameters:
    -----------
    coefs : pandas.DataFrame
        A DataFrame containing the following columns:
        - 'Features': The feature names (including interaction terms, indicated by a colon `:`).
        - 'Coefficient': The coefficient values for each feature.
        - 'Outcome': The outcome variable associated with the features.

    title : str, optional, default: None
        The title for the plot. If not provided, the plot will not have a title.

    leg : str, optional, default: None
        The location for the legend. If not provided, the legend will not be shown. Valid locations are
        'upper right', 'upper left', 'lower right', 'lower left', 'right', 'left', 'center', etc.

    x_only : bool, optional, default: True
        If True, the plot will only display interaction terms (features with a colon `:`). 
        If False, both main effects and interaction terms will be displayed.

    Returns:
    --------
    None
        The function generates and displays the plot, but does not return any value.

    Notes:
    ------
    - Features with a colon (`:`) in their names are treated as interaction terms.
    - The colors of the bars represent the type of feature:
        - 'steelblue' for main effects
        - 'thistle' for interaction terms.
    - Vertical lines at x=0 (gray) and x=±0.7 (green and red) are drawn to highlight significant coefficients.
    - The y-axis is inverted to display features in descending order of their coefficients.
    - The outcome groups are separated by dashed horizontal lines, with each group labeled.
    - A dynamically positioned legend is added if `leg` is provided.

    Example Usage:
    --------------
    import pandas as pd

    # Example data
    data = {
        'Features': ['Feature1', 'Feature2', 'Feature3', 'Feature1:Feature2'],
        'Coefficient': [0.5, -0.3, 0.7, -0.2],
        'Outcome': ['Outcome1', 'Outcome1', 'Outcome2', 'Outcome2']
    }
    df = pd.DataFrame(data)

    # Plotting coefficients
    plot_coefs(df, title='Logistic Regression Coefficients', leg='upper right', x_only=True)
    """
    # Plot the non-zero coefficients
    denom = 3 if len(coefs) > 7 else 2
    fig, ax = plt.subplots(figsize=(10, len(coefs)/denom))
    if x_only:
        coefs = coefs[coefs['Features'].str.contains(':')]
        colors = ['steelblue' if ':' in feature else 'thistle' for feature in coefs.Features]
    else:
        colors = ['thistle' if ':' in feature else 'steelblue' for feature in coefs.Features]
    coefs = coefs.reset_index(drop=True)
    

    p1 = plt.barh(coefs.index, coefs.Coefficient, color=colors)
    plt.xlabel('Logistic Regression Coefficients', fontsize=18)
    plt.gca().invert_yaxis() # Invert y-axis for descending order
    if title:
        plt.title(title, fontsize=20)

    ax.set_yticks(range(len(coefs.Features)))
    new_labels = [x.replace('_', ' ') for x in coefs.Features.tolist()]
    new_labels = [x.replace(':', '::') for x in new_labels]
    ax.set_yticklabels(new_labels, fontsize=14)

    for color, tick in zip(colors, ax.get_yticklabels()):
        tick.set_color(color) #set the color property

    plt.xticks(fontsize=14)

    xmin, xmax = ax.get_xlim()

    # vertical lines
    lcount = 0
    outc = coefs.Outcome.iloc[0]
    ax.text(xmax, 0, f"{outc}", va='center', fontsize=14)
    for out in coefs.Outcome:
        if out == outc:
            lcount += 1
        else:
            ax.hlines(y=lcount-0.5, xmin=xmin, xmax=xmax, colors=['tab:gray'], ls='--', lw=2)
            ax.text(xmax, lcount, f"{out}", va='center', fontsize=14)
            outc = out
            lcount += 1

    #ax.vline(x=0, colors=['tab:gray'], ls='-', lw=2)
    plt.axvline(x = 0, color = 'gray', lw=2)
    plt.axvline(x = 0.7, color = 'green', ls='--', lw=1)
    plt.axvline(x = -0.7, color = 'red', ls='--', lw=1)

    ax.spines[['top', 'right']].set_visible(False)

    plt.ylim(len(coefs)-0.5, -0.5)

    # Add a dynamically positioned legend
    if leg:
        plt.legend(handles=[
            plt.Line2D([0], [0], color='steelblue', lw=4, label='Main Effects'),
            plt.Line2D([0], [0], color='thistle', lw=4, label='Interaction Terms')
        ], loc=leg)

    plt.tight_layout()
    plt.show()




def plot_bipartite_subs(df):
    """
    Plots a series of bipartite network graphs, where each graph represents interactions 
    between a central node and a set of outer nodes, based on the provided data frame.

    Each row in the DataFrame represents a central node with associated outer nodes 
    (columns). The function creates one plot per row, showing the central node in the 
    middle with edges to the outer nodes based on nonzero weights.

    Parameters:
    -----------
    df : pandas.DataFrame
        A DataFrame where each row corresponds to a central node and its associated outer 
        nodes (columns). The values in the DataFrame represent the weights of edges 
        between the central node and the outer nodes. Nonzero values indicate the presence 
        of edges, and the weight of the edge.

    Returns:
    --------
    None
        The function generates and displays a set of bipartite network plots, with each plot 
        representing a central node and its interactions with outer nodes. The plots are arranged 
        in a grid with a maximum of 3 columns per row.

    Notes:
    ------
    - Each central node is represented by a pink node at the center of the plot.
    - Outer nodes are shown in light steel blue color and are connected to the central node 
      by edges weighted according to the values in the DataFrame.
    - The plots are arranged in a grid format with a maximum of 3 columns per row. If there 
      are more rows than can fit in the available columns, additional rows are created.
    - Any extra unused subplots are hidden.
    - The layout is adjusted for optimal display of the plots.

    Example Usage:
    --------------
    import pandas as pd

    # Example DataFrame with central nodes and outer nodes (edges)
    data = {
        'NodeA': [0, 2, 0],
        'NodeB': [1, 0, 0],
        'NodeC': [0, 0, 3],
        'NodeD': [0, 1, 0]
    }
    df = pd.DataFrame(data, index=['Central1', 'Central2', 'Central3'])

    # Plot bipartite networks
    plot_bipartite_subs(df)
    """
    # Get number of subplots
    n_rows = len(df)
    #n_cols = 2 if n_rows > 1 else 1  # Arrange in 2 columns if multiple rows
    n_cols = 3
    total_plots = (n_rows // n_cols) + (n_rows % n_cols)

    fig, axes = plt.subplots(nrows=total_plots, ncols=n_cols, figsize=(18, 6 * total_plots))
    axes = np.ravel(axes)  # Flatten the axes for easy iteration

    # Loop through each row and create a network plot
    for i, (central_node, row) in enumerate(df.iterrows()):
        ax = axes[i]
        G = nx.Graph()

        # Add the central node (pink)
        G.add_node(central_node, color='thistle')

        # Add outer nodes (blue) and edges based on nonzero values
        edges = []
        for node, weight in row.items():
            if weight > 0:  # Ignore zero-weight edges
                G.add_node(node, color='lightsteelblue')
                edges.append((central_node, node, weight))

        G.add_weighted_edges_from(edges)

        # Get positions
        pos = nx.circular_layout(G, scale=1.5)
        pos[central_node] = np.array([0, 0])  # Center the central node

        # Get node colors
        node_colors = [G.nodes[n]['color'] for n in G.nodes()]

        # Draw the network
        nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=1000, ax=ax)
        nx.draw_networkx_edges(G, pos, edgelist=[(u, v) for u, v, _ in edges],
                                width=[w for _, _, w in edges], edge_color='gray', ax=ax)
        nx.draw_networkx_labels(G, pos, font_size=14, font_color='black', ax=ax)

        # Set title
        ax.set_title(f"{central_node} Interactions", fontsize=20)

        ax.axis("off")  # Hide axes

    # Hide any extra unused subplots
    for j in range(i + 1, len(axes)):  
        axes[j].set_visible(False)

    #plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1, wspace=0.4, hspace=0.5)


    # Alternative: Use constrained layout 
    #fig.set_constrained_layout(True)

    # Adjust layout
    plt.tight_layout()


def plot_driver_nx(df2, figout):
    """
    Plots a bipartite network graph from a DataFrame, where rows and columns represent two sets 
    of nodes, and edges represent the relationships between them based on non-null values in the 
    DataFrame. The graph visualizes the relationships with different edge thicknesses and colors 
    based on the values in the DataFrame.

    Parameters:
    -----------
    df2 : pandas.DataFrame
        A DataFrame where the rows and columns represent two distinct sets of nodes. The values in 
        the DataFrame represent the weights of the edges between the row and column nodes. Only non-null 
        values are considered for drawing edges, with negative values represented in red and positive values 
        in blue.

    figout : str
        The file path where the resulting bipartite network graph image will be saved.

    Returns:
    --------
    None
        The function generates and saves a bipartite network graph to the specified file path.
        The graph contains nodes for both rows and columns, with edges drawn between them based on 
        the values in the DataFrame.

    Notes:
    ------
    - Row nodes are positioned in an outer circle, while column nodes are positioned in an inner circle.
    - Edge color and thickness are based on the sign and magnitude of the values in the DataFrame, 
      respectively.
    - Negative values result in red edges, positive values in blue edges.
    - Node colors distinguish row nodes (skyblue) and column nodes (lightcoral).
    - The graph is saved as an image file to the location specified by `figout`.

    Example Usage:
    --------------
    import pandas as pd

    # Example DataFrame with rows and columns as node sets and non-null values representing edge weights
    data = {
        'A': [0.5, -0.3, 1.2],
        'B': [0.8, None, -0.7],
        'C': [None, 0.4, -1.1]
    }
    df = pd.DataFrame(data, index=['Node1', 'Node2', 'Node3'])

    # Save bipartite network plot to a file
    plot_driver_nx(df, 'network_plot.png')
    """
    # Create a network graph
    G = nx.Graph()

    # Define the two sets of nodes
    rows = df2.index
    cols = df2.columns

    # Add nodes: rows and columns as two sets
    G.add_nodes_from(rows, bipartite=0)  # Add row nodes
    G.add_nodes_from(cols, bipartite=1)  # Add column nodes

    # Add edges with attributes for weights and colors
    for row in rows:
        for col in cols:
            value = df2.at[row, col]
            if pd.notnull(value):  # Only include non-null values
                G.add_edge(row, col, weight=abs(value), color='red' if value < 0 else 'blue')

    # Number of row and column nodes
    num_rows = len(rows)
    num_cols = len(cols)

    # Position column nodes in the inner circle (radius = 1)
    theta_cols = np.linspace(0, 2 * np.pi, num_cols, endpoint=False)
    pos_cols = {cols[i]: (np.cos(theta_cols[i]), np.sin(theta_cols[i])) for i in range(num_cols)}

    # Position row nodes in the outer circle (radius = 2)
    theta_rows = np.linspace(0, 2 * np.pi, num_rows, endpoint=False)
    pos_rows = {rows[i]: (2 * np.cos(theta_rows[i]), 2 * np.sin(theta_rows[i])) for i in range(num_rows)}

    # Combine the positions of rows and columns
    pos = {**pos_cols, **pos_rows}

    # Extract edge attributes for drawing
    edge_colors = nx.get_edge_attributes(G, 'color').values()
    edge_widths = [d['weight'] for u, v, d in G.edges(data=True)]  # Scale edge thickness

    # Draw the network graph
    plt.figure(figsize=(10, 10))

    # Draw nodes with different colors for rows and columns
    nx.draw_networkx_nodes(G, pos, nodelist=rows, node_color="skyblue", node_size=1500, label="Row Nodes")
    nx.draw_networkx_nodes(G, pos, nodelist=cols, node_color="lightcoral", node_size=1500, label="Column Nodes")

    # Draw edges with thickness and color
    nx.draw_networkx_edges(G, pos, width=edge_widths, edge_color=edge_colors)

    # Add labels
    nx.draw_networkx_labels(G, pos, font_size=12, font_color="black")

    # Add legend for node groups
    #plt.legend(loc="lower left", title="Node Groups", labels=["Row Nodes", "Column Nodes"])

    # Add a title and show the graph
    plt.title("")
    plt.axis("off")
    plt.savefig(figout, bbox_inches="tight")




def build_cm(cols):
    """
    Creates a custom colormap from a list of RGB tuples.

    This function constructs a colormap by interpolating between the colors defined in 
    the `cols` parameter, along with a default starting color of white (RGB: (1, 1, 1)).

    Parameters:
    -----------
    cols : list of tuples
        A list of tuples where each tuple represents an RGB color value, with each 
        component in the range of 0 to 255. The list should contain at least one color.
        Example: [(255, 0, 0), (0, 255, 0), (0, 0, 255)]

    Returns:
    --------
    matplotlib.colors.LinearSegmentedColormap
        A LinearSegmentedColormap object, which can be used for generating color maps 
        for visualizations in Matplotlib.

    Notes:
    ------
    - The first color in the colormap is always white (RGB: 1, 1, 1).
    - The colormap is discretized into 10 bins (`n_bins`), but this can be adjusted.
    - The resulting colormap is named 'my_list'.
    
    Example Usage:
    --------------
    cols = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
    cm = build_cm(cols)
    plt.imshow(data, cmap=cm)
    plt.colorbar()
    """
    colors = [(1, 1, 1)]
    for tup in cols:
        colors.append(tuple([x/256 for x in tup]))
    n_bins = 10  # Discretizes the interpolation into bins
    cmap_name = 'my_list'
    return mc.LinearSegmentedColormap.from_list(cmap_name, colors, N=n_bins)



def plot_sig_counts(sig_counts, outname):
    """
    Plots a heatmap for each column in a DataFrame representing signature counts.
    Each heatmap shows the counts of different signatures across the rows, with the
    values annotated on the plot. The function saves the resulting plot to the specified 
    output file.

    Parameters:
    -----------
    sig_counts : pandas.DataFrame
        A DataFrame where rows represent different signatures and columns represent 
        different conditions or groups. The values in the DataFrame are integer counts 
        of each signature for each condition/group.

    outname : str
        The name of the output file where the plot will be saved. This file will be saved 
        in the current working directory or the specified path.

    Returns:
    --------
    None
        The function generates a heatmap plot for each column in `sig_counts` and saves it 
        to the file specified by `outname`. No value is returned.

    Notes:
    ------
    - The function generates one heatmap per column in `sig_counts`.
    - The color map (`mycmap`) used for the heatmaps is dynamically selected based on 
      the number of columns in `sig_counts`. If there are fewer than 3 columns, a subset 
      of colors is used. Otherwise, all colors from `cmaps` are applied.
    - Annotations displaying the counts are included in the heatmap.
    - The y-axis labels (signature names) are shown on the leftmost plot and removed 
      for the rest of the plots.

    Example Usage:
    --------------
    import pandas as pd

    # Example DataFrame of signature counts
    data = {'Condition1': [5, 2, 3], 'Condition2': [8, 3, 6], 'Condition3': [4, 1, 5]}
    sig_counts = pd.DataFrame(data, index=['SignatureA', 'SignatureB', 'SignatureC'])

    # Save the plot to a file
    plot_sig_counts(sig_counts, 'signature_heatmaps.png')
    """
    cm_low = build_cm([(70,130,180)])
    cm_hi = build_cm([(255,145,164)])
    cm_mid = build_cm([(143,188,143)])
    cmaps = [cm_low, cm_mid, cm_hi]

    w = cols = sig_counts.shape[1]
    h = sig_counts.shape[0]/4    
    mycmap = cmaps[0:1] + cmaps[2:3] if cols < 3 else cmaps

    fig, axes = plt.subplots(1,cols, figsize=(w, h), constrained_layout=True)

    for ax_i in range(len(axes)):
        ax = sns.heatmap(sig_counts.iloc[:,ax_i:ax_i+1], annot=True, fmt="d", cmap=mycmap[ax_i], cbar=False, ax=axes[ax_i])
        if ax_i == 0:
            ax.set_yticklabels(sig_counts.index.tolist(), rotation=0, ha='right')
        else:
            ax.axes.get_yaxis().set_ticks([])
        ax.axes.get_xaxis().set_ticks([])

    plt.savefig(outname, bbox_inches="tight")