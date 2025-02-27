import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.patches as mpatches
import matplotlib.axis as axis
from collections import OrderedDict

from scipy import stats
from scipy.stats import mannwhitneyu

GENOME_SIZE = 2897.310462

A_col = "#3288BD"
B_col = "#D53E4F"

def get_mut_dict(mat_df):
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

## plotTMB_generic_v4 --> plotTMB_therapy
def plotTMB_therapy(inputDF, pv, scale, color_dict, order=[], Yrange = "adapt", cutoff = 0, output = "TMB_plot.png",
            redbar = "median", xaxis = "Samples (n)", yaxis = "Somatic Mutations per Megabase",
            ascend = True, leftm = 1, rightm = 0.3, topm = 1.4, bottomm = 1):
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
    #fig_length = topm + bottomm + (ymax - ymin) * 0.7
    #print("{} -- {}".format(fig_width, fig_length))
    #fig_width = 4
    fig_length = 4
    fig, ax = plt.subplots(figsize=(fig_width/2, 4))
    if cutoff < 0:
        print("ERROR: cutoff value is less than 0")
        return
    plt.xlim(0,2*ngroups)
    #print(len(names[0]))
    plt.ylim(ymin,ymax)
    yticks_loc = range(ymin,ymax+1,1)
    plt.yticks(yticks_loc,list(map((lambda x: 10**x), list(yticks_loc)))) 
    #list1 = [f"n={x}" for x in list1]
    plt.xticks(np.arange(1, 2*ngroups+1, step = 2), list1, fontsize=14) 
    plt.tick_params(axis = 'both', which = 'both', length = 0)
    plt.hlines(yticks_loc,0,2*ngroups,colors = 'black',linestyles = "dashed",linewidth = 0.5,zorder = 1)
    '''for i in range(0,ngroups,2):
        greystart = [(i)*2,ymin]
        rectangle = mpatches.Rectangle(greystart, 2, ymax-ymin, color = "lightgrey",zorder = 0)
        ax.add_patch(rectangle)'''
    
    bar_y = ymax-1
    if pv < 0.001:
        plt.plot([3,3,5,5], [bar_y, bar_y+.1, bar_y+.1, bar_y], lw=1.5, c='k')
        #plt.text(4, 1.2, f"p = {pv:.2}", ha='center', va='bottom', color='k')
        plt.text(4, bar_y+.2, "***", ha='center', va='bottom', color='k')
    elif pv < 0.01:
        plt.plot([3,3,5,5], [bar_y, bar_y+.1, bar_y+.1, bar_y], lw=1.5, c='k')
        #plt.text(4, 1.2, f"p = {pv:.2}", ha='center', va='bottom', color='k')
        plt.text(4, bar_y+.2, "**", ha='center', va='bottom', color='k')
    elif pv < 0.05:
        plt.plot([3,3,5,5], [bar_y, bar_y+.1, bar_y+.1, bar_y], lw=1.5, c='k')
        #plt.text(4, 1.2, f"p = {pv:.2}", ha='center', va='bottom', color='k')
        plt.text(4, bar_y+.2, "*", ha='center', va='bottom', color='k')
    
    for i in range(len(names)):
        rectangle = mpatches.Rectangle([(i)*2, ymin], 2, ymax-ymin, color=color_dict[names[i]], zorder = 0)
        ax.add_patch(rectangle)


    for i in range(0,ngroups,1):
        X_start = i*2+0.2
        X_end = i*2+2-0.2
        #rg = 1.8
        y_values = groups.get_group(names[i])["log10BURDENpMB"].sort_values(ascending = True).values.tolist()
        x_values = list(np.linspace(start = X_start, stop = X_end, num = counts[i]))
        plt.scatter(x_values,y_values,color = "darkgrey",s=1.5)
        plt.hlines(redbars[i], X_start, X_end, colors='darkred', zorder=2)
        #plt.text(ngroups*2+1, redbars[i], ("%.3f" % 10**redbars[i]), color='red')
        plt.text(X_start, redbars[i]+0.1, ("%.3f" % 10**redbars[i]), color='darkred', fontsize=12)
        
        #plt.text((leftm + 0.2 + i * 0.4) / fig_width , 0.85 / fig_length , "___",  horizontalalignment='center',transform=plt.gcf().transFigure)
    plt.ylabel(yaxis, fontsize=14)
    plt.xlabel(xaxis, fontsize=14)
    axes2 = ax.twiny()
    #plt.text((leftm - 0.3) / fig_width, 0.2 / fig_length, "*Showing samples with counts more than %d" % cutoff, transform=plt.gcf().transFigure) 
    plt.tick_params(axis = 'both', which = 'both',length = 0)
    plt.xticks(np.arange(1, 2*ngroups+1, step = 2),names,rotation = -90,ha = 'right', fontsize=14)
    #fig.subplots_adjust(top = ((ymax - ymin) * 0.7 + bottomm) / fig_length, bottom = bottomm / fig_length, left = leftm / fig_width, right=1 - rightm / fig_width)
    #plt.savefig(output)
    #plt.close()

## TMB_plotter.plotTMB_generic --> plotTMB_type
def plotTMB_type(inputDF, pval_dict, scale, order=[], Yrange = "adapt", cutoff = 0, output = "TMB_plot.png",
            redbar = "median", yaxis = "Somatic Mutations per Megabase",
            ascend = True, leftm = 1, rightm = 0.3, topm = 1.4, bottomm = 1):
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
    #new_labels = [ ''.join(x) for x in zip(tick_labels[0::2], tick_labels[1::2]) ]
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
    #print(len(names[0]))
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
        #rg = 1.8
        y_values = groups.get_group(names[i])["log10BURDENpMB"].sort_values(ascending = True).values.tolist()
        x_values = list(np.linspace(start = X_start, stop = X_end, num = counts[i]))
        plt.scatter(x_values,y_values,color = "black",s=1.5)
        plt.hlines(redbars[i], X_start, X_end, colors='red', zorder=2)
        #plt.text((leftm + 0.2 + i * 0.4) / fig_width , 0.85 / fig_length , "___",  horizontalalignment='center',transform=plt.gcf().transFigure)
    for i in range(1,ngroups,2):
        x_line = i*2+2
        plt.axvline(x_line, color='darkgray')
    plt.ylabel(yaxis)
    #axes2 = ax.twiny()
    #plt.text((leftm - 0.3) / fig_width, 0.2 / fig_length, "*Showing samples with counts more than %d" % cutoff, transform=plt.gcf().transFigure) 
    plt.tick_params(axis = 'both', which = 'both',length = 0)
    #plt.xticks(np.arange(1, 2*ngroups+1, step = 2),names,rotation = -35,ha = 'right')

    #new_names = list(set([x.split('::')[0] for x in names]))
    new_names = list(dict.fromkeys([x.split('::')[0] for x in names]))
    #axes2.set_xticks(np.arange(0.5,23.5), new_names, ha = 'center')
    '''myticks = [2, 6, 9.7, 13.5, 17.5, 21.5, 25.2, 29, 33, 37,
               40.5, 44.5, 48.5,
               52.5, 55.5, 59.5,
               64, 67.5, 70, 74, 78, 82, 86]'''
    #axes2.set_xticks(myticks, list(set(new_names)), ha = 'center')
    #axes2.set_xticks(list(np.arange(2, ax.get_xlim()[1]-2, step = 4))+[86], list(set(new_names)), ha = 'center')
    #axes2.set_xticks(np.arange(2, ax.get_xlim()[1], step = 4), list(set(new_names)), ha = 'center')
    #print(np.arange(2, ax.get_xlim()[1], step = 4)-np.arange(0, 1, step = 0.045))
    #print(np.arange(2, ax.get_xlim()[1], step = 4))

    for i, j in enumerate(np.arange(2, ax.get_xlim()[1], step = 4)):
        ax.text(j, 2.1, new_names[i], horizontalalignment='center')

    for i, n in enumerate(new_names):
        x1 = i*4 + 1
        x2 = i*4 + 3
        '''
        rvs1 = inputDF[inputDF.Types==f"{n}::Naive"].Mut_burden.tolist()
        rvs2 = inputDF[inputDF.Types==f"{n}::Treated"].Mut_burden.tolist()

        _, pv = mannwhitneyu(rvs1, rvs2, method="asymptotic")'''
        pv = pval_dict[n]
        if pv < 0.05:
            #plt.plot([x1,x1,x2,x2], [1, 1.1, 1.1, 1], lw=1.5, c='k')
            plt.text(x1+1, 1.2, f"p={pv:.2}", ha='center', va='bottom', color='k')
            print(n)
        

    fig.subplots_adjust(top = ((ymax - ymin) * 0.7 + bottomm) / fig_length, bottom = bottomm / fig_length, left = leftm / fig_width, right=1 - rightm / fig_width)
    #plt.savefig(output)
    #plt.close()

## TMB_plotter.plotTMB_generic_v2 --> plotTMB_generic
def plotTMB_generic(inputDF, scale, order=[], Yrange = "adapt", cutoff = 0, output = "TMB_plot.png",
            redbar = "median", yaxis = "Somatic Mutations per Megabase",
            ascend = True, leftm = 1, rightm = 0.3, topm = 1.4, bottomm = 1):
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
    #fig_length = topm + bottomm + (ymax - ymin) * 0.7
    #print("{} -- {}".format(fig_width, fig_length))
    #fig_width = 4
    fig_length = 6
    fig, ax = plt.subplots(figsize=(fig_width, fig_length))
    if cutoff < 0:
        print("ERROR: cutoff value is less than 0")
        return
    plt.xlim(0,2*ngroups)
    #print(len(names[0]))
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
        #rg = 1.8
        y_values = groups.get_group(names[i])["log10BURDENpMB"].sort_values(ascending = True).values.tolist()
        x_values = list(np.linspace(start = X_start, stop = X_end, num = counts[i]))
        plt.scatter(x_values,y_values,color = "black",s=1.5)
        plt.hlines(redbars[i], X_start, X_end, colors='red', zorder=2)
        plt.text((leftm + 0.2 + i * 0.4) / fig_width , 0.85 / fig_length , "___",  horizontalalignment='center',transform=plt.gcf().transFigure)
    plt.ylabel(yaxis)
    axes2 = ax.twiny()
    #plt.text((leftm - 0.3) / fig_width, 0.2 / fig_length, "*Showing samples with counts more than %d" % cutoff, transform=plt.gcf().transFigure) 
    plt.tick_params(axis = 'both', which = 'both',length = 0)
    plt.xticks(np.arange(1, 2*ngroups+1, step = 2),names,rotation = -90,ha = 'right')
    fig.subplots_adjust(top = ((ymax - ymin) * 0.7 + bottomm) / fig_length, bottom = bottomm / fig_length, left = leftm / fig_width, right=1 - rightm / fig_width)
    #plt.savefig(output)
    #plt.close()

def prepend(list, str): 
    str += '{0}'
    list = [str.format(i) for i in list] 
    return(list)

## plotTMB_generic_v2 --> plotTMB_SBS
def plotTMB_SBS(inputDF, kzm611_sbs_rel, scale, order=[], Yrange = "adapt", cutoff = 0, output = "TMB_plot.png",
            redbar = "median", yaxis = "Somatic Mutations per Megabase",
            ascend = True, leftm = 1, rightm = 0.3, topm = 1.4, bottomm = 1):
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
    #fig_length = topm + bottomm + (ymax - ymin) * 0.7
    #print("{} -- {}".format(fig_width, fig_length))
    #fig_width = 4
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
    #print(len(names[0]))
    ax2.set_ylim(ymin,ymax)
    yticks_loc = range(ymin,ymax+1,1)
    ax2.set_yticks(yticks_loc,list(map((lambda x: 10**x), list(yticks_loc))))
    ax2.set_xticks(np.arange(1, 2*ngroups+1, step = 2), new_labels) 
    ax2.tick_params(axis = 'both', which = 'both',length = 0)
    ax2.hlines(yticks_loc,0,2*ngroups,colors = 'black',linestyles = "dashed",linewidth = 0.5,zorder = 1)
    '''for i in range(0,ngroups,2):
        greystart = [(i)*2,ymin]
        rectangle = mpatches.Rectangle(greystart, 2, ymax-ymin, color = "lightgrey",zorder = 0)
        ax2.add_patch(rectangle)'''

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
        #rg = 1.8
        y_values = groups.get_group(names[i])["log10BURDENpMB"].sort_values(ascending = True).values.tolist()
        x_values = list(np.linspace(start = X_start, stop = X_end, num = counts[i]))
        ax2.scatter(x_values,y_values,color = "black",s=1.5)
        ax2.hlines(redbars[i], X_start, X_end, colors='red', zorder=2)
        ax2.text((leftm + 0.2 + i * 0.4) / fig_width , 0.85 / fig_length , "___",  horizontalalignment='center',transform=plt.gcf().transFigure)
    ax2.set_ylabel(yaxis)
    #axes2 = ax.twiny()
    #plt.text((leftm - 0.3) / fig_width, 0.2 / fig_length, "*Showing samples with counts more than %d" % cutoff, transform=plt.gcf().transFigure) 
    #plt.tick_params(axis = 'both', which = 'both',length = 0)
    #plt.xticks(np.arange(1, 2*ngroups+1, step = 2),names,rotation = -90,ha = 'right')
    fig.subplots_adjust(top = ((ymax - ymin) * 0.7 + bottomm) / fig_length, bottom = bottomm / fig_length, left = leftm / fig_width, right=1 - rightm / fig_width)
    #plt.tight_layout()
    #plt.savefig(output)
    #plt.close()

## plotDBS_generic_v2 --> plotTMB_DBS
def plotTMB_DBS(inputDF, kzm611_sbs_rel, scale, order=[], Yrange = "adapt", cutoff = 0, output = "TMB_plot.png",
            redbar = "median", yaxis = "Somatic Mutations per Megabase",
            ascend = True, leftm = 1, rightm = 0.3, topm = 1.4, bottomm = 1):
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
    #fig_length = topm + bottomm + (ymax - ymin) * 0.7
    #print("{} -- {}".format(fig_width, fig_length))
    #fig_width = 4
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
    '''for i in range(0,ngroups,2):
        greystart = [(i)*2,ymin]
        rectangle = mpatches.Rectangle(greystart, 2, ymax-ymin, color = "lightgrey",zorder = 0)
        ax2.add_patch(rectangle)'''
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
        #rg = 1.8
        y_values = groups.get_group(names[i])["log10BURDENpMB"].sort_values(ascending = True).values.tolist()
        x_values = list(np.linspace(start = X_start, stop = X_end, num = counts[i]))
        ax2.scatter(x_values,y_values,color = "black",s=1.5)
        ax2.hlines(redbars[i], X_start, X_end, colors='red', zorder=2)
        ax2.text((leftm + 0.2 + i * 0.4) / fig_width , 0.85 / fig_length , "___",  horizontalalignment='center',transform=plt.gcf().transFigure)
    ax2.set_ylabel(yaxis)
    #axes2 = ax.twiny()
    #plt.text((leftm - 0.3) / fig_width, 0.2 / fig_length, "*Showing samples with counts more than %d" % cutoff, transform=plt.gcf().transFigure) 
    #plt.tick_params(axis = 'both', which = 'both',length = 0)
    #plt.xticks(np.arange(1, 2*ngroups+1, step = 2),names,rotation = -90,ha = 'right')
    fig.subplots_adjust(top = ((ymax - ymin) * 0.7 + bottomm) / fig_length, bottom = bottomm / fig_length, left = leftm / fig_width, right=1 - rightm / fig_width)
    #plt.tight_layout()
    #plt.savefig(output)
    #plt.close()

## plotID_generic_v2 --> plotTMB_ID
def plotTMB_ID(inputDF, kzm611_sigs_rel, scale, order=[], Yrange = "adapt", cutoff = 0, output = "TMB_plot.png",
            redbar = "median", yaxis = "Somatic Mutations per Megabase",
            ascend = True, leftm = 1, rightm = 0.3, topm = 1.4, bottomm = 1):
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
    #fig_length = topm + bottomm + (ymax - ymin) * 0.7
    #print("{} -- {}".format(fig_width, fig_length))
    #fig_width = 4
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
    #print(len(names[0]))
    ax2.set_ylim(ymin,ymax)
    yticks_loc = range(ymin,ymax+1,1)
    ax2.set_yticks(yticks_loc,list(map((lambda x: 10**x), list(yticks_loc))))
    ax2.set_xticks(np.arange(1, 2*ngroups+1, step = 2), new_labels) 
    ax2.tick_params(axis = 'both', which = 'both',length = 0)
    ax2.hlines(yticks_loc,0,2*ngroups,colors = 'black',linestyles = "dashed",linewidth = 0.5,zorder = 1)
    '''for i in range(0,ngroups,2):
        greystart = [(i)*2,ymin]
        rectangle = mpatches.Rectangle(greystart, 2, ymax-ymin, color = "lightgrey",zorder = 0)
        ax2.add_patch(rectangle)'''
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
        #rg = 1.8
        y_values = groups.get_group(names[i])["log10BURDENpMB"].sort_values(ascending = True).values.tolist()
        x_values = list(np.linspace(start = X_start, stop = X_end, num = counts[i]))
        ax2.scatter(x_values,y_values,color = "black",s=1.5)
        ax2.hlines(redbars[i], X_start, X_end, colors='red', zorder=2)
        ax2.text((leftm + 0.2 + i * 0.4) / fig_width , 0.85 / fig_length , "___",  horizontalalignment='center',transform=plt.gcf().transFigure)
    ax2.set_ylabel(yaxis)
    #axes2 = ax.twiny()
    #plt.text((leftm - 0.3) / fig_width, 0.2 / fig_length, "*Showing samples with counts more than %d" % cutoff, transform=plt.gcf().transFigure) 
    #plt.tick_params(axis = 'both', which = 'both',length = 0)
    #plt.xticks(np.arange(1, 2*ngroups+1, step = 2),names,rotation = -90,ha = 'right')
    fig.subplots_adjust(top = ((ymax - ymin) * 0.7 + bottomm) / fig_length, bottom = bottomm / fig_length, left = leftm / fig_width, right=1 - rightm / fig_width)
    #plt.tight_layout()
    #plt.savefig(output)
    #plt.close()


## plotID_generic_v2 --> plotTMB_ID
def plotTMB_CN(inputDF, kzm611_sigs_rel, scale, order=[], Yrange = "adapt", cutoff = 0, output = "TMB_plot.png",
            redbar = "median", yaxis = "Somatic Mutations per Megabase",
            ascend = True, leftm = 1, rightm = 0.3, topm = 1.4, bottomm = 1):
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
    #fig_length = topm + bottomm + (ymax - ymin) * 0.7
    #print("{} -- {}".format(fig_width, fig_length))
    #fig_width = 4
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
    #ax1.legend(loc='upper right', bbox_to_anchor=(1,2.75), ncol=15, fontsize=12, facecolor="#f4f0eb")
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
    '''for i in range(0,ngroups,2):
        greystart = [(i)*2,ymin]
        rectangle = mpatches.Rectangle(greystart, 2, ymax-ymin, color = "lightgrey",zorder = 0)
        ax2.add_patch(rectangle)'''
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
    #axes2 = ax.twiny()
    #plt.text((leftm - 0.3) / fig_width, 0.2 / fig_length, "*Showing samples with counts more than %d" % cutoff, transform=plt.gcf().transFigure) 
    #plt.tick_params(axis = 'both', which = 'both',length = 0)
    #plt.xticks(np.arange(1, 2*ngroups+1, step = 2),names,rotation = -90,ha = 'right')
    fig.subplots_adjust(top = ((ymax - ymin) * 0.7 + bottomm) / fig_length, bottom = bottomm / fig_length, left = leftm / fig_width, right=1 - rightm / fig_width)
    #plt.tight_layout()
    #plt.savefig(output)
    #plt.close()

## plotID_generic_v2 --> plotTMB_ID
def plotTMB_SV(inputDF, kzm611_sigs_rel, scale, order=[], Yrange = "adapt", cutoff = 0, output = "TMB_plot.png",
            redbar = "median", yaxis = "Somatic Mutations per Megabase",
            ascend = True, leftm = 1, rightm = 0.3, topm = 1.4, bottomm = 1):
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
    #fig_length = topm + bottomm + (ymax - ymin) * 0.7
    #print("{} -- {}".format(fig_width, fig_length))
    #fig_width = 4
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
    #ax1.legend(loc='upper right', bbox_to_anchor=(1,2.75), ncol=15, fontsize=12, facecolor="#f4f0eb")
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
    '''for i in range(0,ngroups,2):
        greystart = [(i)*2,ymin]
        rectangle = mpatches.Rectangle(greystart, 2, ymax-ymin, color = "lightgrey",zorder = 0)
        ax2.add_patch(rectangle)'''
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
    #axes2 = ax.twiny()
    #plt.text((leftm - 0.3) / fig_width, 0.2 / fig_length, "*Showing samples with counts more than %d" % cutoff, transform=plt.gcf().transFigure) 
    #plt.tick_params(axis = 'both', which = 'both',length = 0)
    #plt.xticks(np.arange(1, 2*ngroups+1, step = 2),names,rotation = -90,ha = 'right')
    fig.subplots_adjust(top = ((ymax - ymin) * 0.7 + bottomm) / fig_length, bottom = bottomm / fig_length, left = leftm / fig_width, right=1 - rightm / fig_width)
    #plt.tight_layout()
    #plt.savefig(output)
    #plt.close()



def get_diff_dict(mat_df, labels):
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

def plot_profile_diff(sample1, sample2, name1, name2, ymax):
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

    panel1.text(0.05, 0.8, name1, fontsize=60, weight="bold", color="black", fontname="Arial", transform=plt.gcf().transFigure)

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
    panel2 = plot1.add_axes([0.04, 0, 0.95, 0.4])
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
    #plt.gca().invert_yaxis()

    
    panel2.text(0.05, 0.32, name2, fontsize=60, weight="bold", color="black", fontname="Arial", transform=plt.gcf().transFigure)

    panel2.tick_params(axis='both',which='both',\
                    bottom=False, labelbottom=False,\
                    left=True, labelleft=True,\
                    right=True, labelright=False,\
                    top=False, labeltop=False,\
                    direction='in', length=25, colors='lightgray', width=2)

    [i.set_color("black") for i in panel2.get_yticklabels()]

    plt.show()

def plot_3samples_SBS96(sample1, sample2, sample3, name1, name2, name3, ymax):
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
    #plt.gca().invert_yaxis()

    
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
    #ymax = 0
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
    #plt.gca().invert_yaxis()

    
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

## plotTMB_generic_v2 --> plotTMB_clustered
def plotTMB_clustered(inputDF, scale, order=[], Yrange = "adapt", cutoff = 0, output = "TMB_plot.png",
            redbar = "median", yaxis = "Somatic Mutations per Megabase",
            ascend = True, leftm = 1, rightm = 0.3, topm = 1.4, bottomm = 1):
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
    #fig_length = topm + bottomm + (ymax - ymin) * 0.7
    #print("{} -- {}".format(fig_width, fig_length))
    #fig_width = 4
    fig_length = 6
    fig, ax = plt.subplots(figsize=(fig_width, fig_length))
    if cutoff < 0:
        print("ERROR: cutoff value is less than 0")
        return
    plt.xlim(0,2*ngroups)
    #print(len(names[0]))
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

    '''for i in range(0,ngroups,2):
        greystart = [(i)*2,ymin]
        rectangle = mpatches.Rectangle(greystart, 2, ymax-ymin, color = "lightgrey",zorder = 0)
        ax.add_patch(rectangle)'''
    for i in range(0,ngroups,1):
        X_start = i*2+0.2
        X_end = i*2+2-0.2
        #rg = 1.8
        y_values = groups.get_group(names[i])["log10BURDENpMB"].sort_values(ascending = True).values.tolist()
        x_values = list(np.linspace(start = X_start, stop = X_end, num = counts[i]))
        plt.scatter(x_values,y_values,color = "black",s=1.5)
        plt.hlines(redbars[i], X_start, X_end, colors='red', zorder=2)
        plt.text((leftm + 0.2 + i * 0.4) / fig_width , 0.85 / fig_length , "___",  horizontalalignment='center',transform=plt.gcf().transFigure)
    plt.ylabel(yaxis)
    axes2 = ax.twiny()
    #plt.text((leftm - 0.3) / fig_width, 0.2 / fig_length, "*Showing samples with counts more than %d" % cutoff, transform=plt.gcf().transFigure) 
    plt.tick_params(axis = 'both', which = 'both',length = 0)
    plt.xticks(np.arange(1, 2*ngroups+1, step = 2),names,rotation = -90,ha = 'right')
    fig.subplots_adjust(top = ((ymax - ymin) * 0.7 + bottomm) / fig_length, bottom = bottomm / fig_length, left = leftm / fig_width, right=1 - rightm / fig_width)
    #plt.savefig(output)
    #plt.close()

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

    #data = coefs[coefs.Condition=='agent_sigs_X_all'].copy()
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
