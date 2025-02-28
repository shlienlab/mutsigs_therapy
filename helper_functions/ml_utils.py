from sklearn.model_selection import train_test_split, cross_val_score, RepeatedStratifiedKFold
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.metrics import RocCurveDisplay, classification_report, ConfusionMatrixDisplay, precision_recall_curve
from sklearn.metrics import auc, roc_curve, roc_auc_score, average_precision_score
from sklearn.metrics import recall_score, precision_score, f1_score, accuracy_score
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

#from imblearn.over_sampling import SMOTE
#from imblearn.under_sampling import RandomUnderSampler
#from imblearn.pipeline import Pipeline

#load libraries
from sklearn.preprocessing import StandardScaler
#import scikitplot as skplt
from sklearn.preprocessing import label_binarize


import xgboost
import shap
import numpy as np
import pandas as pd

from scipy import interp
from statistics import stdev
import random

import matplotlib.pyplot as plt


import warnings
warnings.filterwarnings('ignore')
# Suppress FutureWarning messages
warnings.simplefilter(action='ignore', category=FutureWarning)

fig_out = '/Users/mehdi/Documents/MyRepos/MS1_Figures'

seq96 = ["A[C>A]A", "A[C>A]C", "A[C>A]G", "A[C>A]T", "C[C>A]A", "C[C>A]C", "C[C>A]G",  "C[C>A]T", "G[C>A]A", "G[C>A]C", "G[C>A]G", "G[C>A]T", "T[C>A]A","T[C>A]C", "T[C>A]G", "T[C>A]T",
         "A[C>G]A", "A[C>G]C", "A[C>G]G", "A[C>G]T", "C[C>G]A", "C[C>G]C", "C[C>G]G", "C[C>G]T","G[C>G]A", "G[C>G]C", "G[C>G]G", "G[C>G]T", "T[C>G]A", "T[C>G]C", "T[C>G]G", "T[C>G]T",
         "A[C>T]A", "A[C>T]C", "A[C>T]G", "A[C>T]T", "C[C>T]A", "C[C>T]C", "C[C>T]G", "C[C>T]T", "G[C>T]A", "G[C>T]C", "G[C>T]G", "G[C>T]T", "T[C>T]A", "T[C>T]C", "T[C>T]G", "T[C>T]T",
         "A[T>A]A", "A[T>A]C", "A[T>A]G", "A[T>A]T", "C[T>A]A", "C[T>A]C", "C[T>A]G", "C[T>A]T", "G[T>A]A", "G[T>A]C", "G[T>A]G", "G[T>A]T", "T[T>A]A", "T[T>A]C", "T[T>A]G", "T[T>A]T",
         "A[T>C]A", "A[T>C]C", "A[T>C]G", "A[T>C]T", "C[T>C]A", "C[T>C]C", "C[T>C]G", "C[T>C]T", "G[T>C]A", "G[T>C]C", "G[T>C]G", "G[T>C]T", "T[T>C]A", "T[T>C]C", "T[T>C]G", "T[T>C]T",
         "A[T>G]A", "A[T>G]C", "A[T>G]G", "A[T>G]T", "C[T>G]A", "C[T>G]C", "C[T>G]G", "C[T>G]T", "G[T>G]A", "G[T>G]C", "G[T>G]G", "G[T>G]T", "T[T>G]A", "T[T>G]C", "T[T>G]G", "T[T>G]T"
        ]
seq48 = ['0:homdel:0-100kb', '0:homdel:100kb-1Mb', '0:homdel:>1Mb', '1:LOH:0-100kb', '1:LOH:100kb-1Mb', '1:LOH:1Mb-10Mb', '1:LOH:10Mb-40Mb', '1:LOH:>40Mb', '2:LOH:0-100kb', '2:LOH:100kb-1Mb',
         '2:LOH:1Mb-10Mb', '2:LOH:10Mb-40Mb', '2:LOH:>40Mb', '3-4:LOH:0-100kb', '3-4:LOH:100kb-1Mb', '3-4:LOH:1Mb-10Mb', '3-4:LOH:10Mb-40Mb', '3-4:LOH:>40Mb', '5-8:LOH:0-100kb', '5-8:LOH:100kb-1Mb',
         '5-8:LOH:1Mb-10Mb', '5-8:LOH:10Mb-40Mb', '5-8:LOH:>40Mb', '9+:LOH:0-100kb', '9+:LOH:100kb-1Mb', '9+:LOH:1Mb-10Mb', '9+:LOH:10Mb-40Mb', '9+:LOH:>40Mb', '2:het:0-100kb', '2:het:100kb-1Mb',
         '2:het:1Mb-10Mb', '2:het:10Mb-40Mb', '2:het:>40Mb', '3-4:het:0-100kb', '3-4:het:100kb-1Mb', '3-4:het:1Mb-10Mb', '3-4:het:10Mb-40Mb', '3-4:het:>40Mb', '5-8:het:0-100kb', '5-8:het:100kb-1Mb',
         '5-8:het:1Mb-10Mb', '5-8:het:10Mb-40Mb', '5-8:het:>40Mb', '9+:het:0-100kb', '9+:het:100kb-1Mb', '9+:het:1Mb-10Mb', '9+:het:10Mb-40Mb', '9+:het:>40Mb'
         ]
seq32 = ['clustered_del_1-10Kb', 'clustered_del_10-100Kb', 'clustered_del_100Kb-1Mb', 'clustered_del_1Mb-10Mb', 'clustered_del_>10Mb', 'clustered_tds_1-10Kb', 'clustered_tds_10-100Kb', 'clustered_tds_100Kb-1Mb',
         'clustered_tds_1Mb-10Mb', 'clustered_tds_>10Mb', 'clustered_inv_1-10Kb', 'clustered_inv_10-100Kb', 'clustered_inv_100Kb-1Mb', 'clustered_inv_1Mb-10Mb', 'clustered_inv_>10Mb', 'clustered_trans',
         'non-clustered_del_1-10Kb', 'non-clustered_del_10-100Kb', 'non-clustered_del_100Kb-1Mb', 'non-clustered_del_1Mb-10Mb', 'non-clustered_del_>10Mb', 'non-clustered_tds_1-10Kb', 'non-clustered_tds_10-100Kb',
         'non-clustered_tds_100Kb-1Mb', 'non-clustered_tds_1Mb-10Mb', 'non-clustered_tds_>10Mb', 'non-clustered_inv_1-10Kb', 'non-clustered_inv_10-100Kb', 'non-clustered_inv_100Kb-1Mb', 'non-clustered_inv_1Mb-10Mb',
         'non-clustered_inv_>10Mb', 'non-clustered_trans'
         ]


def get_report_boot(mat, naives, treats, cat='G0'):
    #report = mu.get_logreg_topq_v2(mat, naives, treats, cv=True, extra_plots=False)
    report = get_votingClf_v1(mat, naives, treats, cv=True, extra_plots=False)
    
    report = report.loc[['Sig -', 'Sig +']]
    report = report.reset_index().rename(columns={'index':'Signature'})
    report['Category'] = cat
    return report


def report_boot(mat, reps):
    reports = pd.DataFrame()

    for i in range(reps):
        print('###\n{rep} iteration ...\n')
        pret_random1 = random.sample(pret_samples, k=len(platPos_samples))
        pret_random2 = random.sample(pret_samples, k=len(topr_quad_samples))
        pret_random3 = random.sample(pret_samples, k=len(sigP_samples))
        pret_random4 = random.sample(pret_samples, k=len(sigN_samples))

        reports = pd.concat([reports, get_report_boot(mat, pret_random1, platPos_samples, cat='Platinum\nTreated')], axis=0).reset_index(drop=True)
        reports = pd.concat([reports, get_report_boot(mat, pret_random2, topr_quad_samples, cat="TopRQ")], axis=0).reset_index(drop=True)
        reports = pd.concat([reports, get_report_boot(mat, pret_random3, sigP_samples, cat="Sig+")], axis=0).reset_index(drop=True)
        reports = pd.concat([reports, get_report_boot(mat, pret_random4, sigN_samples, cat="Sig-")], axis=0).reset_index(drop=True)
    reports.columns = ['Signature', 'Precision', 'Recall', 'F1-score', 'Support', 'Positive Class (in Red)']
    return reports




def report_boot_burden(mat, reps):
    reports = pd.DataFrame()

    for i in range(reps):
        pret_random1 = random.sample(pret_samples, k=len(platPos_samples))
        pret_random2 = random.sample(nhm_naives_thresh, k=len(topr_quad_samples))
        pret_random3 = random.sample(nhm_naives_thresh, k=len(sigP_samples))
        pret_random4 = random.sample(nhm_naives_thresh, k=len(sigN_samples))

        reports = pd.concat([reports, get_report_boot(mat, pret_random1, platPos_samples, cat='Platinum\nTreated')], axis=0).reset_index(drop=True)
        reports = pd.concat([reports, get_report_boot(mat, pret_random2, topr_quad_samples, cat="TopRQ")], axis=0).reset_index(drop=True)
        reports = pd.concat([reports, get_report_boot(mat, pret_random3, sigP_samples, cat="Sig+")], axis=0).reset_index(drop=True)
        reports = pd.concat([reports, get_report_boot(mat, pret_random4, sigN_samples, cat="Sig-")], axis=0).reset_index(drop=True)
    reports.columns = ['Signature', 'Precision', 'Recall', 'F1-score', 'Support', 'Positive Class (in Red)']
    return reports


def get_votingClf_v1(mat_df, neg_samples, pos_samples, cv=False, extra_plots=False, plot_title='', verbose=False):

    df = pd.concat([mat_df.loc[neg_samples], mat_df.loc[pos_samples]], axis=0)
    df['label'] = ([0]*len(neg_samples)) + ([1]*len(pos_samples))
    #df['label'] = (['Sig -']*len(neg_samples)) + (['Sig +']*len(pos_samples))

    X, y  = df.iloc[:,0:-1], df['label']
    X_train, X_test, y_train, y_test = split_norm_Xy(X, y)
    target_names = ['Sig -', 'Sig +']

    estimator = []
    estimator.append(('LogisticRegression', LogisticRegression(solver='saga', l1_ratio=0.5, random_state=42, penalty='elasticnet', max_iter = 1000)))
    estimator.append(('RandomForest', RandomForestClassifier(n_estimators=100, random_state=42, min_samples_leaf=.1)))
    #estimator.append(('GradientBoostingClassifier', GradientBoostingClassifier(random_state=42) ))
    estimator.append(('XGB', XGBClassifier(n_estimators=100, objective='binary:logistic', random_state=42)))
    estimator.append(('CatBoost', CatBoostClassifier(logging_level='Silent', random_state=42) ))

    VC_soft = VotingClassifier(estimators = estimator, voting ='soft')
    VC_soft.fit(X_train, y_train)
    y_pred = VC_soft.predict(X_test)
    y_prob = VC_soft.predict_proba(X_test)

    recall_score = cross_val_score(VC_soft, X_train, y_train, cv=5, scoring='recall')
    VC_soft_cv_recall = recall_score.mean()
    VC_soft_cv_stdev_recall = stdev(recall_score)

    prec_score = cross_val_score(VC_soft, X_train, y_train, cv=5, scoring='precision')
    VC_soft_cv_prec = prec_score.mean()
    VC_soft_cv_stdev_prec = stdev(prec_score)

    f1_score = cross_val_score(VC_soft, X_train, y_train, cv=5, scoring='f1')
    VC_soft_cv_f1score = f1_score.mean()
    VC_soft_cv_stdev_f1 = stdev(f1_score)

    '''print('Cross Validation Recall scores are: {}'.format(score))
    print('Average Cross Validation Recall score: ', VC_soft_cv_score)
    print('Cross Validation Recall standard deviation: ', VC_soft_cv_stdev)'''


    ndf = [(VC_soft_cv_recall, VC_soft_cv_stdev_recall, VC_soft_cv_prec, VC_soft_cv_stdev_prec, VC_soft_cv_f1score, VC_soft_cv_stdev_f1)]

    VC_soft_score = pd.DataFrame(data = ndf, columns=
                            ['Avg_CV_Recall', 'SD_CV_Recall', 'Avg_CV_Precision', 'SD_CV_Precision', 'Avg_CV_f1-score', 'SD_CV_f1-score'])
    #VC_soft_score.insert(0, 'Voting Classifier', 'Soft Voting')
    #print(VC_soft_score)

    if verbose:
        print(f"Accuracy: {accuracy_score(y_test, y_pred)}\n")
        print(f"Auroc score: {roc_auc_score(y_test, y_pred)}\n")
        print(f"Average precision score: {average_precision_score(y_test, y_pred)}\n")

    if extra_plots:
        fig, axes = plt.subplots(1,2, figsize=(12, 5))
        plot_roc_curve(y_test, y_prob, title = 'ROC Plot', target_names=['Sig -', 'Sig +'], ax=axes[0])
        _ = axes[0].set(
            xlabel="False Positive Rate (1 - Specificity)",
            ylabel="True Positive Rate (Sensitivity)",
            title="ROC",
            xlim=(-0.03, 1),
            ylim=(0, 1.03),
        )
        plot_precision_recall_curve(y_test, y_prob, title = 'PR Curve', target_names=['Sig -', 'Sig +'], ax=axes[1])
        _ = axes[1].set( xlim=(0, 1.03) )
        
        ax2 = fig.add_axes([.817,.16, .075,.275])
        disp = ConfusionMatrixDisplay.from_estimator(VC_soft, X_test, y_test,
                                                    display_labels=target_names,
                                                    cmap=plt.cm.Blues,
                                                    normalize=None,
                                                    ax=ax2,
                                                    colorbar=False
        )
        fig.suptitle(plot_title, fontsize=16)

    '''cvK = RepeatedStratifiedKFold(n_splits=5, n_repeats=100, random_state=42)
    n_scoresK = cross_val_score(logreg, X, y, scoring='accuracy', cv=cvK, n_jobs=4, error_score='raise')
    print('Accuracy KZM: %.3f (%.3f)' % (np.mean(n_scoresK), np.std(n_scoresK)))'''
    #print(classification_report(y_test, y_pred, target_names=target_names))
    #specificity = tn / (tn+fp)
    report = pd.DataFrame(classification_report(y_test, y_pred, target_names=target_names, output_dict=True)).transpose()    
    #return pd.Series(rfc.feature_importances_, index=X_train.columns)
    return report, VC_soft_score
    
def feats_2df(features, colname):
    features_df = pd.DataFrame(features, columns=['mType', colname])
    features_df.index = [f"{x[0]}[{x[1]}>{x[2]}]{x[-1]}" for x in features_df.mType]
    features_df = features_df.drop(['mType'], axis=1)
    features_df = features_df.loc[seq96]
    features_df = features_df.div(features_df.sum(axis=0), axis=1)
    return features_df

def SV_feats_2df(features, colname):
    features_df = pd.DataFrame(features, columns=['mType', colname])
    features_df.index = [x.replace('_10Mb', '_>10Mb') if '_10Mb' in x else 
                         x.replace(':40Mb', ':>40Mb') if ':40Mb' in x else
                         x.replace(':1Mb', ':>1Mb') if x.endswith(':1Mb') else
                         x for x in features_df.mType]
    #features_df = features_df.set_index('mType')
    features_df = features_df.drop(['mType'], axis=1)
    if features_df.shape[0] == 32:
        features_df = features_df.loc[seq32]
    elif features_df.shape[0] == 48:
        features_df = features_df.loc[seq48]

    features_df = features_df.div(features_df.sum(axis=0), axis=1)
    return features_df

def get_context(fn, low_samples=None):
    mat_df = pd.read_csv(fn, sep='\t', index_col=0).T
    mat_rel = mat_df.div(mat_df.sum(axis=1), axis=0)
    if low_samples:
        low_samples = [x for x in low_samples if x in mat_df.index.tolist()]
        mat_nhm = mat_df.loc[low_samples]
        mat_nhm_rel = mat_rel.loc[low_samples]
        return mat_df, mat_rel, mat_nhm, mat_nhm_rel
    else:
        return mat_df, mat_rel

## SuppFig_15
def load_pog_drugs(drugs_file=None):
    if drugs_file==None:
        drugs_file = '/Users/mehdi/Documents/MyRepos/POG/POG570_all_drugs_YN.tsv'
    drugs_df = pd.read_csv(drugs_file, sep='\t', index_col=0)
    return drugs_df

def split_norm_Xy(X, y):
    # Create the training and test data
    X = StandardScaler().fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, stratify=y, random_state = 42)
    return X_train, X_test, y_train, y_test

def plot_roc_curve(y_true, y_probas, title='ROC Curves',
                   curves=('micro', 'macro', 'each_class'),
                   ax=None, figsize=None, cmap='nipy_spectral',
                   target_names=None,
                   title_fontsize="xx-large", text_fontsize="large"):
    """Generates the ROC curves from labels and predicted scores/probabilities

    Args:
        y_true (array-like, shape (n_samples)):
            Ground truth (correct) target values.

        y_probas (array-like, shape (n_samples, n_classes)):
            Prediction probabilities for each class returned by a classifier.

        title (string, optional): Title of the generated plot. Defaults to
            "ROC Curves".

        curves (array-like): A listing of which curves should be plotted on the
            resulting plot. Defaults to `("micro", "macro", "each_class")`
            i.e. "micro" for micro-averaged curve, "macro" for macro-averaged
            curve

        ax (:class:`matplotlib.axes.Axes`, optional): The axes upon which to
            plot the curve. If None, the plot is drawn on a new set of axes.

        figsize (2-tuple, optional): Tuple denoting figure size of the plot
            e.g. (6, 6). Defaults to ``None``.

        cmap (string or :class:`matplotlib.colors.Colormap` instance, optional):
            Colormap used for plotting the projection. View Matplotlib Colormap
            documentation for available options.
            https://matplotlib.org/users/colormaps.html

        title_fontsize (string or int, optional): Matplotlib-style fontsizes.
            Use e.g. "small", "medium", "large" or integer-values. Defaults to
            "large".

        text_fontsize (string or int, optional): Matplotlib-style fontsizes.
            Use e.g. "small", "medium", "large" or integer-values. Defaults to
            "medium".

    Returns:
        ax (:class:`matplotlib.axes.Axes`): The axes on which the plot was
            drawn.

    Example:
        >>> import scikitplot.plotters as skplt
        >>> nb = GaussianNB()
        >>> nb = nb.fit(X_train, y_train)
        >>> y_probas = nb.predict_proba(X_test)
        >>> skplt.plot_roc_curve(y_test, y_probas)
        <matplotlib.axes._subplots.AxesSubplot object at 0x7fe967d64490>
        >>> plt.show()

        .. image:: _static/examples/plot_roc_curve.png
           :align: center
           :alt: ROC Curves
    """
    y_true = np.array(y_true)
    y_probas = np.array(y_probas)

    if 'micro' not in curves and 'macro' not in curves and \
            'each_class' not in curves:
        raise ValueError('Invalid argument for curves as it '
                         'only takes "micro", "macro", or "each_class"')

    classes = np.unique(y_true)
    probas = y_probas

    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(len(classes)):
        fpr[i], tpr[i], _ = roc_curve(y_true, probas[:, i],
                                      pos_label=classes[i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    micro_key = 'micro'
    i = 0
    while micro_key in fpr:
        i += 1
        micro_key += str(i)

    y_true = label_binarize(y_true, classes=classes)
    if len(classes) == 2:
        y_true = np.hstack((1 - y_true, y_true))

    fpr[micro_key], tpr[micro_key], _ = roc_curve(y_true.ravel(),
                                                  probas.ravel())
    roc_auc[micro_key] = auc(fpr[micro_key], tpr[micro_key])

    # Compute macro-average ROC curve and ROC area

    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[x] for x in range(len(classes))]))

    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(len(classes)):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])

    # Finally average it and compute AUC
    mean_tpr /= len(classes)

    macro_key = 'macro'
    i = 0
    while macro_key in fpr:
        i += 1
        macro_key += str(i)
    fpr[macro_key] = all_fpr
    tpr[macro_key] = mean_tpr
    roc_auc[macro_key] = auc(fpr[macro_key], tpr[macro_key])

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)

    if 'each_class' in curves:
        for i in range(len(classes)):
            color = plt.cm.get_cmap(cmap)(float(i) / len(classes))
            ax.plot(fpr[i], tpr[i], lw=2, color=color,
                    label='{0} (AUC = {1:0.2f})'
                    ''.format(target_names[i], roc_auc[i]))

    if 'micro' in curves:
        ax.plot(fpr[micro_key], tpr[micro_key],
                label='micro-average '
                      '(AUC = {0:0.2f})'.format(roc_auc[micro_key]),
                color='deeppink', linestyle='--', linewidth=2)

    if 'macro' in curves:
        ax.plot(fpr[macro_key], tpr[macro_key],
                label='macro-average '
                      '(AUC = {0:0.2f})'.format(roc_auc[macro_key]),
                color='navy', linestyle='--', linewidth=2)

    ax.plot([0, 1], [0, 1], color = 'k', linestyle = '--', lw=2)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate', fontsize="large")
    ax.set_ylabel('True Positive Rate', fontsize="large")
    #ax.tick_params(labelsize="large")
    ax.legend(loc='lower right', fontsize="large", frameon=False)
    ax.set_title(title, fontsize=26)
    return ax

def plot_precision_recall_curve(y_true, y_probas,
                                title='Precision-Recall Curve',
                                curves=('micro', 'each_class'), ax=None,
                                figsize=None, cmap='nipy_spectral',
                                target_names=None,
                                title_fontsize="xx-large",
                                text_fontsize="large"):
    """Generates the Precision Recall Curve from labels and probabilities

    Args:
        y_true (array-like, shape (n_samples)):
            Ground truth (correct) target values.

        y_probas (array-like, shape (n_samples, n_classes)):
            Prediction probabilities for each class returned by a classifier.

        curves (array-like): A listing of which curves should be plotted on the
            resulting plot. Defaults to `("micro", "each_class")`
            i.e. "micro" for micro-averaged curve

        ax (:class:`matplotlib.axes.Axes`, optional): The axes upon which to
            plot the curve. If None, the plot is drawn on a new set of axes.

        figsize (2-tuple, optional): Tuple denoting figure size of the plot
            e.g. (6, 6). Defaults to ``None``.

        cmap (string or :class:`matplotlib.colors.Colormap` instance, optional):
            Colormap used for plotting the projection. View Matplotlib Colormap
            documentation for available options.
            https://matplotlib.org/users/colormaps.html

        title_fontsize (string or int, optional): Matplotlib-style fontsizes.
            Use e.g. "small", "medium", "large" or integer-values. Defaults to
            "large".

        text_fontsize (string or int, optional): Matplotlib-style fontsizes.
            Use e.g. "small", "medium", "large" or integer-values. Defaults to
            "medium".

    Returns:
        ax (:class:`matplotlib.axes.Axes`): The axes on which the plot was
            drawn.

    Example:
        >>> import scikitplot.plotters as skplt
        >>> nb = GaussianNB()
        >>> nb = nb.fit(X_train, y_train)
        >>> y_probas = nb.predict_proba(X_test)
        >>> skplt.plot_precision_recall_curve(y_test, y_probas)
        <matplotlib.axes._subplots.AxesSubplot object at 0x7fe967d64490>
        >>> plt.show()

        .. image:: _static/examples/plot_precision_recall_curve.png
           :align: center
           :alt: Precision Recall Curve
    """
    chance_ap = sum(y_true) / len(y_true)
    y_true = np.array(y_true)
    y_probas = np.array(y_probas)

    classes = np.unique(y_true)
    probas = y_probas

    if 'micro' not in curves and 'each_class' not in curves:
        raise ValueError('Invalid argument for curves as it '
                         'only takes "micro" or "each_class"')

    # Compute Precision-Recall curve and area for each class
    precision = dict()
    recall = dict()
    average_precision = dict()
    for i in range(len(classes)):
        precision[i], recall[i], _ = precision_recall_curve(
            y_true, probas[:, i], pos_label=classes[i])

    y_true = label_binarize(y_true, classes=classes)
    if len(classes) == 2:
        y_true = np.hstack((1 - y_true, y_true))

    for i in range(len(classes)):
        average_precision[i] = average_precision_score(y_true[:, i],
                                                       probas[:, i])

    # Compute micro-average ROC curve and ROC area
    micro_key = 'micro'
    i = 0
    while micro_key in precision:
        i += 1
        micro_key += str(i)

    precision[micro_key], recall[micro_key], _ = precision_recall_curve(
        y_true.ravel(), probas.ravel())
    average_precision[micro_key] = average_precision_score(y_true, probas,
                                                           average='micro')

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)

    ax.set_title(title, fontsize=title_fontsize)

    if 'each_class' in curves:
        for i in range(len(classes)):
            color = plt.cm.get_cmap(cmap)(float(i) / len(classes))
            ax.plot(recall[i], precision[i], lw=2,
                    label='{0} '
                          '(AP = {1:0.3f})'.format(target_names[i],
                                                     average_precision[i]),
                    color=color)

    if 'micro' in curves:
        ax.plot(recall[micro_key], precision[micro_key],
                label='micro-average '
                      '(AP = {0:0.3f})'.format(average_precision[micro_key]),
                color='deeppink', linestyle='--', linewidth=2)
    ax.axhline(y = chance_ap, color = 'k', linestyle = '--',
               label='Chance '
                      '(AP = {0:0.3f})'.format(chance_ap)) 

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('Recall', fontsize="x-large")
    ax.set_ylabel('Precision', fontsize="x-large")
    ax.tick_params(labelsize="large")
    ax.legend(loc='best', fontsize="large", frameon=False)
    return ax

def shap_swarm_single(mat_df, neg_samples, pos_samples):
    # Create the training and test data
    X = pd.concat([mat_df.loc[neg_samples], mat_df.loc[pos_samples]], axis=0)
    y = ([0]*len(neg_samples)) + ([1]*len(pos_samples))
    #X.columns = [x[0]+x[2]+x[4]+x[6] for x in X.columns]
    X.columns = [x.replace('[', '').replace(']', '').replace('<', '').replace('>', '') for x in X.columns]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

    y_train = np.array(y_train)
    y_test = np.array(y_test)

    d_train = xgboost.DMatrix(X_train, label=y_train)
    d_test = xgboost.DMatrix(X_test, label=y_test)

    # train final model on the full data set
    params = {
        "eta": 0.05,
        "max_depth": 1,
        "objective": "binary:logistic",
        "subsample": 0.5,
        "base_score": np.mean(y_train),
        "eval_metric": "logloss",
    }
    evals_result = {}
    model_ind = xgboost.train(params, d_train, 5000, evals=[(d_test, "test")], verbose_eval=0,
                              early_stopping_rounds=20, evals_result=evals_result)

    #shap_values_ind = shap.TreeExplainer(model_ind).shap_values(X_train)
    explainer = shap.TreeExplainer(model_ind)
    shap_values = explainer(X)


    features = list(zip(model_ind.feature_names, shap_values.abs.mean(0).values))
    #max_feats = sum(shap_values.abs.mean(0).values>=0.05)
    #shap.plots.bar(shap_values.abs.mean(0), max_display=max_feats+1)

    return model_ind, features


def get_SHAP_features(mat_df, class_df, pret_samples, n_iters):
    pret_samples = [x for x in pret_samples if x in mat_df.index.tolist()]

    features_drug = pd.DataFrame()
    for drug in class_df.columns:
        drug_samples = class_df[class_df[drug]=='Y'].index.tolist()
        drug_samples = [x for x in drug_samples if x in mat_df.index.tolist()]
        print(f'{drug}: {len(drug_samples)}')

        features_df_Q = pd.DataFrame()
        for i in range(n_iters):
            if len(drug_samples) - len(pret_samples) <= len(drug_samples)/10:
                pret_random = random.sample(pret_samples, k=len(drug_samples))
            else:
                pret_random = pret_samples

            model, features = shap_swarm_single(mat_df, pret_random, drug_samples)

            features_df = SV_feats_2df(features, f'{i}_{drug}')
            features_df_Q = pd.concat([features_df_Q, features_df], axis=1)

        features_df_Q[f'sum_{drug}'] = features_df_Q[[x for x in features_df_Q.columns if x.endswith(f'{drug}')]].sum(axis=1)

        features_df_Q[f'mean_{drug}'] = features_df_Q[[x for x in features_df_Q.columns if x.endswith(f'{drug}')]].mean(axis=1)

        features_df_Q[drug] = features_df_Q[f'sum_{drug}'] / features_df_Q[f'sum_{drug}'].sum()

        features_drug = pd.concat([features_drug, features_df_Q[drug]], axis=1)

    return features_drug