"""
===========================================
MS: Unique patterns of mutations in childhood cancer highlight chemotherapy’s disease-defining role at relapse
Author: Mehdi Layeghifard
Email: mlayeghi@gmail.com
Date Created: February 28, 2025
Version: 0.1

Description:
This script contains the logistic regression model to find associations between drugs and signatures.

Usage:
These function are called from within the provided Notebooks.

License:
MIT License - You are free to use, modify, and distribute this code with appropriate attribution.
===========================================
"""


## Data processing imports
import pandas as pd
import numpy as np

## ML imports
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.utils import shuffle
from patsy import dmatrices

## Other imports
import time
import os



## Random seed
np.random.seed(42)

## Provide an output directory for figures or use 'installation_folder/figures' directory
root_dir = os.path.dirname(os.getcwd())
fig_out = os.path.join(root_dir, 'figures')

##### Data import & processing
## Load patients metadata
kzm611_md = pd.read_csv(os.path.join(root_dir, 'source_data/KZM_md.tsv'), sep='\t', index_col=0)
low_samples = kzm611_md[kzm611_md.Burden=='low'].index.tolist()

## Load age & purity
kzm611_sample_age = pd.read_csv(os.path.join(root_dir, 'source_data/KZM_sample_age.txt'), sep='\t', index_col=0)
kzm611_md= pd.concat([kzm611_md, kzm611_sample_age], axis=1)

## Load & process divers
kzm_drivers_all = pd.read_csv(os.path.join(root_dir, 'source_data/KZM611_drivers_CosCGI.tsv'), sep='\t')
kzm_drivers_all = kzm_drivers_all.rename(columns={'Tumor_Sample_Barcode': 'Sample_ID'})
kzm_drivers_low = kzm_drivers_all[kzm_drivers_all.Sample_ID.isin(low_samples)].copy()

driver_genes_all = [g for g in kzm_drivers_all.Hugo_Symbol.unique() if kzm_drivers_all[kzm_drivers_all.Hugo_Symbol==g].Sample_ID.nunique()>9]
driver_genes_low = [g for g in kzm_drivers_low.Hugo_Symbol.unique() if kzm_drivers_low[kzm_drivers_low.Hugo_Symbol==g].Sample_ID.nunique()>9]

drivers_all = pd.DataFrame(0, index=kzm611_md.index, columns=driver_genes_all)
drivers_low = pd.DataFrame(0, index=kzm611_md.index, columns=driver_genes_low)

for g in driver_genes_all:
    samples = kzm_drivers_all[kzm_drivers_all.Hugo_Symbol==g].Sample_ID.tolist()
    drivers_all[g] = [1 if x in samples else 0 for x in drivers_all.index]

for g in driver_genes_low:
    samples = kzm_drivers_low[kzm_drivers_low.Hugo_Symbol==g].Sample_ID.tolist()
    drivers_low[g] = [1 if x in samples else 0 for x in drivers_low.index]

## Load signatures
kzm611_sigs = pd.read_csv(os.path.join(root_dir, 'source_data/KZM_signatures.tsv'), sep='\t', index_col=0)
kzm611_sigs_rel = kzm611_sigs.div(kzm611_sigs.sum(axis=1), axis=0)
kzm611_sigs_low = kzm611_sigs.loc[low_samples].copy()

kzm611_sigs_01 = kzm611_sigs.copy()
kzm611_sigs_01 = kzm611_sigs_01.apply(lambda x: [1 if y > 0 else 0 for y in x])
kzm611_sigs_01.columns = [x.replace('/', '') for x in kzm611_sigs_01.columns]

cnv_sigs = pd.read_csv(os.path.join(root_dir, 'source_data/KZM_CN48_sigs.txt'), sep='\t', index_col=0)
sv_sigs = pd.read_csv(os.path.join(root_dir, 'source_data/KZM_SV32_sigs.txt'), sep='\t', index_col=0)

sv_sigs_01 = sv_sigs.apply(lambda x: [1 if y > 0 else 0 for y in x])
cnv_sigs_01 = cnv_sigs.apply(lambda x: [1 if y > 0 else 0 for y in x])

kzm_all_sigs_01 = pd.concat([kzm611_sigs_01, sv_sigs_01, cnv_sigs_01], axis=1)
kzm_low_sigs_01 = kzm_all_sigs_01.loc[low_samples].copy()

## Load & process therapy data
therapy_categs = pd.read_csv('/hpf/largeprojects/adam/mehdi/ml_mutSigs/data/kzm_data/Therapy_categories.tsv', sep='\t')
drug_class_dict = dict(zip(therapy_categs.Drug, therapy_categs.Class))
drug_subclass_dict = dict(zip(therapy_categs.Drug, therapy_categs.Subclass))

labels_dict = {'Alkylating agent, Other': 'Other Alkylators',
               'Anthracyclines': 'Anthracyclines',
               'Antimetabolite': 'Antimetabolites',
               'Camptothecin analogs': 'Camptothecins',
               'Hydrazines and Triazines': 'Hydrazines',
               'Mustard gas derivative': 'Mustard Gases',
               'Platinums': 'Platinums',
               'Podophyllotoxins': 'Podophyllotoxins',
               'Vinca alkaloids': 'Vinca Alkaloids',
               'Radiation': 'Radiation'}

# Subset the drug name dictionary based on the classes above
# To reduce the number of drugs for the logreg analysis
subset_dict = {k: v for k, v in drug_subclass_dict.items() if v in list(labels_dict.keys())}

## Load therapy data at class and individual drug levels
class_df = pd.read_csv(os.path.join(root_dir, 'source_data/KZM_therapy_classes_NY.txt'), sep='\t', index_col=0)
drugs_df = pd.read_csv(os.path.join(root_dir, 'source_data/KZM_therapy_individuals_counts.txt'), sep='\t', index_col=0)

drugs_df = drugs_df.apply(lambda x: [1 if y > 0 else 0 for y in x])
drugs_df = drugs_df.drop([x for x in drugs_df.columns if 1 not in drugs_df[x].tolist()], axis=1)
drugs_df_10p = drugs_df.loc[:, drugs_df.sum()>9].copy()

drugs_agent = drugs_df[[x for x in subset_dict.keys() if x in drugs_df.columns.tolist()]].copy()
drugs_agent_5p = drugs_agent.loc[:, drugs_agent.sum()>4].copy()
drugs_agent_10p = drugs_agent.loc[:, drugs_agent.sum()>9].copy()
drugs_agent_20p = drugs_agent.loc[:, drugs_agent.sum()>19].copy()

drugs_df_class = drugs_df[[x for x in drugs_df.columns if x in list(drug_subclass_dict.keys())]].copy()

drugs_df_class.columns = [drug_subclass_dict[x] for x in drugs_df_class.columns]
drugs_df_class = drugs_df_class.groupby(drugs_df_class.columns, axis=1).max()
drugs_df_class['Radiation'] = [1 if kzm611_md.loc[x, 'Radiation']=='Y' else 0 for x in drugs_df_class.index]

drugs_class = drugs_df_class[list(labels_dict.keys())].copy()
drugs_class.columns = drugs_class.columns.map(lambda x: labels_dict[x])
drugs_class.columns = [x.replace(' ', '') for x in drugs_class.columns]

drugs_class_low = drugs_class.loc[[x for x in drugs_class.index if x in low_samples]].copy()
drugs_agent_10p_low = drugs_agent_10p.loc[[x for x in drugs_agent_10p.index if x in low_samples]].copy()



def get_sig_coefs(model, X, sig):
    """
    Extracts and filters logistic regression significant coefficients from a trained model.

    This function retrieves the coefficients from a fitted model, removes the intercept, 
    filters out coefficients with an absolute value below a threshold (0.7, corresponding 
    to an approximate odds ratio of 2.01), and sorts the remaining coefficients in descending 
    order of absolute magnitude.

    Parameters:
    -----------
    model : sklearn-like fitted model
        A trained model with a `coef_` attribute containing feature coefficients.
    X : pandas.DataFrame
        The feature matrix used for training the model, with column names corresponding to features.
    sig : str
        A label representing the outcome or signature associated with the extracted coefficients.

    Returns:
    --------
    sig_coefs : pandas.DataFrame
        A DataFrame containing the filtered and sorted significant coefficients.
        Columns:
        - 'Features': Feature names with significant coefficients.
        - 'Coefficient': Corresponding coefficient values.
        - 'Outcome': The outcome label provided in `sig`.

    Example:
    --------
    >>> model = LogisticRegression().fit(X_train, y_train)
    >>> get_sig_coefs(model, X_train, "Cancer Risk")
    """
    coefficients = model.coef_[0]
    features = X.columns

    intercept_indices = features != 'Intercept'
    features = np.array(features)[intercept_indices]
    coefficients = coefficients[intercept_indices]

    # Filter out zero coefficients
    non_zero_indices = np.abs(coefficients) > 0.7 ## approximate odds ratio of about 2.01
    non_zero_features = np.array(features)[non_zero_indices]
    non_zero_coefficients = coefficients[non_zero_indices]

    # Sort non-zero coefficients by absolute magnitude
    sorted_indices = np.argsort(np.abs(non_zero_coefficients))[::-1]
    sorted_features = non_zero_features[sorted_indices]
    sorted_coefficients = non_zero_coefficients[sorted_indices]

    sig_coefs = pd.concat([pd.Series(sorted_features), pd.Series(sorted_coefficients)], axis=1)
    sig_coefs.columns = ['Features', 'Coefficient']
    sig_coefs['Outcome'] = sig

    return sig_coefs


def regu_logreg(preds_df, outs_df, outcome, interactions=False, predictors=False, plot=False, verbose=False):
    """
    Performs regularized logistic regression with optional interaction terms and permutation testing.

    This function fits a logistic regression model with L1 (Lasso) regularization on the provided 
    predictor variables to classify the specified outcome. It also performs a permutation test to 
    assess the statistical significance of the observed AUC.

    Parameters:
    -----------
    preds_df : pandas.DataFrame
        A DataFrame containing predictor variables.
    outs_df : pandas.DataFrame
        A DataFrame containing the outcome variable.
    outcome : str
        The name of the outcome variable in `outs_df`.
    interactions : bool, optional (default=False)
        If True, includes interaction terms in the regression formula.
    predictors : list or bool, optional (default=False)
        A list of predictor variable names. If False, all columns from `preds_df` are used.
    plot : bool, optional (default=False)
        Placeholder for potential plotting functionality (currently unused).
    verbose : bool, optional (default=False)
        If True, prints detailed output including model performance and coefficients.

    Returns:
    --------
    coefs : pandas.DataFrame
        A DataFrame containing significant coefficients, AUC, and permutation p-value.
        Columns:
        - 'Features': Feature names with significant coefficients.
        - 'Coefficient': Corresponding coefficient values.
        - 'Outcome': The outcome variable.
        - 'AUC': The observed area under the ROC curve (AUC).
        - 'pvalue': The p-value from the permutation test.

    Notes:
    ------
    - Uses `patsy.dmatrices` to construct the regression formula.
    - Applies L1 regularization (Lasso) with `liblinear` solver.
    - Performs 5000 permutation tests to calculate the p-value.
    - If the outcome variable has only one class or fewer than 5 occurrences of the minority class, the function returns None.

    Example:
    --------
    >>> model_results = regu_logreg(preds_df, outs_df, "DiseaseStatus", interactions=True, verbose=True)
    >>> print(model_results)
    """
    if not predictors:
        predictors = preds_df.columns.tolist()
    
    if len(outs_df[outcome].value_counts()) == 1 or outs_df[outcome].value_counts().min() < 5:
        print(outcome)
        return None

    df = pd.concat([outs_df, preds_df, kzm611_md[['Age_days', 'Sex', 'Purity']]], axis=1)
    df = df.dropna(axis=0)
    
    if interactions:
        formula = f"{outcome} ~ ({' + '.join(predictors)})**2 + Purity + Age_days + Sex"
        #formula = "outcome ~ (drug1 + drug2 + drug3 + drug4 + drug5)**2 + Age_days + Sex + Purity"
    else:
        formula = f"{outcome} ~ {' + '.join(predictors)} + Purity + Age_days + Sex"

    y, X = dmatrices(formula, df, return_type='dataframe')
    #print(X.columns)

    # Convert y to a 1D numpy array
    y = np.ravel(y)

    # Step 2: Fit logistic regression on observed data
    #model = LogisticRegression(max_iter=1000)
    model = LogisticRegression(penalty='l1', solver='liblinear', C=1.0, random_state=42)
    
    model.fit(X, y)
    observed_auc = roc_auc_score(y, model.predict_proba(X)[:, 1])

    '''if observed_auc < 0.75:
        return model
    for feature, coef in zip(X.columns, model.coef_[0]):
        if coef >= 1:
            print(f"{feature}: {coef:.4f}")'''
    
    coefs = get_sig_coefs(model, X, outcome)
    # Results
    print(f"============================== {outcome}")
    if verbose:
        print(f"Observed AUC: {observed_auc:.4f}")
    if len(coefs) == 0:
        return model
    
    if verbose:
        print(coefs)


    # Step 3: Permutation test
    n_permutations = 5000
    permuted_aucs = []

    for _ in range(n_permutations):
        # Permute the outcome variable
        y_permuted = shuffle(y, random_state=None)
        
        # Fit the model on permuted data
        model.fit(X, y_permuted)
        permuted_auc = roc_auc_score(y_permuted, model.predict_proba(X)[:, 1])
        permuted_aucs.append(permuted_auc)

    # Step 4: Calculate p-value
    permuted_aucs = np.array(permuted_aucs)
    p_value = np.mean(permuted_aucs >= observed_auc)
    print(f"Permutation Test P-Value: {p_value:.4f}")


    coefs['AUC'] = observed_auc
    coefs['pvalue'] = p_value

    '''if p_value < pval and observed_auc > auc:
        return sig_coefs
    else:
        return model'''
    
    return coefs



def run_model(preds, outs, condition, interactions, pval=0.05):
    """
    Runs a regularized logistic regression model for multiple outcomes and filters significant results.

    This function iterates through the columns of `outs` (outcome variables), fits a logistic 
    regression model for each using `regu_logreg`, and aggregates significant coefficients based 
    on a specified p-value threshold.

    Parameters:
    -----------
    preds : pandas.DataFrame
        A DataFrame containing predictor variables.
    outs : pandas.DataFrame
        A DataFrame containing multiple outcome variables.
    condition : str
        A label specifying the condition under which the model is run.
    interactions : bool
        If True, includes interaction terms in the regression formula.
    pval : float, optional (default=0.05)
        The significance threshold for filtering results.

    Returns:
    --------
    pandas.DataFrame
        A DataFrame containing significant coefficients across all outcome variables.
        Columns:
        - 'Features': Feature names with significant coefficients.
        - 'Coefficient': Corresponding coefficient values.
        - 'Outcome': The outcome variable.
        - 'AUC': The observed area under the ROC curve (AUC).
        - 'pvalue': The p-value from the permutation test.
        - 'Condition': The specified condition.

    Notes:
    ------
    - Calls `regu_logreg` for each outcome variable in `outs`.
    - Only includes results where the permutation test p-value is ≤ `pval`.
    - If no significant coefficients are found, an empty DataFrame is returned.

    Example:
    --------
    >>> significant_results = run_model(predictors_df, outcomes_df, "CancerTypeA", interactions=True, pval=0.01)
    >>> print(significant_results)
    """
    all_coefs = pd.DataFrame()

    for out in outs.columns: 
        coefs = regu_logreg(preds, outs, out, interactions=interactions)
        if isinstance(coefs, pd.DataFrame) and len(coefs) > 0:
            coefs['Condition'] = condition
            all_coefs = pd.concat([all_coefs, coefs], axis=0).reset_index(drop=True)
    
    return all_coefs[all_coefs.pvalue<=pval]


t0 = time.time()

final_coefs = pd.DataFrame()

##  Predictors = Therapy class / agents  ; Outcome = Signatures ; All samples ; 1+ SV/CN ; No Interactions
final_coefs = pd.concat([final_coefs, run_model(drugs_class, kzm_all_sigs_01, condition='class_sigs01_noX_all', interactions=False)], axis=0)
final_coefs = pd.concat([final_coefs, run_model(drugs_agent_10p, kzm_all_sigs_01, condition='agent_sigs01_noX_all', interactions=False)], axis=0)

##  Predictors = Therapy class / agents  ; Outcome = Signatures ; All samples ; 1+ SV/CN ; with Interactions
final_coefs = pd.concat([final_coefs, run_model(drugs_class, kzm_all_sigs_01, condition='class_sigs01_X_all', interactions=True)], axis=0)
final_coefs = pd.concat([final_coefs, run_model(drugs_agent_10p, kzm_all_sigs_01, condition='agent_sigs01_X_all', interactions=True)], axis=0)

##  Predictors = Signatures     ; Outcome = Driver genes ; Low samples
final_coefs = pd.concat([final_coefs, run_model(kzm_low_sigs_01[[x for x in kzm_low_sigs_01.columns if x.startswith('SBS')]], drivers_low, condition='sbs_driver_noX_low', interactions=False)], axis=0)
final_coefs = pd.concat([final_coefs, run_model(kzm_low_sigs_01[[x for x in kzm_low_sigs_01.columns if x.startswith('DBS')]], drivers_low, condition='dbs_driver_noX_low', interactions=False)], axis=0)
final_coefs = pd.concat([final_coefs, run_model(kzm_low_sigs_01[[x for x in kzm_low_sigs_01.columns if x.startswith('ID')]], drivers_low, condition='id_driver_noX_low', interactions=False)], axis=0)
final_coefs = pd.concat([final_coefs, run_model(kzm_low_sigs_01[[x for x in kzm_low_sigs_01.columns if x.startswith('SV')]], drivers_low, condition='sv_driver_noX_low', interactions=False)], axis=0)
final_coefs = pd.concat([final_coefs, run_model(kzm_low_sigs_01[[x for x in kzm_low_sigs_01.columns if x.startswith('CN')]], drivers_low, condition='cnv_driver_noX_low', interactions=False)], axis=0)


t1 = time.time()
total_n = t1-t0
print(f'Time elapsed: {total_n/3600} hours!')

final_coefs.to_csv(os.path.join(root_dir, 'source_data/logreg_coefs.tsv'), sep='\t')




