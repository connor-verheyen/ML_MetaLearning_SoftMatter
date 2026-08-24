"""
utils.py — shared foundation for the Paper 2 analysis (ML_MetaLearning_SoftMatter).

Structure:
  1. Paths & config           (Claude — infrastructure, ready to use)
  2. Loading & task helpers    (Claude — infrastructure, ready to use)
  3. Plotting defaults         (Claude — infrastructure, ready to use)
  4. ANALYSIS-TRANSFORM funcs   (YOU — paste your verified working-notebook functions into the
                                 marked slots; per-slot porting notes below)

FAITHFUL-PORT RULE: paste your functions verbatim EXCEPT for the two minimal, necessary adaptations
noted in the slots (remove the iteration remap; turn two enclosing-scope globals into parameters).
Do NOT fold in optimizations yet (single-pass groupby, etc.) — those come as separate verified steps
AFTER we confirm this reproduces 19.573M instances and the (135360, 48) frame.
"""
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ============================================================================
# 1. PATHS & CONFIG
# ============================================================================
# utils.py lives at the repo root -> its folder IS the repo root.
REPO_ROOT = Path(__file__).resolve().parent

def _first_existing(cands):
    for c in cands:
        if c.exists():
            return c
    return None

# Data folder holds the *.csv.gz. Confirmed layout: <repo>/data/ (files directly inside).
DATA_DIR = _first_existing([
    REPO_ROOT / "data" / "metalearning_output_datasets",
    REPO_ROOT / "data",
    REPO_ROOT / "metalearning_output_datasets",
])
if DATA_DIR is None or not any(DATA_DIR.glob("*.csv.gz")):
    raise FileNotFoundError(f"No *.csv.gz data folder found under {REPO_ROOT}")

# Dataset-info sheet (File_name, Prob_type, Range, ...).
INFO_PATH = _first_existing([
    DATA_DIR / "basic_dataset_info.xlsx",
    REPO_ROOT / "metadata" / "basic_dataset_info.xlsx",
    REPO_ROOT / "basic_dataset_info.xlsx",
])

KINDS = ("in_agg", "in_splits", "out_agg", "out_splits")


# ============================================================================
# 2. LOADING & TASK HELPERS
# ============================================================================
def load_raw(task, kind, data_dir=None):
    """Load one raw source file for a task/kind. gzip is inferred from the .csv.gz extension.
    NOTE: source data is already corrected (ext_qual_binary iteration is {0,1,2}) — no remap here."""
    data_dir = data_dir or DATA_DIR
    if kind not in KINDS:
        raise ValueError(f"kind must be one of {KINDS}, got {kind!r}")
    path = data_dir / f"{task}__{kind}.csv.gz"
    if not path.exists():
        raise FileNotFoundError(f"{path}\n(DATA_DIR = {data_dir})")
    return pd.read_csv(path)

def list_tasks(data_dir=None):
    data_dir = data_dir or DATA_DIR
    names = set()
    for f in data_dir.glob("*.csv.gz"):
        stem = f.name[:-len(".csv.gz")]
        for kind in KINDS:
            if stem.endswith("__" + kind):
                names.add(stem[:-len("__" + kind)])
    return sorted(names)

def load_basic_info(info_path=None):
    info_path = info_path or INFO_PATH
    if info_path is None:
        raise FileNotFoundError("basic_dataset_info.xlsx not found (checked data/, metadata/, repo root).")
    return pd.read_excel(info_path, index_col=0)

def split_problem_lists(file_names, basic_info):
    """Return (classification_tasks, regression_tasks) using basic_info['Prob_type']."""
    clf, reg = [], []
    for f in file_names:
        pt = basic_info[basic_info["File_name"] == f]["Prob_type"].iloc[0]
        if pt in ("Binary Classification", "Multiclass Classification"):
            clf.append(f)
        elif pt in ("Cross-Section Regression", "Panel Regression"):
            reg.append(f)
    return clf, reg


# ============================================================================
# 3. PLOTTING DEFAULTS (Frontiers)
# ============================================================================
# Single column = 85 mm, double column = 180 mm; 300 dpi; text >= 8 pt.
MM = 1 / 25.4

def frontiers_style():
    plt.rcParams.update({
        "figure.dpi": 300, "savefig.dpi": 300, "savefig.bbox": "tight",
        "font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8,
        "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
        "axes.linewidth": 0.6,
    })

def figsize(cols=1, height_mm=70):
    return (( 85 if cols == 1 else 180) * MM, height_mm * MM)


# ============================================================================
# 4. ANALYSIS-TRANSFORM FUNCTIONS  —  PASTE YOUR VERIFIED FUNCTIONS BELOW
# ============================================================================
# Port each from your working notebook. Notes mark the ONLY changes to make.

# --- SLOT: define_problem_type(basic_info, file_name) ----------------------
#     Move VERBATIM. (Optional hardening later: guard the .iloc[0].)
# Function to define the problem type and set of scoring metrics based on the filename
def define_problem_type(basic_info, file_name):

    # Extract the problem type (note: file_name cannot have the suffix (e.g. __in_splits or __out_splits) attached to it)
    problem_type = basic_info[basic_info['File_name'] == file_name]['Prob_type'].iloc[0]

    # Define model evaluation metrics according to problem type
    if problem_type == 'Binary Classification':
        scoring = ['accuracy', 'precision', 'recall', 'f1', 'balanced_accuracy', 'average_precision']

    elif problem_type == 'Multiclass Classification':
        scoring = ['accuracy', 'precision', 'recall', 'f1', 'balanced_accuracy']

    else:
        scoring = ['neg_mean_absolute_error', 'neg_median_absolute_error', 'neg_mean_absolute_percentage_error', 'r2', 'explained_variance', 'max_error']

    return problem_type, scoring

# --- SLOT: remove_erroneous_results(...) -----------------------------------
#     Move the FIXED version (the .mask(...) assign-back form, NOT inplace).
# Function to filter out failed model instances
def remove_erroneous_results(in_splits, out_splits, basic_info, file_name, problem_type, mult_factor):

    # Get the range of the outputs for the input dataset
    output_range = basic_info[basic_info['File_name'] == file_name]['Range'].values[0]

    # These model instances have MAE values more extreme than 2x the total range of actual outputs that could be predicted (will completely skew analysis)
    # These instances also have MAPE values more extreme than 1000% and r2 values more extreme than -1000
    in_splits_removed = in_splits[(in_splits['neg_mean_absolute_error'] < (-2*output_range)) & (in_splits['neg_mean_absolute_percentage_error'] < -1e3) & (in_splits['r2'] < -1e3)]
    out_splits_removed = out_splits[(out_splits['neg_mean_absolute_error'] < (-2*output_range)) & (out_splits['neg_mean_absolute_percentage_error'] < -1e3) & (out_splits['r2'] < -1e3)]

    # These model instances have MAE and MedAE within 2x the range of outputs and have MAPE<1000% and r2>-1000 (any erroneous values are replaced by NaN and ignored by pandas and seaborn)
    #in_splits = in_splits[(in_splits['neg_mean_absolute_error']>(-2*output_range)) & (in_splits['neg_mean_absolute_percentage_error']>-1e3) & (in_splits['r2']>-1e3)]
    #out_splits = out_splits[(out_splits['neg_mean_absolute_error']>(-2*output_range)) & (out_splits['neg_mean_absolute_percentage_error']>-1e3) & (out_splits['r2']>-1e3)]

    # Assign back instead of .mask(..., inplace=True): chained inplace masking is unreliable under
    # pandas 2.x (can silently no-op / raises ChainedAssignmentError). Same result, done safely.
    in_splits['neg_mean_absolute_error'] = in_splits['neg_mean_absolute_error'].mask(in_splits['neg_mean_absolute_error'] < (-2*output_range))
    in_splits['neg_median_absolute_error'] = in_splits['neg_median_absolute_error'].mask(in_splits['neg_median_absolute_error'] < (-2*output_range))
    in_splits['neg_mean_absolute_percentage_error'] = in_splits['neg_mean_absolute_percentage_error'].mask(in_splits['neg_mean_absolute_percentage_error'] < -1e3)
    in_splits['r2'] = in_splits['r2'].mask(in_splits['r2'] < -1e3)
    in_splits['explained_variance'] = in_splits['explained_variance'].mask(in_splits['explained_variance'] < -1e3)
    in_splits['max_error'] = in_splits['max_error'].mask(in_splits['max_error'] < (-2*output_range))

    out_splits['neg_mean_absolute_error'] = out_splits['neg_mean_absolute_error'].mask(out_splits['neg_mean_absolute_error'] < (-2*output_range))
    out_splits['neg_median_absolute_error'] = out_splits['neg_median_absolute_error'].mask(out_splits['neg_median_absolute_error'] < (-2*output_range))
    out_splits['neg_mean_absolute_percentage_error'] = out_splits['neg_mean_absolute_percentage_error'].mask(out_splits['neg_mean_absolute_percentage_error'] < -1e3)
    out_splits['r2'] = out_splits['r2'].mask(out_splits['r2'] < -1e3)
    out_splits['explained_variance'] = out_splits['explained_variance'].mask(out_splits['explained_variance'] < -1e3)
    out_splits['max_error'] = out_splits['max_error'].mask(out_splits['max_error'] < (-2*output_range))

    return in_splits, out_splits, in_splits_removed, out_splits_removed

# ##################################
# # Run the remove_erroneous_results if the problem type is a regression so you can filter out failed model instances that will skew results

# --- SLOT: aggregated_stats(dataframe, scoring, stats, loop) ----------------
#     Move VERBATIM. (Optimization to single-pass groupby is a LATER step.)
# Function to calculate means, medians, stds, IQRs, for train and test 
def aggregated_stats(dataframe,scoring,stats,loop):

  df = dataframe.copy()
  
  i=0; list_for_stats = []
  # Go through each of the descriptive statistics in the list 
  for stat in stats: 

    # For the first result, save the computed stats AND all of the relevant dimensions
    if i==0:
      if loop=='inner':
        
        # Define the set of dimensions you want to groupby so you can generate aggregated stats across the remaining non-grouping columns 
        aggregator = ['iteration','outer_k','outer_split','inner_k','fold','algorithm','model_config_id','params']
        
        # For the inner loop data, drop 'inner split' columns since you're aggregating across these
        result = df.groupby(aggregator,as_index=False,sort=False).agg(stat).drop(columns={'inner_split'}).rename(columns=dict(zip(scoring,[stat+'_'+score for score in scoring])))

      if loop=='outer': 
        
        # Define the set of dimensions you want to groupby so you can generate aggregated stats across the remaining non-grouping columns 
        aggregator = ['iteration','outer_k','fold','algorithm','model_config_id','params']
        
        # For the outer loop data, drop 'outer split' columns since you're aggregating across these
        result = df.groupby(aggregator,as_index=False,sort=False).agg(stat).drop(columns={'outer_split'}).rename(columns=dict(zip(scoring,[stat+'_'+score for score in scoring])))

    # For the rest of the results, just save the computed stats 
    if i>0:
      result = df.groupby(aggregator,as_index=False,sort=False).agg(stat)[scoring].rename(columns=dict(zip(scoring,[stat+'_'+score for score in scoring])))

    # Add the computed stat to the list
    list_for_stats.append(result); i+=1

  # Compile all stats into a single aggregated dataframe
  agg_stats_df = pd.concat(list_for_stats,axis=1)

  return agg_stats_df

# ##################################
# # Use the aggregated_stats function to compute the aggregated stats for each grouping for the inner and outer loops


# --- SLOT: assign_default(dataframe, problem_type) -------------------------
#     Move, but CHANGE the signature to take `problem_type` as a PARAMETER
#     (it was a free/global variable). The driver already knows problem_type
#     per task, so pass it in.
# Function to assign default configurations based on param vectors (no preprocessing of data for standard configs)
def assign_default(dataframe, problem_type):
  
  df=dataframe.copy()
  
  # Assign the default classifier configurations 
  if problem_type == 'Binary Classification' or problem_type == 'Multiclass Classification':

    # LINEAR MODEL
    default_string = "{'algo__alpha': 0.0001, 'algo__class_weight': None, 'algo__loss': 'hinge', 'algo__penalty': 'l2', 'preprocess': 'passthrough'}"
    df.loc[(df['algorithm']=='LR') & (df['params']==default_string),'default_config']='default'

    # GAUSSIAN PROCESS
    default_string = "{'algo__kernel': RBF(length_scale=1), 'preprocess': 'passthrough'}"
    df.loc[(df['algorithm']=='GP') & (df['params']==default_string),'default_config']='default'

    # K-NEAREST NEIGHBORS
    default_string = "{'algo__n_neighbors': 5, 'algo__p': 2, 'algo__weights': 'uniform', 'preprocess': 'passthrough'}"
    df.loc[(df['algorithm']=='KNN') & (df['params']==default_string),'default_config']='default'

    # SUPPORT VECTOR MACHINE
    default_string = "{'algo__C': 1, 'algo__class_weight': None, 'algo__kernel': 'rbf', 'preprocess': 'passthrough'}"
    df.loc[(df['algorithm']=='SVM') & (df['params']==default_string),'default_config']='default'

    # RANDOM FOREST
    default_string = "{'algo__class_weight': None, 'algo__max_features': 'sqrt', 'algo__min_samples_leaf': 1, 'algo__n_estimators': 100, 'preprocess': 'passthrough'}"
    df.loc[(df['algorithm']=='RF') & (df['params']==default_string),'default_config']='default'

    # GRADIENT BOOSTING
    default_string = "{'algo__learning_rate': 0.1, 'algo__min_samples_leaf': 1, 'algo__n_estimators': 100, 'algo__subsample': 1.0, 'preprocess': 'passthrough'}"
    df.loc[(df['algorithm']=='GB') & (df['params']==default_string),'default_config']='default'

    # MULTILAYER PERCEPTRON (NEURAL NETWORK)
    default_string = "{'algo__alpha': 0.0001, 'algo__hidden_layer_sizes': (100,), 'algo__learning_rate_init': 0.001, 'preprocess': 'passthrough'}"
    df.loc[(df['algorithm']=='MLP') & (df['params']==default_string),'default_config']='default'

  # Assign the default regressor configurations
  else: 

    # LINEAR MODEL
    #default_string = "{'algo__alpha': 0.0001, 'algo__loss': 'squared_error', 'algo__penalty': 'l2', 'algo__shuffle': True, 'preprocess': 'passthrough'}" -> note: linear models failed to converge for the regression problems if preprocess==passthrough
    default_string = "{'algo__alpha': 0.0001, 'algo__loss': 'squared_error', 'algo__penalty': 'l2', 'algo__shuffle': True, 'preprocess': MinMaxScaler()}" # note: just for linear models, use MinMaxScaler() as the default
    df.loc[(df['algorithm']=='LR') & (df['params']==default_string),'default_config']='default'

    # GAUSSIAN PROCESS
    default_string = "{'algo__kernel': 1**2 * RBF(length_scale=1), 'algo__normalize_y': False, 'preprocess': 'passthrough'}"
    df.loc[(df['algorithm']=='GP') & (df['params']==default_string),'default_config']='default'

    # K-NEAREST NEIGHBORS
    default_string = "{'algo__n_neighbors': 5, 'algo__p': 2, 'algo__weights': 'uniform', 'preprocess': 'passthrough'}"
    df.loc[(df['algorithm']=='KNN') & (df['params']==default_string),'default_config']='default'

    # SUPPORT VECTOR MACHINE
    default_string = "{'algo__C': 1, 'algo__epsilon': 0.1, 'algo__kernel': 'rbf', 'preprocess': 'passthrough'}"
    df.loc[(df['algorithm']=='SVM') & (df['params']==default_string),'default_config']='default'

    # RANDOM FOREST
    default_string = "{'algo__max_features': None, 'algo__min_samples_leaf': 1, 'algo__n_estimators': 100, 'preprocess': 'passthrough'}"
    df.loc[(df['algorithm']=='RF') & (df['params']==default_string),'default_config']='default'

    # GRADIENT BOOSTING
    default_string = "{'algo__learning_rate': 0.1, 'algo__min_samples_leaf': 1, 'algo__n_estimators': 100, 'algo__subsample': 1.0, 'preprocess': 'passthrough'}"
    df.loc[(df['algorithm']=='GB') & (df['params']==default_string),'default_config']='default'

    # MULTILAYER PERCEPTRON (NEURAL NETWORK)
    default_string = "{'algo__alpha': 0.0001, 'algo__hidden_layer_sizes': (100,), 'algo__learning_rate_init': 0.001, 'preprocess': 'passthrough'}"
    df.loc[(df['algorithm']=='MLP') & (df['params']==default_string),'default_config']='default'

  # Also fill remaining NaN spaces with 'alt' 
  df['default_config'] = df['default_config'].fillna('alt')

  return df

# ##################################
# # Use the assign_default function to assign the default configurations for each algorithm

# --- SLOT: as_hpo_cash_ranks(dataframe, scoring, loop) ---------------------
#     Move VERBATIM.
# Function to perform rankings for AS (identify top-performing default algos), HPO (identify top-performing configurations on per-algorithm basis), CASH (identify top-performing algorithms AND configurations)
def as_hpo_cash_ranks(dataframe,scoring,loop):
  
  df=dataframe.copy()
  
  if loop=='inner':
    
    as_ranks = df[df['default_config']=='default'].groupby(['iteration','outer_k','outer_split','inner_k','fold',],as_index=False,sort=False).rank(ascending=False,method='min')
    as_ranks = as_ranks[['mean_'+score for score in scoring]].rename(columns=dict(zip(['mean_'+score for score in scoring],['rank_as_mean_'+score for score in scoring])))

    hpo_ranks = df.groupby(['iteration','outer_k','outer_split','inner_k','fold','algorithm',],as_index=False,sort=False).rank(ascending=False,method='min')
    hpo_ranks = hpo_ranks[['mean_'+score for score in scoring]].rename(columns=dict(zip(['mean_'+score for score in scoring],['rank_hpo_mean_'+score for score in scoring])))

    cash_ranks = df.groupby(['iteration','outer_k','outer_split','inner_k','fold',],as_index=False,sort=False).rank(ascending=False,method='min')
    cash_ranks = cash_ranks[['mean_'+score for score in scoring]].rename(columns=dict(zip(['mean_'+score for score in scoring],['rank_cash_mean_'+score for score in scoring])))
    
  if loop=='outer':
    
    as_ranks = df[df['default_config']=='default'].groupby(['iteration','outer_k','fold',],as_index=False,sort=False).rank(ascending=False,method='min')
    as_ranks = as_ranks[['mean_'+score for score in scoring]].rename(columns=dict(zip(['mean_'+score for score in scoring],['rank_as_mean_'+score for score in scoring])))

    hpo_ranks = df.groupby(['iteration','outer_k','fold','algorithm',],as_index=False,sort=False).rank(ascending=False,method='min')
    hpo_ranks = hpo_ranks[['mean_'+score for score in scoring]].rename(columns=dict(zip(['mean_'+score for score in scoring],['rank_hpo_mean_'+score for score in scoring])))

    cash_ranks = df.groupby(['iteration','outer_k','fold',],as_index=False,sort=False).rank(ascending=False,method='min')
    cash_ranks = cash_ranks[['mean_'+score for score in scoring]].rename(columns=dict(zip(['mean_'+score for score in scoring],['rank_cash_mean_'+score for score in scoring])))
    
  compiled = pd.concat([df,as_ranks,hpo_ranks,cash_ranks],axis=1)
  
  return compiled 

# ##################################
# # Use the as_hpo_cash_ranks function to perform the AS ranking, HPO ranking, and CASH rankings for each grouping for the inner and outer loops

# --- SLOT: modify_regression_datasets(dataframe) --------------------------
#     Move the FIXED version (collision-safe min/max swap for MAE/MedAE/MAPE only;
#     r2 NOT swapped).
# Function to modify the scores for the regression problems since everything is negative (for ranking) and the names are too long
def modify_regression_datasets(dataframe):

    df = dataframe.copy()

    # Go through the columns and flip the signs of the negative columns (as long as they aren't the computed stds or the ranks because those are already positive)
    for col in df.columns:
        if 'neg_' in col:
            if 'std_' in col or 'rank_' in col:
                pass
            else:
                df[col] = -1 * df[col]

    # # Replace the extended column names with shorter names
    df.columns = [col.replace('neg_mean_absolute_error', 'MAE') for col in df.columns]
    df.columns = [col.replace('neg_median_absolute_error', 'MedAE') for col in df.columns]
    df.columns = [col.replace('neg_mean_absolute_percentage_error', 'MAPE') for col in df.columns]

    # Flip the min and max columns as well (min/max were applied to the negative scoring metrics, so max becomes min and min becomes max)
    # NOTE: only the sign-FLIPPED error metrics (MAE/MedAE/MAPE) need swapping. r2 (and explained_variance)
    # were never negative, so their min/max are already correct -> do NOT swap them. Collision-safe explicit
    # swap replaces the original positional-zip rename (which was fragile AND wrongly swapped r2's min/max).
    rename_map = {}
    for m in ['MAE', 'MedAE', 'MAPE']:
        lo, hi = 'min_' + m, 'max_' + m
        if lo in df.columns and hi in df.columns:
            rename_map[lo] = '__TMP_max_' + m   # min_<m> holds the max error after the sign flip
            rename_map[hi] = 'min_' + m         # max_<m> holds the min error after the sign flip
    df = df.rename(columns=rename_map)
    df.columns = [col.replace('__TMP_', '') for col in df.columns]

    return df

# ##################################
# # If the problem type is a regression, run the function to modify the dataframes (shorten the names and flip the signs of the negative error metrics (negative was used for ranking purposes))


# --- SLOT: combined_ids(in_splits, in_agg, out_splits, out_agg) ------------
#     Move VERBATIM. (Optionally fix the stale 'outer_val' comment -> 'outer_test'.)
# Function to generate additional unique ids to help with plotting and analysis
def combined_ids(in_splits,in_agg,out_splits,out_agg): 

  # To make a combined identifier that covers both the algorithm (e.g. algorithm=LR) and the specific configuration (e.g. model_config_id=23) -> (combo_id = LR23)

  in_splits['combo_id'] = [algo+str(config_id) for algo,config_id in zip(in_splits['algorithm'],in_splits['model_config_id'])]
  in_agg['combo_id'] = [algo+str(config_id) for algo,config_id in zip(in_agg['algorithm'],in_agg['model_config_id'])]
  out_splits['combo_id'] = [algo+str(config_id) for algo,config_id in zip(out_splits['algorithm'],out_splits['model_config_id'])]
  out_agg['combo_id'] = [algo+str(config_id) for algo,config_id in zip(out_agg['algorithm'],out_agg['model_config_id'])]


  # To make a combined identifier for the location of the fold (e.g. inner_train, inner_val, outer_train, outer_val) 

  in_splits['loopfold'] = 'inner_'+in_splits['fold']; 
  in_agg['loopfold'] = 'inner_'+in_agg['fold']
  out_splits['loopfold'] = 'outer_'+out_splits['fold']
  out_agg['loopfold'] = 'outer_'+out_agg['fold']


  # To make a combined identifier for the nested cv (e.g. if outer_k is 3 and inner_k is 4 then nested_cv = 3x4)

  in_splits['nested_cv'] = [str(outer_k)+'x'+str(inner_k) for outer_k,inner_k in zip(in_splits['outer_k'],in_splits['inner_k'])]
  in_agg['nested_cv'] = [str(outer_k)+'x'+str(inner_k) for outer_k,inner_k in zip(in_agg['outer_k'],in_agg['inner_k'])]
  
  return in_splits,in_agg,out_splits,out_agg

# ##################################
# # Run the combined_ids function to generate additional unique identifiers to facilitate analysis and plotting 

# --- SLOT: combine_inner_and_outer(in_agg, out_agg, out_splits, problem_type, scores_to_analyze) ---
#     Move, but CHANGE the signature to take `scores_to_analyze` as a PARAMETER
#     (it was a free/global). Pass it from the driver.
# >>> FLAG (porting): uses undefined GLOBAL `scores_to_analyze` (like `problem_type` in
#     assign_default). Works only if the call site sets it before calling. When porting to
#     utils, make it an explicit parameter: combine_inner_and_outer(..., scores_to_analyze).

# Function to produce combined dataframes which contain information from both the outer and inner construction/evaluation schemes 

def combine_inner_and_outer(in_agg,out_agg,out_splits,problem_type,scores_to_analyze):

  # Concatenate the in_agg and out_agg data (just stack them on top of each other) -> maintain tidy format... useful for some types of analysis and plotting

  in_agg_out_agg_concat = pd.concat([in_agg,out_agg]).reset_index(drop=True)


  # Merge the in_agg and out_splits data (merge left) -> pair the top inner results with the corresponding outer split results that you would get if you used refit=True in normal search protocol
  temp = out_splits[out_splits['fold']=='test'].rename(columns=dict(zip([score for score in scores_to_analyze],['outer_'+score for score in scores_to_analyze])))

  in_agg_out_splits_merge = in_agg[in_agg['fold']=='val'].merge(temp[['iteration','outer_k','outer_split','combo_id',*['outer_'+score for score in scores_to_analyze]]],
                                                                on=['iteration','outer_k','outer_split','combo_id',],
                                                                how='left').reset_index(drop=True)

  # Merge the in_agg and out_agg data (stack them sideways) -> break tidy format and provide different names to each column -> useful for some types of analysis and plotting
  # Look at the columns and copy the regression-related scores and ranks columns (need to be present in both in_agg and out_agg); 

  if problem_type == 'Binary Classification' or problem_type == 'Multiclass Classification':
    means_and_ranks = ['mean_accuracy', 'mean_precision', 'mean_recall', 'mean_f1',
                       'median_accuracy', 'median_precision', 'median_recall', 'median_f1',
                       'std_accuracy', 'std_precision', 'std_recall', 'std_f1', 
                       'max_accuracy','max_precision', 'max_recall', 'max_f1', 
                       'min_accuracy','min_precision', 'min_recall', 'min_f1',
                       'rank_as_mean_accuracy', 'rank_as_mean_precision','rank_as_mean_recall', 'rank_as_mean_f1', 
                       'rank_hpo_mean_accuracy','rank_hpo_mean_precision', 'rank_hpo_mean_recall', 'rank_hpo_mean_f1',
                       'rank_cash_mean_accuracy', 'rank_cash_mean_precision','rank_cash_mean_recall', 'rank_cash_mean_f1',]
  
# >>> NOTE: intentionally carries a metric subset (drops balanced_accuracy/average_precision for
#     classification, explained_variance/max_error for regression) to narrow the analysis space.
    
  if problem_type=='Cross-Section Regression' or problem_type=='Panel Regression':
    means_and_ranks = ['mean_MAE', 'mean_MedAE', 'mean_MAPE','mean_r2', 
                       'median_MAE', 'median_MedAE', 'median_MAPE', 'median_r2',
                       'std_MAE', 'std_MedAE', 'std_MAPE', 'std_r2', 
                       'max_MAE', 'max_MedAE','max_MAPE', 'max_r2', 
                       'min_MAE', 'min_MedAE', 'min_MAPE', 'min_r2', 
                       'rank_as_mean_MAE', 'rank_as_mean_MedAE','rank_as_mean_MAPE', 'rank_as_mean_r2', 
                       'rank_hpo_mean_MAE','rank_hpo_mean_MedAE', 'rank_hpo_mean_MAPE', 'rank_hpo_mean_r2',
                       'rank_cash_mean_MAE', 'rank_cash_mean_MedAE', 'rank_cash_mean_MAPE','rank_cash_mean_r2',]

  # Make temporary dataframes for each fold of in_agg where the means and ranks columns have been renamed (based on the inner fold (train or val))
  temp_itrn=in_agg[in_agg['fold']=='train'].rename(columns=dict(zip([col for col in means_and_ranks],['itrn_'+col for col in means_and_ranks])))
  temp_ival=in_agg[in_agg['fold']=='val'].rename(columns=dict(zip([col for col in means_and_ranks],['ival_'+col for col in means_and_ranks])))

  # Merge the two renamed inner dataframes 
  temp_in = temp_itrn.merge(temp_ival[['iteration','outer_k','outer_split','inner_k','combo_id',*['ival_'+col for col in means_and_ranks]]],
                            on=['iteration','outer_k','outer_split','inner_k','combo_id'],how='left').reset_index(drop=True)

  # Make temporary dataframes for each fold of out_agg where the means and ranks columns have been renamed (based on the outer fold (train or test))
  temp_otrn=out_agg[out_agg['fold']=='train'].rename(columns=dict(zip([col for col in means_and_ranks],['otrn_'+col for col in means_and_ranks])))
  temp_otst=out_agg[out_agg['fold']=='test'].rename(columns=dict(zip([col for col in means_and_ranks],['otst_'+col for col in means_and_ranks])))

  # Merge the two renamed outer dataframes 
  temp_out = temp_otrn.merge(temp_otst[['iteration','outer_k','combo_id',*['otst_'+col for col in means_and_ranks]]],
                             on=['iteration','outer_k','combo_id'],how='left').reset_index(drop=True)

# >>> FLAG (analysis): outer means/ranks are DUPLICATED across inner splits and inner_k values
#     when broadcast onto the finer inner rows (see note below). De-duplicate on the outer keys
#     (iteration, outer_k, combo_id) BEFORE averaging/counting over this frame, or outer values
#     get multiply-counted. Applies to anything built on in_agg_out_agg_merge.
    
  # Put everything together -> now you can follow the aggregated scores and ranks across all folds (train-val-train-test) across all iterations and all outer k values 
  # Note, outer means/ranks are duplicated for different outer splits and different inner k values because those dimensions are only applicable to the inner loops
  in_agg_out_agg_merge = temp_in.merge(temp_out[['iteration','outer_k','combo_id',*['otrn_'+col for col in means_and_ranks],*['otst_'+col for col in means_and_ranks]]],
                                       on=['iteration','outer_k','combo_id'],how='left').reset_index(drop=True)
  
  return in_agg_out_agg_concat, in_agg_out_splits_merge, in_agg_out_agg_merge

# ##################################
# # Run combine_inner_and_outer to generate combined dataframes (simple concatention of inner_agg and outer_agg, merging of inner_agg and outer_splits for performance evaluations, merging of in_agg and out_agg with performance evaluations and rankings)


# --- SLOT: process_all_tasks(...)  (your driver, cell 12, as a function) ----
#     Move the driver loop into a function that returns the dicts. REMOVE the
#     ext_qual_binary iteration remap (now fixed at source). It should set
#     problem_type and scores_to_analyze per task and pass them explicitly to
#     assign_default / combine_inner_and_outer. Return the 9 dicts, e.g.:
#         return dict(problem_type=problem_type_dict, scores=scores_to_analyze_dict,
#                     in_splits=in_splits_dict, out_splits=out_splits_dict,
#                     in_agg=in_agg_dict, out_agg=out_agg_dict,
#                     ia_oa_concat=..., ia_os_merge=..., ia_oa_merge=...)

### TURNED PROCESSING CELL INTO SELF-CONTAINED FUNCTION 

def process_all_tasks(file_names=None, basic_info=None):
    if file_names is None:
        file_names = list_tasks()
    if basic_info is None:
        basic_info = load_basic_info()

    # build df_dict 
    df_dict = {}
    for f in file_names:
        df_dict[f + '__in_splits']  = load_raw(f, 'in_splits')
        df_dict[f + '__out_splits'] = load_raw(f, 'out_splits')

    # Go through each file and produce the processed datasets (in_splits,out_splits,in_agg,out_agg,in_agg_out_agg_concat,in_agg_out_splits_merge,in_agg_out_agg_merge)
    
    # Initialize the dicts to hold all of the results
    problem_type_dict={}; scores_to_analyze_dict={}; in_splits_dict={};out_splits_dict={};in_agg_dict={};out_agg_dict={}; in_agg_out_agg_concat_dict={}; in_agg_out_splits_merge_dict={}; in_agg_out_agg_merge_dict={};
    
    # Iterate through and assign file names 
    for file_name in file_names:
        print('---',file_name)
        
        ################################## 
        # Get the split-by-split inner and outer data from the appropriate dicts (the datasets) and rearrange the datasets so you have train,val for the inner and train,test for the outer 
        in_splits = df_dict[file_name+'__in_splits']; out_splits = df_dict[file_name+'__out_splits']; 
        in_splits = pd.concat([in_splits[in_splits['fold']=='train'],in_splits[in_splits['fold']=='val']]); out_splits = pd.concat([out_splits[out_splits['fold']=='train'],out_splits[out_splits['fold']=='test']])
        
        ##################################
        # Run the define_problem_type function to assign the ML problem type and corresponding scoring metrics 
        problem_type, scoring = define_problem_type(basic_info,file_name)
        
        ##################################
        # Run the remove_erroneous_results (if the problem type is a regression) so you can filter out failed model instances that will skew results
        if problem_type=='Cross-Section Regression' or problem_type=='Panel Regression':
            mult_factor=2; in_splits,out_splits,_,_ = remove_erroneous_results(in_splits,out_splits,basic_info,file_name,problem_type,mult_factor)
        
        ##################################
        # Run the aggregated_stats function to compute the aggregated stats for each grouping for the inner and outer loops (need to define the stats you want and the scoring metrics that you actually want to analyze)
        stats = ['mean','median','std','max','min']; scores_to_analyze = scoring[:4]
        in_agg = aggregated_stats(in_splits,scores_to_analyze,stats,'inner').drop(columns=[score for score in scoring if score not in scores_to_analyze]); out_agg = aggregated_stats(out_splits,scores_to_analyze,stats,'outer').drop(columns=[score for score in scoring if score not in scores_to_analyze])
        
        ##################################
        # Run the assign_default function to assign the default configurations for each algorithm
        in_agg = assign_default(in_agg,problem_type); out_agg = assign_default(out_agg,problem_type)
        
        ##################################
        # Run the as_hpo_cash_ranks function to perform the AS ranking, HPO ranking, and CASH rankings for each grouping for the inner and outer loops
        in_agg = as_hpo_cash_ranks(in_agg,scores_to_analyze,'inner'); out_agg = as_hpo_cash_ranks(out_agg,scores_to_analyze,'outer')
        
        ##################################
        # If the problem type is a regression, run the modify_regression_datasets function to shorten the score names and flip the signs of the negative error metrics (negative was used for ranking purposes)) -> also redefine the scores_to_analyze list
        if problem_type=='Cross-Section Regression' or problem_type=='Panel Regression':
            in_splits = modify_regression_datasets(in_splits); in_agg = modify_regression_datasets(in_agg); out_splits = modify_regression_datasets(out_splits); out_agg = modify_regression_datasets(out_agg); scores_to_analyze = ['MAE','MedAE','MAPE','r2']
        
        ##################################
        # Run the combined_ids function to generate additional unique identifiers to facilitate analysis and plotting 
        in_splits,in_agg,out_splits,out_agg = combined_ids(in_splits,in_agg,out_splits,out_agg)
        
        ##################################
        # Run combine_inner_and_outer to generate combined dataframes (simple concatention of inner_agg and outer_agg, merging of inner_agg and outer_splits for performance evaluations, merging of in_agg and out_agg with performance evaluations and rankings)
        in_agg_out_agg_concat, in_agg_out_splits_merge, in_agg_out_agg_merge = combine_inner_and_outer(in_agg,out_agg,out_splits,problem_type,scores_to_analyze)
        
        ##################################
        # Add the processed outputs/dataframes to their corresponding dicts and pair them with the appropriate filename key 
        problem_type_dict[file_name]=problem_type; scores_to_analyze_dict[file_name]=scores_to_analyze;
        in_splits_dict[file_name]=in_splits; out_splits_dict[file_name]=out_splits; in_agg_dict[file_name]=in_agg; out_agg_dict[file_name]=out_agg
        in_agg_out_agg_concat_dict[file_name]=in_agg_out_agg_concat; in_agg_out_splits_merge_dict[file_name]=in_agg_out_splits_merge; in_agg_out_agg_merge_dict[file_name]=in_agg_out_agg_merge;

    return {
        'problem_type':  problem_type_dict,
        'scores':        scores_to_analyze_dict,
        'in_splits':     in_splits_dict,
        'out_splits':    out_splits_dict,
        'in_agg':        in_agg_dict,
        'out_agg':       out_agg_dict,
        'ia_oa_concat':  in_agg_out_agg_concat_dict,
        'ia_os_merge':   in_agg_out_splits_merge_dict,
        'ia_oa_merge':   in_agg_out_agg_merge_dict,
    }