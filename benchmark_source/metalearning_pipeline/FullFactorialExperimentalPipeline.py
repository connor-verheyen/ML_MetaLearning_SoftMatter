#!/usr/bin/env python
# coding: utf-8

# In[3]:


# To resize the notebook
from IPython.display import display, HTML
display(HTML("<style>.container { width:100% !important; }</style>"))


# In[2]:


# Import basic libraries 
import os 
import pandas as pd 
import numpy as np 
import scipy as sp 
import math

# Use the time library to time the program
import time

# Commands to check CPU info 
#import multiprocessing #multiprocessing.cpu_count() #!cat /proc/cpuinfo | grep 'model name' #!cat /proc/cpuinfo | grep 'processor' | wc -l #!lscpu

# To list items in the directory 
#%ls raw_data/


# In[3]:


# Assign the filename of the dataset 
file_name = 'rheo_osc_stress_storage_curves'


# In[4]:


# Read in the pre-processed dataset 
data = pd.read_excel('raw_data/'+file_name+'.xlsx',index_col=0) 

# Assign X values 
X = data.iloc[:,:-2].reset_index(drop=True)

# Assign y values 
y = data.iloc[:,-2].reset_index(drop=True)

# Assign groups 
groups = data.iloc[:,-1].reset_index(drop=True)


# In[5]:


# Automatically determine whether the problem is a classification or regression task and assign it to a variable (which is used to select algorithms, scoring metrics, and CV schemes)
def determine_ML_task(y):
  if y.unique().shape[0] <= 5:
    problem_type='classification'
  else:
    problem_type='regression'
  return problem_type

problem_type = determine_ML_task(y); print(problem_type)


# In[6]:


# Imports for models 
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import SGDClassifier, SGDRegressor
from sklearn.naive_bayes import CategoricalNB
from sklearn.gaussian_process import GaussianProcessClassifier, GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, DotProduct, RationalQuadratic, Matern, ConstantKernel, WhiteKernel
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.svm import SVC, SVR
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor

# Dict to hold the models 
algo_dict = {}; grid_dict = {}

random_state=0; n_jobs=None

### LINEAR MODEL ###
if problem_type == 'classification': # Default = 'hinge','l2',0.0001,None
  name = 'LR'; algo = SGDClassifier(random_state=random_state); algo_dict[name]=algo 
  loss = ['hinge','log','modified_huber']; penalty = ['l1','l2']; alpha = [0.00001,0.0001,0.001]; class_weight = [None,'balanced'] # log_loss = log for older versions of sklearn
  grid = dict(loss=loss,penalty=penalty,alpha=alpha,class_weight=class_weight); 
  grid_dict[name] = grid 

else: # Default = 'squared_error','l2',0.0001,True
  name = 'LR'; algo = SGDRegressor(random_state=random_state); algo_dict[name]=algo
  loss = ['squared_error','huber','epsilon_insensitive']; penalty = ['l1','l2']; alpha = [0.00001,0.0001,0.001]; shuffle = [True,False]
  grid = dict(loss=loss,penalty=penalty,alpha=alpha,shuffle=shuffle); grid_dict[name] = grid 

### GAUSSIAN PROCESS ### -> note: removed DotProduct() from regression
if problem_type == 'classification': # Default = RBF()
  name = 'GP'; algo = GaussianProcessClassifier(random_state=random_state,n_jobs=n_jobs); algo_dict[name]=algo
  kernel=[RBF(), DotProduct(), RationalQuadratic(), Matern()]
  grid = dict(kernel=kernel); grid_dict[name] = grid 

else: # Default = ConstantKernel(1.0, constant_value_bounds="fixed") * RBF(1.0, length_scale_bounds="fixed"), False
  name = 'GP'; algo = GaussianProcessRegressor(random_state=random_state,); algo_dict[name]=algo
  kernel=[ConstantKernel(1.0, constant_value_bounds="fixed") * RBF(1.0, length_scale_bounds="fixed"), RBF(), RationalQuadratic(), Matern()]; normalize_y=[False,True]
  grid = dict(kernel=kernel,normalize_y=normalize_y); grid_dict[name] = grid 

### K-NEAREST NEIGHBORS ###
if problem_type == 'classification': # Default = 5,'uniform',2
  name = 'KNN'; algo = KNeighborsClassifier(n_jobs=n_jobs); algo_dict[name]=algo
  n_neighbors = [2,5,8,15]; weights = ['uniform','distance']; p = [1,2]
  grid=dict(n_neighbors=n_neighbors,weights=weights,p=p); grid_dict[name] = grid 

else: # Default = 5,'uniform',2
  name = 'KNN'; algo = KNeighborsRegressor(n_jobs=n_jobs); algo_dict[name]=algo
  n_neighbors = [2,5,8,15]; weights = ['uniform','distance']; p = [1,2]
  grid=dict(n_neighbors=n_neighbors,weights=weights,p=p); grid_dict[name] = grid 

### SUPPORT VECTOR MACHINE ### 
if problem_type == 'classification': # Default = 'rbf',1,
  name = 'SVM'; algo = SVC(max_iter=1000); algo_dict[name]=algo
  kernel = ['rbf', 'poly', 'sigmoid']; C = [0.01, 1, 100]; class_weight = [None,'balanced']
  grid = dict(kernel=kernel,C=C,class_weight=class_weight); grid_dict[name] = grid 

else: # Default = 'rbf',1,0.1
  name = 'SVM'; algo = SVR(max_iter=1000); algo_dict[name]=algo
  kernel = ['rbf', 'poly', 'sigmoid']; C = [0.01, 1, 100]; epsilon = [0.1, 0.01]
  grid = dict(kernel=kernel,C=C,epsilon=epsilon); 
  grid_dict[name] = grid 

### RANDOM FOREST ###
if problem_type == 'classification': # Default = 100,'sqrt',1,None
  name = 'RF'; algo = RandomForestClassifier(random_state=random_state,n_jobs=n_jobs); algo_dict[name]=algo
  n_estimators = [10, 100, 500]; max_features = [0.2, 'sqrt', None]; min_samples_leaf = [1,3,10]; class_weight = [None,'balanced']
  grid = dict(n_estimators=n_estimators,max_features=max_features,min_samples_leaf=min_samples_leaf,class_weight=class_weight); grid_dict[name] = grid 

else: # Default = 100,None,1
  name = 'RF'; algo = RandomForestRegressor(random_state=random_state,n_jobs=n_jobs); algo_dict[name]=algo
  n_estimators = [10, 100, 500]; max_features = [0.2, 'sqrt', None]; min_samples_leaf = [1,3,10]
  grid = dict(n_estimators=n_estimators,max_features=max_features,min_samples_leaf=min_samples_leaf); grid_dict[name] = grid 

### GRADIENT BOOSTING ### 
if problem_type == 'classification': # Default = 100,0.1,1,1
  name = 'GB'; algo = GradientBoostingClassifier(random_state=random_state); algo_dict[name]=algo
  n_estimators = [10, 100, 500]; learning_rate = [0.01, 0.1]; subsample = [0.5, 1.0]; min_samples_leaf = [1,3,10];
  grid = dict(n_estimators=n_estimators, learning_rate=learning_rate, subsample=subsample, min_samples_leaf=min_samples_leaf); grid_dict[name] = grid 
else: # Default = 100,0.1,1,1
  name = 'GB'; algo = GradientBoostingRegressor(random_state=random_state); algo_dict[name]=algo
  n_estimators = [10, 100, 500]; learning_rate = [0.01, 0.1]; subsample = [0.5, 1.0]; min_samples_leaf = [1,3,10];
  grid = dict(n_estimators=n_estimators, learning_rate=learning_rate, subsample=subsample, min_samples_leaf=min_samples_leaf); grid_dict[name] = grid 

### MULTILAYER PERCEPTRON (NEURAL NETWORK) ###  
if problem_type == 'classification': # Default = (100,),0.001,0.0001
  name = 'MLP'; algo = MLPClassifier(random_state=random_state); algo_dict[name]=algo
  hidden_layer_sizes = [(100,), (50,50), (100,100), (50,50,50)]; learning_rate_init=[0.01,0.001,0.0001]; alpha = [0.0001, 0.1]
  grid = dict(hidden_layer_sizes=hidden_layer_sizes, learning_rate_init=learning_rate_init, alpha=alpha); grid_dict[name] = grid 
else: # Default = (100,),0.001,0.0001
  name = 'MLP'; algo = MLPRegressor(random_state=random_state); algo_dict[name]=algo
  hidden_layer_sizes = [(100,), (50,50), (100,100), (50,50,50)]; learning_rate_init=[0.01,0.001,0.0001]; alpha = [0.0001, 0.1]
  grid = dict(hidden_layer_sizes=hidden_layer_sizes, learning_rate_init=learning_rate_init, alpha=alpha); grid_dict[name] = grid 


# In[7]:


# Imports for model evaluation 
if problem_type == 'classification':
  from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, average_precision_score, roc_auc_score
  # List to hold the scoring metrics 
  scoring = ['accuracy', 'precision', 'recall', 'f1', 'balanced_accuracy', 'average_precision']
  if y.unique().shape[0]!=2: 
    scoring = ['accuracy', 'precision_weighted', 'recall_weighted', 'f1_weighted', 'balanced_accuracy']
else: 
  from sklearn.metrics import mean_absolute_error, median_absolute_error, mean_absolute_percentage_error, r2_score, explained_variance_score, max_error
  # List to hold the scoring metrics 
  scoring = ['neg_mean_absolute_error', 'neg_median_absolute_error', 'neg_mean_absolute_percentage_error', 'r2', 'explained_variance', 'max_error']

# Imports for cross validation
if problem_type == 'classification':
  from sklearn.model_selection import StratifiedGroupKFold
else:
  from sklearn.model_selection import GroupKFold

# Imports for grid-searching 
from sklearn.model_selection import GridSearchCV

# Imports for modeling pipeline 
from sklearn.pipeline import Pipeline


# In[8]:


# Function to extract relevant information from the exhaustive grid search of potential hyperparameter configurations
def extract_info_from_grid_search(grid_search_result,scoring): 

  # Store all of the grid search results as a dataframe 
  grid_search_result_df = pd.DataFrame(grid_search_result.cv_results_); 

  # Extract the split-by-split results, pair them with the params, and add a new column to store the unique model configuration ID
  split_results_df = pd.concat([grid_search_result_df.filter(regex='split'), grid_search_result_df.filter(regex='param').iloc[:, ::-1]],axis=1).reset_index(drop=True).reset_index().rename(columns={'index':'model_config_id'})

  # Extract the columns for the model config and hyperparams (take everything BUT the split-by-split train and test fold evaluation outcomes) -> these will be used as id_vars for the melt
  index_for_melt = split_results_df.columns[~split_results_df.columns.str.contains('split')]

  # "Melt" the dataframe to convert it from wide format to long format so you can pair the split and fold data with the corresponding outcomes 
  melt_df = split_results_df.melt(id_vars=index_for_melt)

  # Break apart the 'variable' column (produced during the df.melt process) so you can isolate the specific split, the specific fold, and the specific evaluation metric for a given score
  melt_df[['split', 'fold', 'metric']] = melt_df['variable'].str.split('_', n=2, expand=True)

  # Extract the split value and make sure it is numerical
  melt_df['split'] = melt_df['split'].str.split('split',expand=True).drop(columns=0).astype(int)

  # Uncomment this to drop the observations where the model or split resulted in a training or testing score that is NaN 
  #melt_df = melt_df[~melt_df['value'].isna()]

  list_formatted_splits=[]; i=0
  # Go through each of the scoring metrics 
  for scorer in scoring:

    # Reformat the evaluation results and save all of the parameter info 
    if i==0:
      hold = melt_df[melt_df['metric']==scorer].drop(columns=['variable','metric']).rename(columns={'value':scorer}).reset_index(drop=True)

    # Just reformat the evaluation results (don't need to duplicate the parameter info)
    else: 
      hold = melt_df[melt_df['metric']==scorer]['value'].rename(scorer).reset_index(drop=True)

    # Add the reformatted results matrices to the list 
    list_formatted_splits.append(hold); i+=1

  # Concatenate and then save the reformatted matrices into a single dataframe
  score_df = pd.concat(list_formatted_splits,axis=1)

  # Shuffle around the columns so it goes 'split', 'fold', 'model_config_id', all evaluation metrics, all parameters
  split_results_df = score_df[['split','fold','model_config_id',*scoring] + [c for c in score_df if c not in ['split','fold','model_config_id',*scoring]] ] 

  # Sort the values by the 'split', 'fold', and 'model_config_id'
  split_results_df = split_results_df.sort_values(by=['split','fold','model_config_id'])

  # Obtain the split-by-split results so that you can go through and obtain the aggregated results (mean, std, rank(test-only))
  splits_only = grid_search_result_df.filter(regex='split')

  list_aggregated_test=[]; list_aggregated_train=[]
  # Go through each of the scoring metrics 
  for scorer in scoring:

    # Get the mean, std, and rank for the test scores [[[[[ ALSO ADD MEDIAN ?????? AND MEDIAN RANK ???? ]]]]]
    df1 = pd.DataFrame({'mean_'+scorer:splits_only.filter(regex='test_'+scorer).mean(axis=1),
                        'std_'+scorer:splits_only.filter(regex='test_'+scorer).std(axis=1),
                        'rank_'+scorer:splits_only.filter(regex='test_'+scorer).mean(axis=1).rank(ascending=False,method='min'),
                        })

    # Get the mean and std for the train scores [[[[[ ALSO ADD MEDIAN ?????? AND MEDIAN RANK ???? ]]]]]
    df2 = pd.DataFrame({'mean_'+scorer:splits_only.filter(regex='train_'+scorer).mean(axis=1),
                        'std_'+scorer:splits_only.filter(regex='train_'+scorer).std(axis=1),
                        'rank_'+scorer:splits_only.filter(regex='test_'+scorer).mean(axis=1).rank(ascending=False,method='min')
                        })

    # Store the results in the lists 
    list_aggregated_test.append(df1); list_aggregated_train.append(df2)

  # Concatenate all of the results and add a model_config_id column 
  agg_test = pd.concat(list_aggregated_test,axis=1).reset_index(drop=True).reset_index().rename(columns={'index':'model_config_id'}); agg_train = pd.concat(list_aggregated_train,axis=1).reset_index(drop=True).reset_index().rename(columns={'index':'model_config_id'})

  # Concatenate the results with the fit times and params columns 
  agg_test = pd.concat([agg_test,pd.concat([grid_search_result_df.filter(regex='time'),grid_search_result_df.filter(regex='param').iloc[:, ::-1]],axis=1)],axis=1); agg_train = pd.concat([agg_train,pd.concat([grid_search_result_df.filter(regex='time'),grid_search_result_df.filter(regex='param').iloc[:, ::-1]],axis=1)],axis=1)

  # Concatenate the aggregated results and identify the fold
  aggregated_results_df = pd.concat([agg_test,agg_train],keys=['test','train']).reset_index().drop(columns={'level_1'}).rename(columns={'level_0':'fold'})

  # Uncomment this to drop the observations where the model resulted in a training or testing score that is NaN 
  #aggregated_results_df = aggregated_results_df[~aggregated_results_df['mean_'+scoring[0]].isna()]

  # Drop the _weighted so the multi-class classification problems match the binary classification problems
  split_results_df.columns = [col.replace('_weighted', '') for col in split_results_df.columns]; aggregated_results_df.columns = [col.replace('_weighted', '') for col in aggregated_results_df.columns] 

  return split_results_df, aggregated_results_df, #grid_search_result_df


# In[9]:


# Function to perform a grid search on the set of inner loops of a nested CV protocol (note, the outer test data is ignored here, we only use the outer train data as a starting dataset for the inner loops)
def nested_cv__inner_loops(X,y,groups,cv_inner,cv_outer,estimator,param_grid,scoring,refit):

  # Perform exhaustive inner CV operation (execute a new GridSearchCV (using inner CV function) within each training fold of outer CV)

  # List to store all of results as they're generated by the configuration and evaluation/selection protocol
  inner_split_results =[]; inner_aggregated_results=[]; 

  # Iterate through each set of train and test indices generated by the outer CV protocol 
  for train_ix, test_ix in cv_outer.split(X,y,groups=groups):

    # Use the outer CV-generated indices to extract and reformat the X and y train and test sets so they'll be usable in the subsequent inner CV protocol 
    X_train, X_test = X.iloc[train_ix.tolist(),:].reset_index(drop=True), X.iloc[test_ix.tolist(), :].reset_index(drop=True)
    y_train, y_test = y.iloc[train_ix.tolist()].reset_index(drop=True), y.iloc[test_ix.tolist()].reset_index(drop=True)

    # Set up the grid search protocol 
    inner_grid_setup = GridSearchCV(estimator=estimator, param_grid=param_grid, n_jobs=-1, return_train_score=True, cv=cv_inner, scoring=scoring, refit=refit, error_score=np.nan)

    try: 

      # Execute the grid search protocol on the outer training data (the grid search executes its own inner CV scheme to train and validate on the outer training data). The outer test data is withheld because that should remain unseen for unbiased model evaluation
      inner_grid_search = inner_grid_setup.fit(X_train, y_train, groups=groups.iloc[train_ix]);

      # Extract the results from the completed grid search 
      split_results_df, aggregated_results_df = extract_info_from_grid_search(inner_grid_search,scoring)

      # # Update the dataframes with the number of splits (k) from inner loop
      split_results_df = split_results_df.reset_index().rename(columns={'index':'inner_k'}); split_results_df.loc[:,'inner_k']=cv_inner.n_splits; 
      aggregated_results_df = aggregated_results_df.reset_index().rename(columns={'index':'inner_k'}); aggregated_results_df.loc[:,'inner_k']=cv_inner.n_splits

      # Append the results to the storage lists 
      inner_split_results.append(split_results_df); inner_aggregated_results.append(aggregated_results_df);

    except: 
      print('    inner grid search failed')

  # Store all of the results as a dataframe
  inner_split_results_df = pd.concat(inner_split_results,keys=range(cv_outer.n_splits)).reset_index().drop(columns='level_1').rename(columns={'level_0':'outer_split'})
  inner_aggregated_results_df = pd.concat(inner_aggregated_results,keys=range(cv_outer.n_splits)).reset_index().drop(columns='level_1').rename(columns={'level_0':'outer_split'})

  # Update the dataframes with the number of splits (k) from the outer loop 
  inner_split_results_df = inner_split_results_df.reset_index().rename(columns={'index':'outer_k'}); inner_split_results_df.loc[:,'outer_k']=cv_outer.n_splits
  inner_aggregated_results_df = inner_aggregated_results_df.reset_index().rename(columns={'index':'outer_k'}); inner_aggregated_results_df.loc[:,'outer_k']=cv_outer.n_splits

  return inner_split_results_df, inner_aggregated_results_df


# Function to perform a grid search on the set of outer loops of a standard CV protocol 
def cv__outer_loops(X,y,groups,cv_inner,cv_outer,estimator,param_grid,scoring,refit):

  # Perform exhaustive outer CV operation (execute a single GridSearchCV using the outer CV function)

  # Set up the grid search protocol
  outer_grid_setup = GridSearchCV(estimator=estimator, param_grid=param_grid, n_jobs=-1, return_train_score=True, cv=cv_outer, scoring=scoring, refit=refit, error_score=np.nan)

  #try:
  # Execute the grid search protocol on the full data (the grid search executes its own outer CV scheme to train and validate on the full dataset)
  outer_grid_search = outer_grid_setup.fit(X, y, groups=groups);

  # Extract the results from the completed grid search 
  split_results_df, aggregated_results_df = extract_info_from_grid_search(outer_grid_search,scoring)

  # Update the results dataframe with the number of splits from the outer loop
  split_results_df = split_results_df.reset_index().rename(columns={'index':'outer_k'}); split_results_df.loc[:,'outer_k']=cv_outer.n_splits
  aggregated_results_df = aggregated_results_df.reset_index().rename(columns={'index':'outer_k'}); aggregated_results_df.loc[:,'outer_k']=cv_outer.n_splits

  return split_results_df, aggregated_results_df


# In[10]:


# Function to run the nested_cv__inner_loops for all algorithms surveyed in the study 
def run_all_algos__inner(algo_dict,grid_dict,X,y,groups,cv_inner,cv_outer,scoring,refit=False): 

  # List to store all of the inner grid search results as they're generated for each algorithm
  inner_splits_results=[]; inner_agg_results=[];

  # Iterate through each algorithm key in the pre-defined dictionary 
  for key in algo_dict.keys():

    # Extract the given algorithm
    algo=algo_dict[key]
    print('    INNER -',algo)

    # Extract the set of associated hyperparameters to search 
    grid = grid_dict[key] 

    # Modify the parameter names so they're compatible with the Pipeline construct 
    param_grid = {'algo__'+key: grid[key] for key in grid.keys()} 

    # Define a preprocessing step (passthrough = no change, MinMaxScaler scales all inputs to be between 0 and 1)
    param_grid['preprocess']=['passthrough',MinMaxScaler()] #print(param_grid)

    # Set up the pipeline that will be configured and optimized by the grid search protocol 
    pipeline = Pipeline([
                        ('preprocess', MinMaxScaler()), 
                        ('algo', algo) 
                        ]) #print(pipeline)

    # Run the nested_cv__inner_loops function to perform the inner grid search operations and store all of the associated results 
    inner_splits,inner_agg = nested_cv__inner_loops(X=X,y=y,groups=groups,cv_inner=cv_inner,cv_outer=cv_outer,estimator=pipeline,param_grid=param_grid,scoring=scoring,refit=refit)

    # Append the results to the storage list 
    inner_splits_results.append(inner_splits); inner_agg_results.append(inner_agg)

    # Store all of the results as a dataframe 
    inner_splits_results_df = pd.concat(inner_splits_results,keys=algo_dict).reset_index().drop(columns='level_1').rename(columns={'level_0':'algorithm'})
    inner_agg_results_df = pd.concat(inner_agg_results,keys=algo_dict).reset_index().drop(columns='level_1').rename(columns={'level_0':'algorithm'})

  # Go through the "fold" columns and replace all instances of "test" with "val" since this is the inner validation fold rather than the outer test fold
  inner_splits_results_df.loc[:,'fold'] = inner_splits_results_df.loc[:,'fold'].replace('test','val')
  inner_agg_results_df.loc[:,'fold'] = inner_agg_results_df.loc[:,'fold'].replace('test','val')

  # Change the generic 'split' column names to 'inner_split' so you clarify which split you're referring to
  inner_splits_results_df.rename(columns={'split':'inner_split'},inplace=True); inner_agg_results_df.rename(columns={'split':'inner_split'},inplace=True); 

  return inner_splits_results_df, inner_agg_results_df


# Function to run the cv__outer_loops for all algorithms surveyed in the study 
def run_all_algos__outer(algo_dict,grid_dict,X,y,groups,cv_inner,cv_outer,scoring,refit=False): 

  #  List to store all of the outer grid search results as they're generated for each algorithm
  outer_splits_results=[]; outer_agg_results=[];

  # Iterate through each algorithm key in the pre-defined dictionary 
  for key in algo_dict.keys():

    # Extract the given algorithm 
    algo=algo_dict[key]
    print('    OUTER -',algo)

    # Extract the set of associated hyperparameters to search 
    grid = grid_dict[key] 

    # Modify the parameter names so they're compatible with the Pipeline construct
    param_grid = {'algo__'+key: grid[key] for key in grid.keys()} 

    # Define a preprocessing step (passthrough = no change, MinMaxScaler scales all inputs to be between 0 and 1)
    param_grid['preprocess']=['passthrough',MinMaxScaler()] #print(param_grid)

    # Set up the pipeline that will be configured and optimized by the grid search protocol 
    pipeline = Pipeline([
                        ('preprocess', MinMaxScaler()), 
                        ('algo', algo) 
                        ]) 

    # Run the cv__outer_loops function to perform the outer grid search operations and store all of the associated results 
    outer_splits, outer_agg = cv__outer_loops(X=X,y=y,groups=groups,cv_inner=cv_inner,cv_outer=cv_outer,estimator=pipeline,param_grid=param_grid,scoring=scoring,refit=refit)

    # Append the results to the storage list 
    outer_splits_results.append(outer_splits); outer_agg_results.append(outer_agg)

    # Store all of the results as a dataframe 
    outer_splits_results_df = pd.concat(outer_splits_results,keys=algo_dict).reset_index().drop(columns='level_1').rename(columns={'level_0':'algorithm'})
    outer_agg_results_df = pd.concat(outer_agg_results,keys=algo_dict).reset_index().drop(columns='level_1').rename(columns={'level_0':'algorithm'})

    # Change the generic 'split' column names to 'inner_split' so you clarify which split you're referring to
    outer_splits_results_df.rename(columns={'split':'outer_split'},inplace=True)

  return outer_splits_results_df, outer_agg_results_df


# In[11]:


# Silence the verbose sklearn warnings for infeasible param combinations or non-finite scoring evaluations (all of these are just stored as NaNs)
# from sklearn.exceptions import FitFailedWarning, ConvergenceWarning, UndefinedMetricWarning
# https://github.com/scikit-learn/scikit-learn/issues/12939 # https://stackoverflow.com/questions/53784971/how-to-disable-convergencewarning-using-sklearn/55595680#55595680

import warnings 
warnings.simplefilter("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"

# Define the outer and inner splits to explore 
outer_k_values = [2,3,4,5,10]; inner_k_values = [2,3,4,5,10] 

# Import random module for shuffling
import random

# Define the random seeds random_seeds=[0]
random_seeds = [0,1,2]

# Get the initial time for the whole experimental process
init_time=time.perf_counter()

# Lists to store all of the results from all of the different experimental iterations (with different initial randomizations)
list_in_splits=[]; list_in_agg=[]; list_out_splits=[]; list_out_agg=[]

# Iterate through different randomizations of the dataset and apply the data-splitting protocol and the iterative [algorithm+configuration train/val/test] protocol and store all results
for seed in random_seeds:

  # Initialize the random number generator 
  print('***************')
  random.seed(seed); print('* Iteration',seed,'*')
  print('***************')

  # Break the dataset apart by the unique experimental grouping ID
  grouped_data = [grouped_data for _, grouped_data in data.groupby('Groups')] 

  # Shuffle the dataset but keep all of the grouped data together
  random.shuffle(grouped_data)

  # Turn the shuffled (but grouped) data back into a dataframe
  data_shuffled = pd.concat(grouped_data).reset_index(drop=True).copy()

  # Assign X values 
  X = data_shuffled.iloc[:,:-2]

  # Assign y values 
  y = data_shuffled.iloc[:,-2]

  # Assign groups 
  groups = data_shuffled.iloc[:,-1]

  # Start the timer to measure the duration of the protocol 
  start_time = time.perf_counter()

  # Lists to store all of the results for all of the algorithms for all of the outer and inner splits 
  list_outer_splits_results = []; list_inner_splits_results = []; list_outer_agg_results = []; list_inner_agg_results = []; 

  # Iterate through the set of potential outer splits 
  for outer_k in outer_k_values: 

    # Obtain the time for each new start of the full outer loop protocol 
    start_outer = time.perf_counter()

    # Configure the outer cross validation procedure -> used for evaluating the quality & generalizability of the tuned model on unseen test data 
    if problem_type == 'classification':
      cv_outer = StratifiedGroupKFold(n_splits=outer_k)
    else: 
      cv_outer = GroupKFold(n_splits=outer_k)

    # Iterate through the set of potential inner splits 
    for inner_k in inner_k_values: 

      # Obtain the time for each new start of the inner loops protocol 
      start_inner = time.perf_counter()

      print('======================================================================================')
      print('OUTER K =',outer_k,'---- INNER K =',inner_k)

      # Configure the inner cross validation procedure -> used for hyperparameter tuning and model selection 
      if problem_type == 'classification':
        cv_inner = StratifiedGroupKFold(n_splits=inner_k)
      else: 
        cv_inner = GroupKFold(n_splits=inner_k)

      # Execute the set of grid searches within the inner loop portion of the nested CV (one grid search for each algorithm for each training fold of the outer loop)
      print(' Starting INNER Protocol')
      inner_splits_results, inner_agg_results = run_all_algos__inner(algo_dict,grid_dict,X,y,groups,cv_inner,cv_outer,scoring)
      list_inner_splits_results.append(inner_splits_results); list_inner_agg_results.append(inner_agg_results); 

      # Output the time of the inner-only protocol (inner loops for all algorithms)
      print('    Inner-Only Time:',np.round((time.perf_counter()-start_inner),2))

    print('======================================================================================')
    print('OUTER K =',outer_k,)

    # Execute the single grid search within the outer loop portion of the nested CV (one grid search for each algorithm)
    print(' Starting OUTER Protocol')
    outer_splits_results, outer_agg_results = run_all_algos__outer(algo_dict,grid_dict,X,y,groups,cv_inner,cv_outer,scoring)
    list_outer_splits_results.append(outer_splits_results); list_outer_agg_results.append(outer_agg_results); 

    # Output the combined time of the outer and inner protocols (outer loop for all algorithms and corresponding inner loops for all algorithms)
    print('    Inner + Outer Time:',np.round((time.perf_counter()-start_outer),2))

  # Output the total time that was elapsed for the entire protocol (all algos, all inner and outer, all splits)
  print('======================================================================================')
  print('======================================================================================')
  print('TOTAL TIME FOR ROUTINE:',np.round((time.perf_counter()-start_time),2),'\n\n')

  # Compile all of the results from this iteration into dataframes and add them to the appropriate list
  inner_splits = pd.concat(list_inner_splits_results).reset_index(drop=True); inner_agg = pd.concat(list_inner_agg_results).reset_index(drop=True); list_in_splits.append(inner_splits); list_in_agg.append(inner_agg); 
  outer_splits = pd.concat(list_outer_splits_results).reset_index(drop=True); outer_agg = pd.concat(list_outer_agg_results).reset_index(drop=True); list_out_splits.append(outer_splits); list_out_agg.append(outer_agg)

# Compile the modeling results from all of the different iterations and save to their own dataframes (inner loops, outer loops... split-by-split data and aggregated data)
in_splits = pd.concat(list_in_splits,keys=random_seeds).reset_index().drop(columns={'level_1'}).rename(columns={'level_0':'iteration'}); in_agg = pd.concat(list_in_agg,keys=random_seeds).reset_index().drop(columns={'level_1'}).rename(columns={'level_0':'iteration'})
out_splits = pd.concat(list_out_splits,keys=random_seeds).reset_index().drop(columns={'level_1'}).rename(columns={'level_0':'iteration'}); out_agg = pd.concat(list_out_agg,keys=random_seeds).reset_index().drop(columns={'level_1'}).rename(columns={'level_0':'iteration'})

# Organize the columns in the order you'd like 
in_splits = in_splits[['iteration','outer_k','outer_split','inner_k','inner_split','fold','algorithm','model_config_id',] + 
                      [c for c in in_splits if c not in ['iteration','outer_k','outer_split','inner_k','inner_split','fold','algorithm','model_config_id',]] ]

in_agg = in_agg[['iteration','outer_k','outer_split','inner_k','fold','algorithm','model_config_id',] + 
                [c for c in in_agg if c not in ['iteration','outer_k','outer_split','inner_k','fold','algorithm','model_config_id',]] ]

out_splits = out_splits[['iteration','outer_k','outer_split','fold','algorithm','model_config_id',] + 
                        [c for c in out_splits if c not in ['iteration','outer_k','outer_split','fold','algorithm','model_config_id',]] ]

out_agg = out_agg[['iteration','outer_k','fold','algorithm','model_config_id',] + 
                  [c for c in out_agg if c not in ['iteration','outer_k','fold','algorithm','model_config_id',]] ]

print('\nFINAL TIME FOR WHOLE EXPERIMENT:',np.round((time.perf_counter()-init_time),2))


# In[12]:


# # Note - the condensed 'params' column can be easily split out into a pandas dataframe structure using the below code -> so to reduce file size just delete all of the individual param-specific columns 
# pd.DataFrame(in_splits['params'].values.tolist())

# Function to reduce the memory of the dataframes produced by the experiment (Python automatically uses memory-intensive datatypes)
def reduce_memory(dataframe,):

    df = dataframe.copy()

    # Drop all of the individual / param-specific columns (only keep the condensed 'params' columns which contains a dictionary of all of the unique hyperparameters and values for each algorithm configuration)
    df = df.drop(columns=df.filter(regex='param_').columns)

    # Convert the algorithm and fold columns into category dtypes 
    df['algorithm'] = df['algorithm'].astype('category'); df['fold'] = df['fold'].astype('category')

    # Go through the remaining columns 
    for column in df.columns: 

        # Convert the int columns to uint8 to reduce memory 
        if df[column].dtype == 'int64':

            if abs(df[column]).max() < 255: 
                df[column] = df[column].astype('uint8')
            else: 
                df[column] = df[column].astype('uint16')

        # Convert the float columns to float32 to reduce memory 
        if df[column].dtype == 'float64':

            df[column] = df[column].astype('float32')

    return df


# In[13]:


# Apply the function to reduce the memory of the dataframe

in_splits_copy = reduce_memory(in_splits)

in_agg_copy = reduce_memory(in_agg)

out_splits_copy = reduce_memory(out_splits)

out_agg_copy = reduce_memory(out_agg)


# In[14]:


# Export the dataframes as csv files to the directory (use gzip for significant file compression)

in_splits_copy.to_csv('results/'+file_name+'__in_splits'+'.csv',index=False,compression='gzip')

in_agg_copy.to_csv('results/'+file_name+'__in_agg'+'.csv',index=False,compression='gzip')

out_splits_copy.to_csv('results/'+file_name+'__out_splits'+'.csv',index=False,compression='gzip')

out_agg_copy.to_csv('results/'+file_name+'__out_agg'+'.csv',index=False,compression='gzip')