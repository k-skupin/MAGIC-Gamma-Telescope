import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from imblearn.pipeline import Pipeline 
from sklearn.model_selection import cross_val_score
import pickle
#from pathlib import Path
from portfolio_projekt.paths import MODELS_DIR

import optuna 
from optuna.samplers import TPESampler
import time

def random_forest(features, target, threshold, cv, scoring, thresholds, model_names, files, cv_scores ):
    """instantiate unoptimized random forest with engineered features for a given threshold of fpr
    
        Args:
            features (dataframe): Dataframe with features of dataset
            target (series): serien with target values
            threshold (str): to identify which scorer is used
            cv (StratifiedKFold): params for cross-validation
            scoring (dict): self defined scorer dictionary
            thresholds: list which tracks which threshold is used for which model
             model_names: list of model_names
            files: list of file_names
            cv_scores: list of cv scores for different models
            
            
        Output:
            thresholds: list which tracks which threshold is used for which model
            model_names: list of model_names
            files: list of file_names
            cv_scores: list of cv scores for different models
        """
    # build unoptimized model

    # instantiate model
    model_rf = RandomForestClassifier(
    n_estimators=100,
    max_depth=12,
    random_state=42,
    class_weight='balanced'
    )


    # Instantiating final pipeline, no need for scaling or PCA in RandomForest
    rf_pipe = model_rf
    cv_rf = cross_val_score(estimator= rf_pipe,
                            X=features,
                            y=target,
                            cv=cv,
                            scoring=scoring[threshold],
                           n_jobs=-1)
    
    print("treshold: ", threshold)
    print(f"CV mean Score: {cv_rf.mean():.3f}")
    
    thresholds.append(threshold)
    model_names.append('RandomForest simple')
    cv_scores.append(cv_rf.mean())
    
    # fit pipeline 
    rf_pipe.fit(features, target)
        
    #save fitted pipeline
    model_dir = MODELS_DIR
    model_dir.mkdir(parents=True, exist_ok=True)
    
    model_path = model_dir / f"rf_{threshold}.p"
    
    pickle.dump(rf_pipe, open(model_path, 'wb'))
    files.append(model_path)
        
    return(thresholds, model_names, files, cv_scores)


def random_forest_opt(features, target, threshold, cv, scoring, thresholds, model_names, files, cv_scores ):
    """instantiate optimized random forest with engineered features for a given threshold of fpr
    
        Args:
            features (dataframe): Dataframe with features of dataset
            target (series): serien with target values
            threshold (str): to identify which scorer is used
            cv (StratifiedKFold): params for cross-validation
            scoring (dict): self defined scorer dictionary
            thresholds: list which tracks which threshold is used for which model
             model_names: list of model_names
            files: list of file_names
            cv_scores: list of cv scores for different models
            
            
        Output:
            thresholds: list which tracks which threshold is used for which model
            model_names: list of model_names
            files: list of file_names
            cv_scores: list of cv scores for different models
        """
        
    # instantiate model
    model_rf = RandomForestClassifier(
    n_estimators=100,
    max_depth=12,
    random_state=42,
    class_weight='balanced'
    )
    
    
    # Instantiating final pipeline, no need for scaling or PCA in RandomForest
    rf_pipe = model_rf

    # tune hyperparameters with Bayesian Optimization

    def objective_rf(trial):  
        # search space
        n_estimators = trial.suggest_int("n_estimators", 50, 250, step=50)
        max_depth = trial.suggest_int("max_depth", 5, 15)  
        min_samples_split = trial.suggest_int("min_samples_split", 2, 10, step=2)   
        min_samples_leaf = trial.suggest_int("min_samples_leaf", 3,10, step=1)   
        max_features = trial.suggest_categorical("max_features", ["sqrt", "log2"])
    
        
        params_rf = {"n_estimators": n_estimators,
                    'max_depth': max_depth,
                    'min_samples_split': min_samples_split,
                    'min_samples_leaf': min_samples_leaf,
                    'max_features': max_features
                }
        rf_pipe.set_params(**params_rf)
    
        
        # initiating cv
        score_rf =  cross_val_score(estimator=rf_pipe, 
                                    X=features, 
                                    y=target, 
                                    scoring=scoring[threshold],
                                    cv=cv,
                                    n_jobs=-1).mean()
        
        return score_rf

    # create a study (aim to maximize score) und setting a seed (random_state) for reproduceability
    study_rf = optuna.create_study(sampler=TPESampler(seed = 42), direction='maximize')

    # perform hyperparamter tuning (while timing the process)
    time_start = time.time()
    # starting optimization process with our defined function and 50 iterations
    study_rf.optimize(objective_rf, n_trials=50)   # more trial took to long for time limit
    time_bayesian = time.time() - time_start

    # store result in a data frame 
    values_bayesian_rf = [50, study_rf.best_trial.number, study_rf.best_trial.value, time_bayesian]
    results_bayesian_rf = pd.DataFrame([values_bayesian_rf], columns = ['Number of iterations', 
                                                                            'Iteration Number of Optimal Hyperparamters', 
                                                                            'Score', 
                                                                            'Time Elapsed (s)'])
            
    print("threshold: ", threshold)       
    # print results
    print("results: ", results_bayesian_rf)

    # print best parameter
    print("best params: ", study_rf.best_params)

    best_params_rf = study_rf.best_params
    rf_pipe_bo = rf_pipe.set_params(
            n_estimators=best_params_rf["n_estimators"],
            max_depth=best_params_rf["max_depth"],
            min_samples_split=best_params_rf["min_samples_split"],
            min_samples_leaf=best_params_rf["min_samples_leaf"],
            max_features=best_params_rf["max_features"]
    )

    cv_rf_bo = cross_val_score(estimator= rf_pipe_bo,
                            X=features,
                            y=target,
                            cv=cv,
                            scoring=scoring[threshold],
                            n_jobs=-1)
    #print(cv_log_reg_bo.mean())
    print(f"CV Score: {cv_rf_bo.mean():.3f}")

    thresholds.append(threshold)
    model_names.append('RandomForest optimized')
    cv_scores.append(cv_rf_bo.mean())

    rf_pipe_bo.fit(features, target)
    
     #save fitted pipeline
    model_dir = MODELS_DIR
    model_dir.mkdir(parents=True, exist_ok=True)        
    model_path = model_dir / f"rf__opt_{threshold}.p"
    pickle.dump(rf_pipe_bo, open(model_path, 'wb'))
    files.append(model_path)
    
    return(thresholds, model_names, files, cv_scores)


def random_forest_opt_2(features, target, threshold, cv, scoring, thresholds, model_names, files, cv_scores):

    # instantiate model
    model_rf = RandomForestClassifier(
    n_estimators=100,
    max_depth=12,
    random_state=42,
    class_weight='balanced'
    )
    
    
    # Instantiating final pipeline, no need for scaling or PCA in RandomForest
    rf_pipe = model_rf
    
    # optimize model with bayesian optimization
    def objective_rf_2(trial):
        """return maximized f1-score"""
    
        # search space
        n_estimators = trial.suggest_int("n_estimators", 300, 1500, step=100)
        max_depth = trial.suggest_int("max_depth", 4, 30)  
        min_samples_split = trial.suggest_int("min_samples_split", 2, 30)   
        min_samples_leaf = trial.suggest_int("min_samples_leaf", 1,20, step=1)   
        max_features = trial.suggest_float("max_features", 0.3, 1.0)
        class_weight = trial.suggest_categorical("class_weight", [None, "balanced", "balanced_subsample"])
        criterion = trial.suggest_categorical("criterion", ["gini", "entropy", "log_loss"])
        max_samples = trial.suggest_float("max_samples", 0.6, 1.0)
        

        
        params_rf_2 = {"n_estimators": n_estimators,
                    'max_depth': max_depth,
                'min_samples_split': min_samples_split,
                'min_samples_leaf': min_samples_leaf,
                'max_features': max_features,
                'class_weight': class_weight,
                'criterion': criterion,
                'max_samples': max_samples
                }
        rf_pipe.set_params(**params_rf_2)

        
        # initiating cv
        score_rf_2 =  cross_val_score(estimator=rf_pipe, 
                                X=features, 
                                y=target, 
                                scoring=scoring[threshold],
                                cv=cv,
                                n_jobs=-1).mean()
        
        return score_rf_2

    # create a study (aim to maximize score) und setting a seed (random_state) for reproduceability
    study_rf_2 = optuna.create_study(sampler=TPESampler(seed = 42), direction='maximize')

    # perform hyperparamter tuning (while timing the process)
    time_start = time.time()
    # starting optimization process with our defined function and 50 iterations
    study_rf_2.optimize(objective_rf_2, n_trials=50)
    time_bayesian = time.time() - time_start

    # store result in a data frame 
    values_bayesian_rf_2 = [50, study_rf_2.best_trial.number, study_rf_2.best_trial.value, time_bayesian]
    results_bayesian_rf_2 = pd.DataFrame([values_bayesian_rf_2], columns = ['Number of iterations', 
                                                                            'Iteration Number of Optimal Hyperparamters', 
                                                                            'Score', 
                                                                            'Time Elapsed (s)'])
    
    print("threshold: ", threshold)       
    # print results
    print("results: ", results_bayesian_rf_2)
    
    best_params_rf_2 = study_rf_2.best_params
    rf_pipe_bo_2 = rf_pipe.set_params(
            n_estimators=best_params_rf_2["n_estimators"],
            max_depth=best_params_rf_2["max_depth"],
            min_samples_split=best_params_rf_2["min_samples_split"],
            min_samples_leaf=best_params_rf_2["min_samples_leaf"],
            max_features=best_params_rf_2["max_features"],
            class_weight=best_params_rf_2["class_weight"],
            criterion=best_params_rf_2["criterion"],
            max_samples=best_params_rf_2["max_samples"]
    )
    
    cv_rf_bo_2 = cross_val_score(estimator= rf_pipe_bo_2,
                            X=features,
                            y=target,
                            cv=cv,
                            scoring=scoring[threshold],
                            n_jobs=-1)
    #print(cv_log_reg_bo.mean())
    print(f"CV Score: {cv_rf_bo_2.mean():.3f}")
    
    thresholds.append(threshold)
    model_names.append('RandomForest opt 2')
    cv_scores.append(cv_rf_bo_2.mean())

    rf_pipe_bo_2.fit(features, target)
    
        #save fitted pipeline
    model_dir = MODELS_DIR
    model_dir.mkdir(parents=True, exist_ok=True)        
    model_path = model_dir / f"rf__opt_2_{threshold}.p"
    pickle.dump(rf_pipe_bo_2, open(model_path, 'wb'))
    files.append(model_path)
    
    return(thresholds, model_names, files, cv_scores)