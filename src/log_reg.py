import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from imblearn.pipeline import Pipeline 
from sklearn.model_selection import cross_val_score
import pickle
from pathlib import Path

import optuna 
from optuna.samplers import TPESampler
import time

def logistic_regression(features, target, threshold, cv, scoring, thresholds, model_names, files, cv_scores ):
    """instantiate unoptimized logistic regression with engineered features for a given threshold of fpr
    
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

    # instantiate scaler
    scaler = StandardScaler()

    # instantiate model
    log_reg = LogisticRegression(
        solver="lbfgs",  
        max_iter=10000,
        random_state=42,
        class_weight='balanced', 
        n_jobs=-1
    )


    # Instantiating final pipeline with preprocessor and model (logreg)
    log_reg_pipe = Pipeline(steps = [('scaler', StandardScaler()), 
                                    ('model', log_reg)])

    cv_log_reg = cross_val_score(estimator= log_reg_pipe,
                                X=features,
                                y=target,
                                cv=cv,
                                scoring=scoring[threshold],
                                n_jobs=-1)

    
    print("treshold: ", threshold)
    print(f"CV mean Score: {cv_log_reg.mean():.3f}")
    
    thresholds.append(threshold)
    model_names.append('logistic regression simple')
    cv_scores.append(cv_log_reg.mean())
    
    # fit pipeline 
    log_reg_pipe.fit(features, target)
        
    #save fitted pipeline
    model_dir = Path("../models")
    model_dir.mkdir(parents=True, exist_ok=True)
    
    model_path = model_dir / f"log_reg_{threshold}.p"
    
    pickle.dump(log_reg_pipe, open(model_path, 'wb'))
    files.append(model_path)
        
    return(thresholds, model_names, files, cv_scores)


def logistic_regression_pca(features, target, threshold, cv, scoring, thresholds, model_names, files, cv_scores ):
    """instantiate unoptimized logistic regression with engineered features for a given threshold of fpr
    
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

    # instantiate scaler
    scaler = StandardScaler()

    # instantiate model
    log_reg = LogisticRegression(
        solver="lbfgs",  
        max_iter=10000,
        random_state=42,
        class_weight='balanced', 
        n_jobs=-1
    )


    # Instantiating final pipeline with scaler, PCA and model (logreg)
    log_reg_pipe_PCA = Pipeline(steps = [('scaler', StandardScaler()), 
                                 ('PCA', PCA(n_components=0.95, random_state=42)),
                                ('model', log_reg)])


    cv_log_reg_PCA = cross_val_score(estimator= log_reg_pipe_PCA,
                                X=features,
                                y=target,
                                cv=cv,
                                scoring=scoring[threshold],
                                n_jobs=-1)

    
    print("treshold: ", threshold)
    print(f"CV mean Score: {cv_log_reg_PCA.mean():.3f}")
    
    thresholds.append(threshold)
    model_names.append('logistische Regression PCA')
    cv_scores.append(cv_log_reg_PCA.mean())
    
    # fit pipeline 
    log_reg_pipe_PCA.fit(features, target)
        
    #save fitted pipeline
    model_dir = Path("../models")
    model_dir.mkdir(parents=True, exist_ok=True)
    
    model_path = model_dir / f"log_reg_pca_{threshold}.p"
    
    pickle.dump(log_reg_pipe_PCA, open(model_path, 'wb'))
    files.append(model_path)
        
    return(thresholds, model_names, files, cv_scores)


def logistic_regression_opt(features, target, threshold, cv, scoring, thresholds, model_names, files, cv_scores ):
    """instantiate optimized logistic regression with engineered features for a given threshold of fpr
    
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
        
    # instantiate model for optimization
    log_reg = LogisticRegression(
        solver="liblinear",  
        max_iter=10000,
        random_state=42,
        class_weight='balanced', 
        n_jobs=-1
    )

    # Instantiating pipeline for optimization
    log_reg_pipe_opt = Pipeline(steps = [('scaler', StandardScaler()), 
                                    ('pca', PCA()),
                                    ('model', log_reg)])

    # tune hyperparameters with Bayesian Optimization

    def objective_log_reg(trial):
        
        # search space
        C = trial.suggest_float("C", 1e-4, 1000, log=True)        # inverse of regularization strength (small --> simple model)
        l1_ratio = trial.suggest_categorical("l1_ratio", [0.0, 1.0])   # 
        n_components = trial.suggest_int("n_components", 1, 17)   # PCA

        
        params_log_reg = {'model__C': C,
                'model__l1_ratio': l1_ratio,
                'pca__n_components': n_components,
                'model__max_iter' : 10000
                }
        
        log_reg_pipe_opt.set_params(**params_log_reg)

        
        # initiating cv
        score_log_reg =  cross_val_score(estimator=log_reg_pipe_opt, 
                                X=features, 
                                y=target, 
                                scoring=scoring[threshold],
                                cv=cv,
                                n_jobs=-1).mean()
        
        return score_log_reg

    # create a study (aim to maximize score) und setting a seed (random_state) for reproduceability
    study_log_reg = optuna.create_study(sampler=TPESampler(seed = 42), direction='maximize')

    # perform hyperparamter tuning (while timing the process)
    time_start = time.time()
    # starting optimization process with our defined function and 50 iterations
    study_log_reg.optimize(objective_log_reg, n_trials=50)   # more trial took to long for time limit
    time_bayesian = time.time() - time_start

    # store result in a data frame 
    values_bayesian_log_reg = [50, study_log_reg.best_trial.number, study_log_reg.best_trial.value, time_bayesian]
    results_bayesian_log_reg = pd.DataFrame([values_bayesian_log_reg], columns = ['Number of iterations', 
                                                                            'Iteration Number of Optimal Hyperparamters', 
                                                                            'Score', 
                                                                            'Time Elapsed (s)'])
            
    print("threshold: ", threshold)       
    # print results
    print("results: ", results_bayesian_log_reg)

    # print best parameter
    print("best params: ", study_log_reg.best_params)

    best_params_log_reg = study_log_reg.best_params
    log_reg_pipe_bo = log_reg_pipe_opt.set_params(
                model__C=best_params_log_reg["C"],
                model__l1_ratio=best_params_log_reg["l1_ratio"],
                pca__n_components=best_params_log_reg["n_components"]
    )

    cv_log_reg_bo = cross_val_score(estimator= log_reg_pipe_bo,
                                X=features,
                                y=target,
                                cv=cv,
                                scoring=scoring[threshold],
                                n_jobs=-1)

    print(f"CV Score: {cv_log_reg_bo.mean():.3f}")

    thresholds.append(threshold)
    model_names.append('Logistic regression optimized')
    cv_scores.append(cv_log_reg_bo.mean())

    log_reg_pipe_bo.fit(features, target)
    
     #save fitted pipeline
    model_dir = Path("../models")
    model_dir.mkdir(parents=True, exist_ok=True)        
    model_path = model_dir / f"log_reg__opt_{threshold}.p"
    pickle.dump(log_reg_pipe_bo, open(model_path, 'wb'))
    files.append(model_path)
    
    return(thresholds, model_names, files, cv_scores)