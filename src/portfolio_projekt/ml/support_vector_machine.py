import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from imblearn.pipeline import Pipeline 
from sklearn.model_selection import cross_val_score
import pickle
#from pathlib import Path
from portfolio_projekt.paths import MODELS_DIR

import optuna 
from optuna.samplers import TPESampler
import time

def support_vector_machine(features, target, threshold, cv, scoring, thresholds, model_names, files, cv_scores ):
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
    svm_pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("model", SVC(
            kernel="rbf",
            random_state=42
        ))
    ])

    

    cv_svm = cross_val_score(estimator= svm_pipe,
                            X=features,
                            y=target,
                            cv=cv,
                            scoring=scoring[threshold],
                            n_jobs=-1)

    print(f"CV Score: {cv_svm.mean():.3f}")

    
    print("treshold: ", threshold)
    print(f"CV mean Score: {cv_svm.mean():.3f}")
    
    thresholds.append(threshold)
    model_names.append('SVM simple')
    cv_scores.append(cv_svm.mean())
    
    # fit pipeline 
    svm_pipe.fit(features, target)
        
    #save fitted pipeline
    model_dir = MODELS_DIR
    model_dir.mkdir(parents=True, exist_ok=True)
    
    model_path = model_dir / f"svm_{threshold}.p"
    
    pickle.dump(svm_pipe, open(model_path, 'wb'))
    files.append(model_path)
        
    return(thresholds, model_names, files, cv_scores)



def support_vector_machine_opt(features, target, threshold, cv, scoring, thresholds, model_names, files, cv_scores ):
    """instantiate optimized support vector machine with engineered features for a given threshold of fpr
    
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
    svm_pipe_opt = Pipeline([
        ("scaler", StandardScaler()),
        ("pca", PCA()),
        ("model", SVC(
            kernel="rbf",
            random_state=42
        ))
    ])

    # tune hyperparameters with Bayesian Optimization

    def objective_svm(trial):
    
        # search space
        C = trial.suggest_float("C", 1e-3, 1000, log=True)        # steuert Trade-Off zwischen einfacher Entscheidungsgrenze und möglichst wenig Trainingsfehlern
        gamma = trial.suggest_float("gamma", 1e-4, 1e1, log=True)   #  bestimmt wie lokal der Einfluss eines einzelnen Trainingspunkts ist
        n_components = trial.suggest_int("n_components", 1, 17)   # PCA  

        
        params_svm = {'model__C': C,
                'model__gamma': gamma,
                'pca__n_components': n_components,
                'model__max_iter' : 10000
                }
        
        svm_pipe_opt.set_params(**params_svm)

        
        # initiating cv
        score_svm =  cross_val_score(estimator=svm_pipe_opt, 
                                X=features, 
                                y=target, 
                                scoring=scoring[threshold],
                                cv=cv,
                                n_jobs=-1).mean()
        
        print(f"CV fertig: {score_svm:.4f}")
            
        return score_svm

    # create a study (aim to maximize score) und setting a seed (random_state) for reproduceability
    study_svm = optuna.create_study(sampler=TPESampler(seed = 42), direction='maximize')

    # perform hyperparamter tuning (while timing the process)
    time_start = time.time()
    # starting optimization process with our defined function and 50 iterations
    study_svm.optimize(objective_svm, n_trials=50)   # more trial took to long for time limit
    time_bayesian = time.time() - time_start

    # store result in a data frame 
    values_bayesian_svm = [50, study_svm.best_trial.number, study_svm.best_trial.value, time_bayesian]
    results_bayesian_svm = pd.DataFrame([values_bayesian_svm], columns = ['Number of iterations', 
                                                                        'Iteration Number of Optimal Hyperparamters', 
                                                                        'Score', 
                                                                        'Time Elapsed (s)'])
            
    print("threshold: ", threshold)       
    # print results
    print("results: ", results_bayesian_svm)

    # print best parameter
    print("best params: ", study_svm.best_params)

    best_params_svm = study_svm.best_params
    svm_pipe_bo = svm_pipe_opt.set_params(
            model__C=best_params_svm["C"],
            model__gamma=best_params_svm["gamma"],
            pca__n_components=best_params_svm["n_components"]
    )
    cv_svm_bo = cross_val_score(estimator= svm_pipe_bo,
                                X=features,
                                y=target,
                                cv=cv,
                                scoring=scoring[threshold],
                                n_jobs=-1)
    #print(cv_log_reg_bo.mean())
    print(f"CV Score: {cv_svm_bo.mean():.3f}")

    thresholds.append(threshold)
    model_names.append('Support Vector Machine optimized')
    cv_scores.append(cv_svm_bo.mean())

    svm_pipe_bo.fit(features, target)
    
     #save fitted pipeline
    model_dir = MODELS_DIR
    model_dir.mkdir(parents=True, exist_ok=True)        
    model_path = model_dir / f"svm__opt_{threshold}.p"
    pickle.dump(svm_pipe_bo, open(model_path, 'wb'))
    files.append(model_path)
    
    return(thresholds, model_names, files, cv_scores)