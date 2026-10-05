import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPClassifier
from imblearn.pipeline import Pipeline 
from sklearn.model_selection import cross_val_score
import pickle
#from pathlib import Path
from portfolio_projekt.paths import MODELS_DIR

import optuna 
from optuna.samplers import TPESampler
import time

def neural_network(features, target, threshold, cv, scoring, thresholds, model_names, files, cv_scores ):
    """instantiate unoptimized neural network with engineered features for a given threshold of fpr
    
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
    model_nn = MLPClassifier(
        hidden_layer_sizes=(32, 16),   # zwei hidden layer, 1. mit  32 dann mit 16 
        activation="relu",             # acivation function
        solver="adam",
        alpha=1e-4,                    # L2 Regularisierung
        learning_rate_init=1e-3,       # anfängliche Schrittweite
        max_iter=500,
        early_stopping=True,
        validation_fraction=0.15,
        n_iter_no_change=20,
        random_state=42
    )



    nn_pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("model", model_nn)
    ])

    cv_nn = cross_val_score(estimator= nn_pipe,
                            X=features,
                            y=target,
                            cv=cv,
                            scoring=scoring[threshold],
                           n_jobs=-1)

    print("treshold: ", threshold)
    print(f"CV mean Score: {cv_nn.mean():.3f}")
    
    thresholds.append(threshold)
    model_names.append('NN simple')
    cv_scores.append(cv_nn.mean())
    
    # fit pipeline 
    nn_pipe.fit(features, target)
        
    #save fitted pipeline
    from portfolio_projekt.paths import MODELS_DIR
    model_dir.mkdir(parents=True, exist_ok=True)
    
    model_path = model_dir / f"nn_{threshold}.p"
    
    pickle.dump(nn_pipe, open(model_path, 'wb'))
    files.append(model_path)
        
    return(thresholds, model_names, files, cv_scores)



def neural_network_opt(features, target, threshold, cv, scoring, thresholds, model_names, files, cv_scores ):
    """instantiate optimized neural network with engineered features for a given threshold of fpr
    
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
    model_nn = MLPClassifier(
            hidden_layer_sizes=(32, 16),   # zwei hidden layer, 1. mit  32 dann mit 16 
            activation="relu",             # acivation function
            solver="adam",
            alpha=1e-4,                    # L2 Regularisierung
            learning_rate_init=1e-3,       # anfängliche Schrittweite
            max_iter=500,
            early_stopping=True,
            validation_fraction=0.15,
            n_iter_no_change=20,
            random_state=42
    )
    
    
    
    nn_pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("model", model_nn)
    ])    
    
    architectures = {
        "16": (16,),
        "32": (32,),
        "64": (64,),
        "32_16": (32, 16),
        "64_32": (64, 32),
        "64_32_16": (64, 32, 16)
    }
    
    
    def objective_nn(trial):
        """return maximized score"""
    
    
        architecture_name = trial.suggest_categorical(
            "architecture",
            list(architectures.keys())
        )
        
       
        # search space
        hidden_layer_sizes = architectures[architecture_name]
        alpha = trial.suggest_float("alpha", 1e-6, 1e-2, log=True)  
        learning_rate_init = trial.suggest_float("learning_rate_init", 1e-4, 1e-2, log=True)   
        activation = trial.suggest_categorical("activation", ["relu", "tanh"])  
    
        
        params_nn = {"model__hidden_layer_sizes": hidden_layer_sizes,
                    'model__alpha': alpha,
                  'model__learning_rate_init': learning_rate_init,
                 'model__activation': activation  
                }
        nn_pipe.set_params(**params_nn)
    
        
        # initiating cv
        score_nn =  cross_val_score(estimator=nn_pipe, 
                                 X=features, 
                                 y=target, 
                                 scoring=scoring[threshold],
                                 cv=cv,
                                 n_jobs=-1).mean()
        
        return score_nn

    # create a study (aim to maximize score) und setting a seed (random_state) for reproduceability
    study_nn = optuna.create_study(sampler=TPESampler(seed = 42), direction='maximize')

    # perform hyperparamter tuning (while timing the process)
    time_start = time.time()
    # starting optimization process with our defined function and 50 iterations
    study_nn.optimize(objective_nn, n_trials=50)   # more trial took to long for time limit
    time_bayesian = time.time() - time_start

    # store result in a data frame 
    values_bayesian_nn = [50, study_nn.best_trial.number, study_nn.best_trial.value, time_bayesian]
    results_bayesian_nn = pd.DataFrame([values_bayesian_nn], columns = ['Number of iterations', 
                                                                        'Iteration Number of Optimal Hyperparamters', 
                                                                        'Score', 
                                                                        'Time Elapsed (s)'])
            
    print("threshold: ", threshold)       
    # print results
    print("results: ", results_bayesian_nn)

    # print best parameter
    print("best params: ", study_nn.best_params)

    best_params_nn = study_nn.best_params

    best_hidden_layer_sizes = architectures[
        best_params_nn["architecture"]
    ]   

    nn_pipe_bo = nn_pipe.set_params(
            model__hidden_layer_sizes=best_hidden_layer_sizes,
            model__alpha=best_params_nn["alpha"],
            model__learning_rate_init=best_params_nn["learning_rate_init"],
            model__activation=best_params_nn["activation"]
    )
    cv_nn_bo = cross_val_score(estimator= nn_pipe_bo,
                            X=features,
                            y=target,
                            cv=cv,
                            scoring=scoring[threshold],
                            n_jobs=-1)
    #print(cv_log_reg_bo.mean())
    print(f"CV Score: {cv_nn_bo.mean():.3f}")

    thresholds.append(threshold)
    model_names.append('neural network optimized')
    cv_scores.append(cv_nn_bo.mean())

    nn_pipe_bo.fit(features, target)
    
     #save fitted pipeline
    from portfolio_projekt.paths import MODELS_DIR
    model_dir.mkdir(parents=True, exist_ok=True)        
    model_path = model_dir / f"nn__opt_{threshold}.p"
    pickle.dump(nn_pipe_bo, open(model_path, 'wb'))
    files.append(model_path)
    
    return(thresholds, model_names, files, cv_scores)