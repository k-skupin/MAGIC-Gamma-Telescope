from sklearn.linear_model import LogisticRegression
from imblearn.pipeline import Pipeline 
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
import pickle
from pathlib import Path
from portfolio_projekt.paths import MODELS_DIR

def baseline(features, target, threshold, cv, scoring, thresholds, model_names, files, cv_scores):
    """instantiate simple logistic regression as baseline model for a given threshold of fpr

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
    # select baseline features
    baseline_features = [
        "fLength",
        "fWidth",
        "fSize",
        "fConc",
        "fConc1",
        "fAsym",
        "fM3Long",
        "fM3Trans",
        "fAlpha",
        "fDist"
    ]

    # instantiate model
    log_reg = LogisticRegression(
        max_iter=10000,
        random_state=42
    )

    # build pipeline
    log_reg_bl = Pipeline(steps = [('scaler', StandardScaler()), 
                                    ('model', log_reg)])

    # validate baseline model
    cv_log_reg_bl = cross_val_score(estimator= log_reg_bl,
                                X=features[baseline_features],
                                y=target,
                                cv=cv,
                                scoring=scoring[threshold],
                                n_jobs=-1)


    print("treshold: ", threshold)
    print(f"CV mean Score: {cv_log_reg_bl.mean():.3f}")

    thresholds.append(threshold)
    model_names.append('logistic regression baseline')
    cv_scores.append(cv_log_reg_bl.mean())

    # fit pipeline 
    log_reg_bl.fit(features[baseline_features], target)
    
    #save fitted pipeline
    model_dir = MODELS_DIR
    model_dir.mkdir(parents=True, exist_ok=True)

    model_path = model_dir / f"log_reg_bl_{threshold}.p"

    pickle.dump(log_reg_bl, open(model_path, 'wb'))
    files.append(model_path)
    
    return(thresholds, model_names, files, cv_scores, baseline_features)