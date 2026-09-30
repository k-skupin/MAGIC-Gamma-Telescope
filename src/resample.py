from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline 
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

def resampling(features, target, threshold, cv, scoring):
    
    """
    Input: Dataframe with features, dataframe with target, threshold value, parameter for cross-validstion, self-made scorer
    checks for best resample method for a given threshold of fpr using logistic regression
    Output: 
    """

    selected_features = features.columns   # use all columns

    strategies = [   # combination of strategy name, sampling method and class_weights

        # No treatment of class imbalance
        (
            "baseline",
            "passthrough",
            None
        ),

        # Randomly duplicate minority-class observations
        (
            "oversampling",
            RandomOverSampler(random_state=42),
            None
        ),

        # Randomly remove majority-class observations
        (
            "undersampling",
            RandomUnderSampler(random_state=42),
            None
        ),

        # Handle imbalance through Logistic Regression itself
        (
            "class_weights",
            "passthrough",
            "balanced"
        ),

        # Generate synthetic minority observations
        (
            "SMOTE",
            SMOTE(random_state=42),
            None
        )
    ]


    # ------------------------------------------------------------
    # Hyperparameter search space
    # ------------------------------------------------------------

    param_grid = {
        "estimator__C": [
            0.001,
            0.01,
            0.1,
            1,
            10,
            100
        ]
    }


    # ------------------------------------------------------------
    # GridSearch for every imbalance strategy
    # ------------------------------------------------------------

    results = []

    best_models = {}
    
    print('threshold: ', threshold)

    for name, sampler, class_weight in strategies:

        sampling_pipeline = Pipeline([
            (
                "scaler",
                StandardScaler()
            ),
            (
                "sampler",
                sampler
            ),
            (
                "estimator",
                LogisticRegression(
                    solver="lbfgs",
                    class_weight=class_weight,
                    max_iter=1000,
                    random_state=42
                )
            )
        ])
        grid_search = GridSearchCV(estimator = sampling_pipeline,
                              param_grid = param_grid,
                               cv = cv,
                              scoring = scoring[threshold],
                              n_jobs = -1)
        grid_search.fit(features, target)
        model = grid_search.best_estimator_.named_steps['estimator']
        print(name)
        print(f"{grid_search.scoring} on Validationset: {grid_search.best_score_:.3f}")
        print(model)
        print('#'*11)
        

   

