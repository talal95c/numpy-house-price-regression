"""
NumPy House Price Regression

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - impute_nan_with_mean
import numpy as np

def impute_nan_with_mean(X):
    """
    Remplace chaque NaN par la moyenne de sa colonne.
    Si une colonne ne contient que des NaN, elle est remplie avec 0.
    """
    X = np.asarray(X, dtype=float)
    temp = X.copy()

    # Moyenne de chaque colonne
    with np.errstate(invalid="ignore"):
        column_means = np.nanmean(temp, axis=0)

    # Une moyenne NaN signifie que toute la colonne était composée de NaN
    column_means = np.nan_to_num(column_means, nan=0.0)

    # Positions de tous les NaN
    rows, columns = np.where(np.isnan(temp))

    # Remplacement par la moyenne correspondant à chaque colonne
    temp[rows, columns] = column_means[columns]

    return temp

# Step 2 - compute_iqr_bounds
def compute_iqr_bounds(X, k=1.5):
    X = np.asarray(X,dtype=float)
    q1 = np.percentile(X,25,axis=0)
    q3 = np.percentile(X,75,axis=0)




    iqr = q3 - q1 

    upper = q1 - k * iqr
    lower = q3 + k * iqr 


    


    return (upper,lower)

# Step 3 - clip_columns
def clip_columns(X, lower, upper):
    # TODO: Clip every entry of a feature matrix to per-column lower/upper bounds.
    pass

    X = np.asarray(X,dtype=float)
    lower = np.asarray(lower,dtype=float)
    upper = np.asarray(upper,dtype=float)

    X_clip = np.clip(X,lower,upper)
    return X_clip

# Step 4 - make_ratio_feature
def make_ratio_feature(numerator, denominator, eps=1e-8):
    # TODO: Form a derived ratio feature from two 1-D arrays with safe division.
    pass


    numerator = np.asarray(numerator,dtype=float)
    denominator=np.asarray(denominator,dtype=float)

    if numerator.shape== denominator.shape:
        denominator += eps
        return numerator / denominator
    
    else :
        return None

# Step 5 - append_column
def append_column(X, col):
    # TODO: Horizontally append one 1-D feature column onto a design matrix.
    pass
    X = np.asarray(X,dtype=float)
    col = np.asarray(col,dtype=float).reshape(-1,1)

    return np.concatenate([X,col],axis=1)

# Step 6 - one_hot_encode
def one_hot_encode(labels):
    labels = np.asarray(labels).reshape(-1)

    unique = np.unique(labels)

    one_hot = np.zeros((labels.shape[0], unique.shape[0]), dtype=float)

    for i, label in enumerate(labels):
        j = np.where(unique == label)[0][0]
        one_hot[i, j] = 1.0

    return one_hot

# Step 7 - fit_standardizer
def fit_standardizer(X):
    # TODO: Compute per-column mean and std used to standardize features...
    X = np.asarray(X,dtype=float)

    mean = np.mean(X,axis=0)
    std = np.std(X,axis=0)
    
    std = np.where(std == 0,1.0,std)

    return (mean,std)

# Step 8 - apply_standardizer
def apply_standardizer(X, mean, std):
    # TODO: Return the scaled matrix (X - mean) / std via broadcasting.
    pass

    X = np.asarray(X,dtype=float)
    mean=np.asarray(mean,dtype=float) 
    std= np.asarray(std,dtype=float)

    output = (X-mean)/std 
    return output

# Step 9 - add_bias_column
def add_bias_column(X):
    # TODO: Prepend a column of ones to a 2-D feature matrix X...
    X = np.asarray(X,dtype=float)
    bias = np.ones([X.shape[0],1])

    n = np.concatenate([bias,X],axis=1)

    return n

# Step 10 - make_shuffled_indices (not yet solved)
# TODO: implement

# Step 11 - partition_indices (not yet solved)
# TODO: implement

# Step 12 - subset_xy (not yet solved)
# TODO: implement

# Step 13 - ols_fit (not yet solved)
# TODO: implement

# Step 14 - ols_predict (not yet solved)
# TODO: implement

# Step 15 - mean_absolute_error (not yet solved)
# TODO: implement

# Step 16 - root_mean_squared_error (not yet solved)
# TODO: implement

# Step 17 - r_squared (not yet solved)
# TODO: implement

# Step 18 - residual_summary (not yet solved)
# TODO: implement

# Step 19 - prepare_cleaned_features (not yet solved)
# TODO: implement

# Step 20 - assemble_feature_matrix (not yet solved)
# TODO: implement

# Step 21 - make_train_val_test (not yet solved)
# TODO: implement

# Step 22 - standardize_and_add_bias (not yet solved)
# TODO: implement

# Step 23 - evaluate_predictions (not yet solved)
# TODO: implement

# Step 24 - house_price_pipeline (not yet solved)
# TODO: implement

