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

# Step 10 - make_shuffled_indices
def make_shuffled_indices(n_samples, seed):
    # TODO: Create a reproducibly shuffled permutation of row indices.
    
    n_ordererd = np.arange(n_samples)
    random_generator = np.random.RandomState(seed)
    
    permu = random_generator.permutation(n_ordererd)
    return permu

# Step 11 - partition_indices
def partition_indices(indices, train_ratio, val_ratio):
    # TODO: Split a shuffled index array into train, validation, and test index arrays.
    pass
    
    n_sample=indices.shape[0]
    train_ratio=int(n_sample*train_ratio)
    val_ratio = int(val_ratio*n_sample)

    
    train_slice = indices[0 : train_ratio]
    val_slice = indices[train_ratio : train_ratio+val_ratio]
    test_slice = indices[train_ratio+val_ratio : ]

    return (train_slice,val_slice,test_slice)

# Step 12 - subset_xy
def subset_xy(X, y, indices):
    # TODO: Select the rows of X and y at the given indices.
    X = np.asarray(X,dtype=float)
    y = np.asarray(y,dtype=float)
    #indices = np.asarray(indices,dtype=float)
    

    X_sub = X[indices]
    y_sub = y[indices]




    return X_sub,y_sub
    pass

# Step 13 - ols_fit
def ols_fit(X, y):
    # TODO: return the ordinary-least-squares weight vector for a linear model.
    
    X = np.asarray(X,dtype=float)
    y = np.asarray(y,dtype=float)

    GramMatrix = X.T @ X
    rightMatrix = X.T @ y 
    Gram_inv = np.linalg.inv(GramMatrix)
    theta = Gram_inv @ rightMatrix
    return theta

# Step 14 - ols_predict
def ols_predict(X, theta):
    # TODO: Predict continuous targets with a fitted linear model.
    pass

    X = np.asarray(X,dtype=float)
    theta = np.asarray(theta)

    return X @ theta

# Step 15 - mean_absolute_error
def mean_absolute_error(y_true, y_pred):
    # TODO: return the mean absolute error between targets and predictions
    pass
    y_true = np.asarray(y_true,dtype=float)
    y_pred = np.asarray(y_pred,dtype=float)
     
    error = np.mean(abs(y_true-y_pred))
    return error

# Step 16 - root_mean_squared_error
def root_mean_squared_error(y_true, y_pred):
    """Compute root mean squared error between targets and predictions.

    Args:
        y_true (np.ndarray): Ground-truth targets, shape (N,).
        y_pred (np.ndarray): Predicted targets, shape (N,).

    Returns:
        float: RMSE value.
    """
    # TODO: return the root mean squared error as a Python float

    error = abs(y_true - y_pred)
    MSE = np.sqrt(np.mean(error**2))
    return MSE

# Step 17 - r_squared
def r_squared(y_true, y_pred):
    # TODO: Compute R^2 = 1 - SS_res/SS_tot (return 0.0 if SS_tot is 0)...
    

    y_true = np.asarray(y_true,dtype=float)
    y_pred = np.asarray(y_pred,dtype=float)
    y_pred_mean = np.mean(y_pred)

    SS_tot = np.mean((y_true-y_pred_mean)**2) 
    
    if SS_tot == 0:
        return 0 
    else : 
        SS_res = np.mean((y_pred-y_true)**2)
        return 1.0 - SS_res/SS_tot

# Step 18 - residual_summary
def residual_summary(y_true,y_pred):
    # TODO: Return a compact dict summarizing prediction residuals...
    pass

    y_true = np.asarray(y_true,dtype=float)
    y_pred = np.asarray(y_pred,dtype=float)

    r = y_true - y_pred

    mean  = float(np.mean(r))
    std = float(np.std(r))
    median_abs = float(np.median(abs(r)))

    return {
    "mean": mean,
    "std": std,
    "median_abs": median_abs
}

# Step 19 - prepare_cleaned_features
def prepare_cleaned_features(X, iqr_k=1.5):
    """Impute NaNs then IQR-clip columns to produce a clean numeric matrix.

    Args:
        X: (N, F) array-like of floats, may contain NaN.
        iqr_k: IQR multiplier passed to compute_iqr_bounds (default 1.5).

    Returns:
        (N, F) float ndarray with no NaNs, columns clipped to IQR bounds.
    """
    # TODO: Produce a clean numeric matrix via impute then IQR clip
    X = np.asarray(X,dtype=float)
    X_no_nan = impute_nan_with_mean(X)
    lower,upper  = compute_iqr_bounds(X_no_nan,iqr_k)
    X_clip = clip_columns(X_no_nan,lower,upper)

    return X_clip

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

