# Basic stuff
from termcolor import colored
from .utils import *
from collections import Counter
# Imbalanced-learn
from imblearn.over_sampling import RandomOverSampler
from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import KMeansSMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.under_sampling import ClusterCentroids
from imblearn.under_sampling import EditedNearestNeighbours
from sklearn.cluster import KMeans
from sklearn.cluster import MiniBatchKMeans
  

def get_custom_undersampling(method, X, y):
    """
    Calculates no samples per category for custom undersampling.

    Args:
        method: Undersampling method
        X (list or np.array): Input data
        y (list or np.array): Categories
        
    Returns:
        Samples per category (dict)
    """
    cnt = Counter(y)
    labels = []
    for key,n in cnt.items():
        if n > method["max_samples"]:
            nn = method["max_samples"]
            if "max_decrease_factor" in method and nn < n / method["max_decrease_factor"]:
                nn = int(n / method["max_decrease_factor"])
            cnt[key] = nn
            if n != nn:
                labels.append(key)
            
    # Edited Nearest Neighbors
    if method["sampler"] == "e":
        return labels
            
    return cnt


def get_custom_oversampling(method, X, y):
    """
    Calculates no samples per category for custom oversampling.

    Args:
        method: Oversampling method
        X (list or np.array): Input data
        y (list or np.array): Categories
        
    Returns:
        Samples per category (dict)
    """
    
    cnt = Counter(y)
    for key,n in cnt.items():
        if n < method["min_samples"]:
            nn = method["min_samples"]
            if "max_increase_factor" in method and nn > n * method["max_increase_factor"]:
                nn = int(n * method["max_increase_factor"])
            cnt[key] = nn
            
    return cnt


def resample(session, X, y, verbose=1):
    """
    Runs resampling on data.

    Args:
        session: Session object (created in load_data())
        X (list or np.array): Input data
        y (list or np.array): Categories
        verbose (int): Set verbose (output messages) level (0 for no output messages) (default: 1)
        
    Returns:
        Resampled data
    """
    
    # Check training set size before resampling
    if type(X) == list:
        x_orig = len(X)
    else:
        x_orig = X.shape[0]         
    
    # Resampling
    ycnt_orig = Counter(y)
    for method in list(session["resample"]["methods"]):
        # Undersampling
        if method["sampler"] in ["u","c","e"]:
            strategy = method["strategy"]
            if strategy == "custom":
                strategy = get_custom_undersampling(method, X, y)
        # Oversampling
        if method["sampler"] in ["o","k","s"]:
            strategy = method["strategy"]
            if strategy == "custom":
                strategy = get_custom_oversampling(method, X, y)
        
        # Undersamplers
        if method["sampler"] == "u":
            rsmp = RandomUnderSampler(random_state=session["resample"]["seed"], sampling_strategy=strategy)
            X, y = rsmp.fit_resample(X, y)
        if method["sampler"] == "c": 
            clusters = 8
            if "clusters" in method:
                clusters = method["clusters"]
            rsmp = ClusterCentroids(estimator=KMeans(n_clusters=clusters, n_init="auto", random_state=session["resample"]["seed"]), random_state=session["resample"]["seed"], sampling_strategy=strategy)
            X, y = rsmp.fit_resample(X, y)
        if method["sampler"] == "e": 
            rsmp = EditedNearestNeighbours(sampling_strategy=strategy)
            X, y = rsmp.fit_resample(X, y)
            
        # Oversamplers
        if method["sampler"] == "o":
            rsmp = RandomOverSampler(random_state=session["resample"]["seed"], sampling_strategy=strategy)
            X, y = rsmp.fit_resample(X, y)
        if method["sampler"] == "s":
            rsmp = SMOTE(random_state=session["resample"]["seed"], sampling_strategy=strategy)
            X, y = rsmp.fit_resample(X, y)
        if method["sampler"] == "k":
            clusters = 8
            if "clusters" in method:
                clusters = method["clusters"]
            rsmp = KMeansSMOTE(kmeans_estimator=MiniBatchKMeans(n_clusters=clusters, n_init="auto", random_state=session["resample"]["seed"]), random_state=session["resample"]["seed"], sampling_strategy=strategy)
            X, y = rsmp.fit_resample(X, y)
    
    if verbose >= 1:
        affected = 0
        ycnt_rsmp = Counter(y)
        for cat, n_orig in ycnt_orig.items():
            n_rsmp = ycnt_rsmp[cat]
            if n_orig != n_rsmp:
                affected += 1
        tot_orig = sum(ycnt_orig.values())
        tot_rsmp = sum(ycnt_rsmp.values())
        diff = tot_rsmp-tot_orig
        diff_pct = (tot_rsmp-tot_orig)/tot_orig
        if diff <= 0:
            diff = f"{diff}"
            diff_pct = f"{diff_pct*100:.1f}%"
            col = "green"
        else:
            diff = f"+{diff}"
            diff_pct = f"+{diff_pct*100:.1f}%"
            col = "red"
        info("Resampling affected " + colored(affected, "blue") + " categories and training set size changed with " + colored(diff, col) + " samples (" + colored(diff_pct, col) + ")")    
    
    return X, y
