# Basic stuff
from termcolor import colored
from .utils import *
from collections import Counter
# Imbalanced-learn
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE
  

def rnd_undersampling(session, X, y):
    """
    Runs random undersampling on data.

    Args:
        session: Session object (created in load_data())
        X (list or np.array): Input data
        y (list or np.array): Categories
        
    Returns:
        Undersampled data
    """
    
    # Set no per label
    lcnt = Counter(y)
    for key,n in lcnt.items():
        if n > session["resample"]["max_samples"]:
            nn = session["resample"]["max_samples"]-n
            if nn/n*-1 > session["resample"]["decrease_limit"]:
                nn = -int(n * session["resample"]["decrease_limit"])
            lcnt.update({key: nn})
    # Perform undersampling
    rsmp = RandomUnderSampler(random_state=session["resample"]["seed"], sampling_strategy=lcnt)
    X, y = rsmp.fit_resample(X, y)
    return X, y


def rnd_oversampling(session, X, y):
    """
    Runs random oversampling on data.

    Args:
        session: Session object (created in load_data())
        X (list or np.array): Input data
        y (list or np.array): Categories
        
    Returns:
        Oversampled data
    """
    
    # Set no per label
    lcnt = Counter(y)
    for key,n in lcnt.items():
        if n < session["resample"]["min_samples"]:
            nn = session["resample"]["min_samples"] - n
            if nn/n > session["resample"]["increase_limit"]:
                nn = int(n * session["resample"]["increase_limit"])
            lcnt.update({key: nn})
    # Perform oversampling
    rsmp = RandomOverSampler(random_state=session["resample"]["seed"], sampling_strategy=lcnt)
    X, y = rsmp.fit_resample(X, y)
    return X, y
    

def smote_oversampling(session, X, y):
    """
    Runs SMOTE oversampling on data.

    Args:
        session: Session object (created in load_data())
        X (list or np.array): Input data
        y (list or np.array): Categories
        
    Returns:
        Oversampled data
    """
    
    # Error check
    if "auto" in session["resample"]:
        lcnt = "auto"
    else:
        # Set no per label
        lcnt = Counter(y)
        for key,n in lcnt.items():
            if n < session["resample"]["min_samples"]:
                nn = session["resample"]["min_samples"] - n
                if nn/n > session["resample"]["increase_limit"]:
                    nn = int(n * session["resample"]["increase_limit"])
                lcnt.update({key: nn})

    # Perform oversampling
    rsmp = SMOTE(random_state=session["resample"]["seed"], sampling_strategy=lcnt)
    X, y = rsmp.fit_resample(X, y)
    return X, y
    

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
    for mode in list(session["resample"]["mode"]):
        if mode == "u": 
            X, y = rnd_undersampling(session, X, y)
        if mode == "o":
            X, y = rnd_oversampling(session, X, y)
        if mode == "s":
            X, y = smote_oversampling(session, X, y)
        
    # Check training set size after resampling
    if type(X) == list:
        x_rsmp = len(X)
    else:
        x_rsmp = X.shape[0]
    
    if verbose >= 1:
        if x_rsmp < x_orig:
            info("Resampling reduced no samples with " + colored(f"{x_orig-x_rsmp} ", "green") + "(" + colored(f"{(x_orig-x_rsmp)/x_orig*100:.1f}%", "green") + ")")
        elif x_rsmp > x_orig:
            info("Resampling increased no samples with " + colored(f"{x_rsmp-x_orig} ", "green") + "(" + colored(f"{(x_rsmp-x_orig)/x_orig*100:.1f}%", "green") + ")")
        else:
            info("Resampling did not change no samples")
    return X, y
