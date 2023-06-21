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
            nn = session["resample"]["max_samples"]
            if nn < n / session["resample"]["max_decrease"]:
                nn = int(n / session["resample"]["max_decrease"])
            lcnt[key] = nn
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
            nn = session["resample"]["min_samples"]
            if nn > n * session["resample"]["max_increase"]:
                nn = int(n * session["resample"]["max_increase"])
            lcnt[key] = nn
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
                nn = session["resample"]["min_samples"]
                if nn > n * session["resample"]["max_increase"]:
                    nn = int(n * session["resample"]["max_increase"])
                lcnt[key] = nn
    # Perform SMOTE oversampling
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
    ycnt_orig = Counter(y)
    for mode in list(session["resample"]["mode"]):
        if mode == "u": 
            X, y = rnd_undersampling(session, X, y)
        if mode == "o":
            X, y = rnd_oversampling(session, X, y)
        if mode == "s":
            X, y = smote_oversampling(session, X, y)
    
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
        info("Resampling affected " + colored(affected, "blue") + " categories and dataset size changed with " + colored(diff, col) + " samples (" + colored(diff_pct, col) + ")")    
    
    return X, y
