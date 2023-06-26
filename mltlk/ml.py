# Basic stuff
from termcolor import colored
import numpy as np
import pandas as pd
from collections import Counter
from customized_table import *
import time
import matplotlib.pyplot as plt
import re
from .utils import *
# Pre-processing
from sklearn.base import is_classifier, is_regressor
from sklearn.utils import shuffle
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, Normalizer, OneHotEncoder, OrdinalEncoder, LabelEncoder
# Evaluation
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import mean_absolute_error,r2_score,mean_squared_error
from sklearn.calibration import CalibratedClassifierCV
# Cross-validation
from sklearn.model_selection import KFold
from sklearn.base import clone
# File stuff
from pickle import dump,load
from os.path import exists
from os import makedirs
import gzip
# Resampling
from .resampling import resample
# Word vectors
from .word2vec import *
from .embeddings import *


def load_data(file, 
              Xcols=None, 
              ycol=None,
              mode="classification",
              preprocess=None,
              shuffle_data=False,
              seed=None,
              min_samples=None,
              encode_categories=False,
              category_descriptions=None,
              clean_text="letters digits",
              stopwords=None,
              max_features=None,
              tf_idf=True,
              w2v_vector_size=75,
              w2v_rebuild=False,
              embeddings_size=75,
              embeddings_max_length=None,
              verbose=1):
    """
    Loads and pre-processes a data file and returns a session.

    Args:
        Xcols (list, range or None): Indexes of columns in the csv to be used as input features, for example [0,1,2]. If None, all columns except the last will be used as features (default: None)
        ycol (int or None): Index of the column in the csv to be used as category. If None, the last column will be used as category (default: None)
        mode (str): Type of target - 'classification' or 'regression' (default: 'classification')
        preprocess (str or None): Pre-processing to use: 'normalize', 'scale', 'one-hot', 'ordinal', 'bag-of-words', 'word2vec', 'embeddings' or None. If None, no pre-processing will be used (default: None)
        shuffle_data (bool): True if data shall be shuffled (default: False)
        seed (int or None): Seed value to be used by the randomizer. If None, no seed will be used and results can differ between runs (default: None)
        min_samples (int or None): Set if minority categories (categories with less than min_samples examples) shall be removed. If None, all categories will be included (default: None)
        encode_categories (bool): True if category labels shall be encoded as integers (default: False)
        category_descriptions (dict or None): Optional descriptions for categories, or None for no descriptions. Descriptions shall contain categories as keys and descriptive texts as values, for example {1: 'Iris-setosa'} (default: None)
        clean_text (str or None): When cleaning text data, set if only letters ('letters') or both letters and digits ('letters digits') shall remain in the text. If None, no cleaning will be used (default: 'letters digits')
        stopwords (str, list or None): Lists of stopwords to be used for text pre-processing. Can be either languages (from the nltk.corpus package) or paths to csv files, for example ['english', 'data/custom_stopwords.csv']. If None, no stopwords will be used (default: None)
        max_features (int or None): Max features to be used for bag-of-words text pre-processing. If None, all features will be used (default: None)
        tf_idf (bool): True if TF-IDF shall be used for bag-of-words text pre-processing (default: True)
        w2v_vector_size (int): Size of word vectors for word2vec text pre-processing (default: 75)
        w2v_rebuild (bool): Set if previously stored word2vec model, if found, shall be loaded (False) or if the model shall be rebuilt (True) (default: False)
        embeddings_size (int): Size of word vectors for embeddings pre-processing (default: 75)
        embeddings_max_length (int or None): Max size of word vectors for embeddings pre-processing. If none, word vectors will not be cut (default: None)
        verbose (int): Set verbose (output messages) level (0 for no output messages) (default: 1)

    Returns:
        session: Session object.
    """
    session = {}
    
    # Check params
    if not check_param(mode, "mode", [str], vals=["classification", "regression"]): return None
    session["mode"] = mode
    if not check_param(preprocess, "preprocess", [str,None], vals=["normalize", "scale", "one-hot", "ordinal", "bag-of-words", "word2vec", "embeddings"]): return None
    session["preprocess"] = preprocess
    if not check_param(shuffle_data, "shuffle_data", [bool]): return None
    if not check_param(seed, "seed", [int,None], expr=seed is None or seed>=0, expr_msg="seed cannot be negative"): return None
    if not check_param(min_samples, "min_samples", [int,None], expr=min_samples is None or min_samples>1, expr_msg="min samples higher than 1"): return None
    if not check_param(encode_categories, "encode_categories", [bool]): return None
    if not check_param(category_descriptions, "category_descriptions", [dict,None]): return None
    if not check_param(clean_text, "clean_text", [str,None], vals=["letters", "letters digits"]): return None
    if not check_param(stopwords, "stopwords", [list,str,None]): return None
    if not check_param(max_features, "max_features", [int,None], expr=max_features is None or max_features>1, expr_msg="max features must be 1 or higher"): return None
    if not check_param(tf_idf, "tf_idf", [bool], None): return None
    if not check_param(w2v_vector_size, "w2v_vector_size", [int], expr=w2v_vector_size>1, expr_msg="vector size must be larger than 1"): return None
    if not check_param(w2v_rebuild, "w2v_rebuild", [bool], None): return None
    if not check_param(embeddings_size, "embeddings_size", [int], expr=embeddings_size>1, expr_msg="embeddings size must be larger than 1"): return None
    if not check_param(embeddings_max_length, "embeddings_max_length", [int,None], expr=embeddings_max_length is None or embeddings_max_length>1, expr_msg="embeddings max length must be larger than 1"): return None
    if not check_param(verbose, "verbose", [int], vals=[0,1]): return None
    
    # Load data
    if not exists(file):
        error("data file " + colored(file, "cyan") + " not found")
        return None
    data = pd.read_csv(file)
    cols = list(data.columns)
    data = data.values
    
    session["file"] = file
    
    # Set X features to be all but last column
    if Xcols is None:
        Xcols = range(0,len(data[0]) - 1)
    # Set y to be last column
    if ycol is None:
        ycol = len(data[0]) - 1
    
    # Update columns
    session["columns"] = []
    for idx in Xcols:
        session["columns"].append(cols[idx])
    session["target"] = cols[ycol]
    
    # Convert to X and y
    X = []
    y = []
    for r in data:
        row = []
        for c,val in enumerate(r):
            if c in Xcols:
                row.append(val)
        if len(row) == 1:
            row = row[0]
        X.append(row)
        y.append(r[ycol])
        
    # If single feature only and not text, convert to list of lists
    if type(X[0]) != list and type(X[0]) != str:
        X = [[xi] for xi in X]
        
    # Check if all yi is integer
    y_tmp = [yi for yi in y if type(yi) in [float, np.float64]]
    if len(y_tmp) == len(y):
        y_tmp = [yi for yi in y_tmp if float(yi).is_integer()]
        if len(y_tmp) == len(y):
            # Convert to int
            y = [int(yi) for yi in y]
            
    # Shuffle
    if shuffle_data:
        X, y = shuffle(X, y, random_state=seed)
            
    # Update session
    session["X_original"] = X
    session["y_original"] = y
    session["X"] = X.copy()
    session["y"] = y.copy()
    
    # Regression
    if mode == "regression":
        if verbose >= 1:
            if type(session["y"]) == list:
                nex = len(session["y"])
            else:
                nex = session["y"].shape[0]
            info("Loaded " + colored(f"{nex}", "blue") + " examples for regression target")
        return session
    
    # Check type of categories
    y_tmp = [yi for yi in y if type(yi) in [float, np.float64]]
    if len(y_tmp) > 0:
        warning("Data contains float categories and regression preprocess is not set")
    
    # Skip minority categories
    if min_samples is not None:
        cnt = Counter(session["y"])
        X = []
        y = []
        for xi,yi in zip(session["X"], session["y"]):
            if cnt[yi] >= min_samples:
                X.append(xi)
                y.append(yi)
        session["X_original"] = X
        session["y_original"] = y
        session["X"] = X.copy()
        session["y"] = y.copy()
        if verbose >= 1:
            s = ""
            for li,ni in cnt.items():
                if ni < min_samples:
                    s += li + ", "
            if s != "":
                info("Removed minority categories " + colored(s[:-2], "cyan"))
        
    # Check text inputs without text preprocessing
    if preprocess not in ["bag-of-words", "word2vec", "embeddings"]:
        if type(session["X"][0]) == str:
            error("Input seems to be text but no text-preprocessing is set")
            return None
        
    # Check ordinal features without encoding
    if preprocess not in ["one-hot", "ordinal"]:
        if type(session["X"][0]) != str:
            for xi in session["X"][0]:
                if type(xi) == str:
                    error("Input contains ordinal features but no encoding is set (use " + colored("one-hot", "blue") + " or " + colored("ordinal", "blue") + ")")
                    return None
    
    # Clean text inputs
    if clean_text is not None and preprocess in ["word2vec", "bag-of-words", "embeddings"]:
        if clean_text == "letters digits":
            info("Clean texts keeping letters and digits")
        elif clean_text == "letters":
            info("Clean texts keeping letters only")
        for i,xi in enumerate(session["X"]):
            # Remove new line and whitespaces
            xi = xi.replace("<br>", " ")
            xi = xi.replace("&nbsp;", " ")
            xi = xi.replace("\n", " ")
            # Remove special chars
            if clean_text == "letters digits":
                xi = re.sub("[^a-zA-Z0-9åäöÅÄÖ ]", " ", xi)
            elif clean_text == "letters":
                xi = re.sub("[^a-zA-ZåäöÅÄÖ ]", " ", xi)
            # Remove multiple whitespaces
            xi = " ".join(xi.split())
            # Set to lower case
            xi = xi.lower()
            # Strip trailing/leading whitespaces
            xi = xi.strip()
            session["X"][i] = xi
        session["X_original"] = session["X"].copy()
    
    # Encode categories
    if encode_categories:
        session["label_encoder"] = LabelEncoder().fit(session["y"])
        session["y"] = session["label_encoder"].transform(session["y"])
        if verbose >= 1:
            info("Categories encoded")
            
    # Category descriptions
    if category_descriptions is not None:
        session["descriptions"] = category_descriptions
        
    # Bag-of-words representation for input texts
    if preprocess == "bag-of-words":
        sw = load_stopwords(stopwords, verbose=verbose)
        l = "Used bag-of-words"
        if stopwords not in [[],"",None]:
            l += " with stopwords removed"
        elif verbose >= 1:
            l = "Used bag-of-words"
        session["bow"] = CountVectorizer(stop_words=sw, max_features=max_features).fit(session["X"]) #TODO: ngram_range=ngram
        session["X"] = session["bow"].transform(session["X"])
        session["stopwords"] = sw
        
        # TF-IDF conversion for bag-of-words
        if tf_idf:
            session["TF-IDF"] = TfidfTransformer().fit(session["X"])
            session["X"] = session["TF-IDF"].transform(session["X"])
            l += " and TF-IDF"
        if verbose >= 1:
            info(l)
            
    # Word2vec
    if preprocess == "word2vec":
        load_word2vec_data(session, w2v_vector_size, w2v_rebuild, stopwords, verbose=verbose)
        
    # Keras embeddings
    if preprocess == "embeddings":
        load_embeddings_data(session, embeddings_size, embeddings_max_length, stopwords, verbose=verbose)
    
    # One-hot encoding
    if preprocess == "one-hot":
        session["scaler"] = OneHotEncoder(handle_unknown="ignore").fit(session["X"])
        session["X"] = session["scaler"].transform(session["X"])
        if verbose >= 1:
            info("Transformed input data using one-hot encoding")
            
    # Ordinal encoding
    if preprocess == "ordinal":
        session["scaler"] = OrdinalEncoder().fit(session["X"])
        session["X"] = session["scaler"].transform(session["X"])
        if verbose >= 1:
            info("Transformed input data using ordinal encoding")
        
    # Standard scaler
    if preprocess == "scale":
        session["scaler"] = StandardScaler().fit(session["X"])
        session["X"] = session["scaler"].transform(session["X"])
        if verbose >= 1:
            info("Scaled input data using standard scaler")
            
    # Normalize
    if preprocess == "normalize":
        session["scaler"] = Normalizer().fit(session["X"])
        session["X"] = session["scaler"].transform(session["X"])
        if verbose >= 1:
            info("Normalized input data")
            
    if verbose >= 1:
        if type(session["y"]) == list:
            nex = len(session["y"])
        else:
            nex = session["y"].shape[0]
        session["categories"] = len(Counter(session['y_original']))
        info("Loaded " + colored(f"{nex}", "blue") + " examples in " + colored(f"{session['categories']}", "blue") + " categories")
    
    return session


def data_stats(session, max_rows=None, show_graph=False):
    """
    Show statistics about the loaded dataset.

    Args:
        session: Session object (created in load_data())
        max_rows (int or None): Max categories to show. If None, all categories are shown (default: None)
        show_graph (bool): Show bar graph with examples per category (default: False)
    """
    
    # Check params
    if not check_param(session, "session", [dict], expr=session is not None, expr_msg="session is None"): return
    if not check_param(max_rows, "max_rows", [int,None], expr=max_rows is None or max_rows>=1, expr_msg="max rows must be at least 1"): return None
    if not check_param(show_graph, "show_graph", [bool]): return None
    
    # Regression
    if session["mode"] == "regression":
        if type(session["y"]) == list:
            nex = len(session["y"])
        else:
            nex = session["y"].shape[0]
            
        t = CustomizedTable(["",session["target"]])
        t.column_style(1, {"color": "value", "num-format": "int-2"})
        t.add_row(["Examples:", nex])
        t.add_row(["Mean:", float(np.mean(session['y']))])
        t.add_row(["Min:", float(np.min(session['y']))])
        t.add_row(["Max:", float(np.max(session['y']))])
        t.add_row(["Stdev:", float(np.std(session['y']))])
        t.display()
        
        return
    
    # Get categories
    y = session["y"]
    
    cnt = Counter(y)
    tab = []
    for key,no in cnt.items():
        tab.append([key,no,f"{no/len(y)*100:.1f}%"])
    tab = sorted(tab, key=lambda x: x[1], reverse=True)
    rno = 0
    labels = []
    vals = []
    for r in tab:
        rno += r[1]
        r.append(f"{rno/len(y)*100:.1f}%")
        labels.append(r[0])
        vals.append(r[1])
    if max_rows is not None:
        if type(max_rows) != int or max_rows <= 0:
            error("Max rows must be integer and > 0")
            return
        tab = tab[:max_rows]
    
    # Graph of no per category
    if show_graph:
        plt.figure(figsize=(14, 4))
        plt.bar(labels, vals, color="maroon", width=0.4)
        plt.ylim(bottom=0)
        plt.xticks(rotation=90)
        plt.show()
    
    # Reformat to 3 columns
    tab2 = [[],[],[]]
    s = int(len(tab) / 3)
    if len(tab) % 3 != 0:
        s += 1
    c = 0
    for i,r in enumerate(tab):
        tab2[c].append(r)
        if (i+1) % s == 0:
            c += 1
    
    # Show table
    if "descriptions" in session:
        t = CustomizedTable(["Category", "No", "%", "Σ%", "Description", "Category", "No", "%", "Σ%", "Description", "Category", "No", "%", "Σ%", "Description"])
        t.column_style([0,5,10], {"color": "name"})
        t.column_style([1,6,11], {"color": "value"})
        t.column_style([2,7,12], {"color": "percent"})
        t.column_style([3,8,13], {"color": "green"})
        t.column_style([4,9,14], {"color": "#666"})
    else:
        t = CustomizedTable(["Category", "No", "%", "Σ%", "Category", "No", "%", "Σ%", "Category", "No", "%", "Σ%"])
        t.column_style([0,4,8], {"color": "name"}) # id
        t.column_style([1,5,9], {"color": "value"})
        t.column_style([2,6,10], {"color": "percent"})
        t.column_style([3,7,11], {"color": "green"})
    for i in range(0,s):
        r = []
        for j in range(0,3):
            if i < len(tab2[j]):
                if "label_encoder" in session and type(session["label_encoder"]) == LabelEncoder:
                    l = session["label_encoder"].inverse_transform([tab2[j][i][0]])[0]
                    r.append(f"{l} ({tab2[j][i][0]})")
                else:
                    r.append(tab2[j][i][0])
                r.append(tab2[j][i][1])
                r.append(tab2[j][i][2])
                r.append(tab2[j][i][3])
                if "descriptions" in session:
                    desc = ""
                    if tab2[j][i][0] in session["descriptions"]:
                        desc = session["descriptions"][tab2[j][i][0]]
                    r.append(desc)
        # Fill row, if not full
        rsize = 15
        if "descriptions" not in session:
            rsize = 12
        if len(r) < rsize:
            i = rsize - len(r)
            r += [""] * (i)
        t.add_row(r)
    
    # Overall stats
    if type(session["X"]) == list:
        fts = len(session["X"][0])
    else:
        fts = session["X"].shape[1]
    if "descriptions" in session:
        t.add_row(["Examples:", len(y), "", "", "", "Features:", fts, "", "", "", "Categories:", len(cnt), "", "", ""], style={"row-toggle-background": 0, "background": "#eee", "border": "top"})
        t.cell_style([0,5,10], -1, {"font": "bold", "color": "#666"})
    else:
        t.add_row(["Examples:", len(y), "", "", "Features:", fts, "", "", "Categories:", len(cnt), "", ""], style={"row-toggle-background": 0, "background": "#eee", "border": "top"})
        t.cell_style([0,4,8], -1, {"font": "bold", "color": "#666"})
    
    t.display()


def split_data(session,
               test_size=0.2,
               seed=None,
               stratify=False,
               verbose=1,
              ):
    """
    Split data into training and test sets.

    Args:
        session: Session object (created in load_data())
        test_size (float): Size of test set. Must be between 0 and 1 (default: 0.2)
        seed (int or None): Seed value to be used by the randomizer. If None, no seed will be used and results can differ between runs (default: None)
        stratify (bool): If stratify, test set will as much as possible keep the ratio between examples of each category (default: False)
        verbose (int): Set verbose (output messages) level (0 for no output messages) (default: 1)
    """
    
    # Check params
    if not check_param(session, "session", [dict], expr=session is not None, expr_msg="session is None"): return
    if not check_param(test_size, "test_size", [float,int], expr=test_size>0 and test_size<1, expr_msg="test size must be between 0 and 1"): return
    if not check_param(seed, "seed", [int,None], expr=seed is None or seed>=0, expr_msg="seed cannot be negative"): return
    if not check_param(stratify, "stratify", [bool]): return
    
    # Stratify
    if stratify:
        stratify = session["y"]
    else:
        stratify = None
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(session["X"], session["y"], test_size=test_size, random_state=seed, stratify=stratify)
    
    # Update session
    session["X_train"] = X_train
    session["X_test"] = X_test
    session["y_train"] = y_train
    session["y_test"] = y_test
    session["test_size"] = test_size
    session["train_size"] = 1 - test_size
    session["eval_mode"] = ""
    
    if verbose >= 1:
        # Info string
        if type(y_train) == list:
            ntr = len(y_train)
        else:
            ntr = y_train.shape[0]
        if type(y_test) == list:
            nte = len(y_test)
        else:
            nte = y_test.shape[0]
        s = "Split data using " + colored(f"{(1-test_size)*100:.0f}%", "blue") + " training data (" + colored(ntr, "blue") + " samples) and " + colored(f"{(test_size)*100:.0f}%", "blue") + " test data (" + colored(nte, "blue")+ " samples)"
        if seed is not None:
            s += " with seed " + colored(seed, "blue") 
        if stratify is not None:
            s += " and stratify"
        info(s)


def set_resample(session, 
                 methods=[],
                 seed=None,
                 verbose=2,
                ):
    """
    Sets over- and undersampling methods to be used when training models.

    Args:
        session: Session object (created in load_data())
        methods (list): List of the sampling methods to use on the training data. Each entry in the list must be a dict with settings for each sampling method. The supported methods are 'u' (random undersampling), 'e' (edited nearest neighbors undersampling), 'c' (Cluster Centroids undersampling), 'o' (random oversampling), 'k' (KMEans SMOTE oversampling) and 's' (SMOTE oversampling)
        Example:
        [{"sampler": "s", "strategy: "custom", "min_samples": 200, "max_increase": 4}, {"sampler": "e", "strategy": "auto"}]
        ... specifies that first is SMOTE oversampling used, then edited nearest neighbor undersampling. Strategy can be:
        'majority': resample only the majority category (undersampling only)
        'minority': resample only the minority category (oversampling only)
        'not minority': resample all classes but the minority category
        'not majority': resample all classes but the majority category
        'all': resample all categories
        'auto': equivalent to 'not minority' for undersampling, and 'not majority' for oversampling
        'custom': specifies the max and min samples for over- and/or undersampling. Combines with:
            'max_samples': Max samples of a category (undersampling only)
            'max_decrease_factor': Specifies the highest factor by which a category is decreased (undersampling, optional)
            'min_samples': Min samples of a category (oversampling only)
            'max_increase_factor': Specifies the highest factor by which a category is increased (oversampling, optional)
            'cluster': Number of clusters to use (Cluster Centroids and KMeans SMOTE only)
        seed (int or None): Seed value to be used by the randomizer. If None, no seed will be used and results can differ between runs (default: None)
        verbose (int): Set verbose (output messages) level (0 for no output messages) (default: 2)
    """
    
    # Check params
    if not check_param(session, "session", [dict], expr=session is not None, expr_msg="session is None"): return
    if not check_param(seed, "seed", [int,None], expr=seed is None or seed>=0, expr_msg="seed cannot be negative"): return
    if not check_param(verbose, "verbose", [int], vals=[0,1,2]): return

    # Sampler names
    names = {
        "u": "Random Undersampling",
        "e": "Edited Nearest Neighbors Undersampling",
        "c": "Cluser Centroids Undersampling",
        "o": "Random Oversampling",
        "s": "SMOTE Oversampling",
        "k": "KMeans SMOTE Oversampling",
    }
    
    # Methods placeholder
    session["resample"] = {
        "seed": seed,
        "methods": [],
    }

    # Check methods
    samplers = []
    for method in methods:
        # Error checks
        if method["sampler"] not in ["u","c","e","o","k","s"]:
            error("Invalid sampler " + colored(method["sampler"], "blue") + ". Must be " + colored("'u', 'c', 'e', 'o', 'k', 's'", "cyan"))
            return
        if "strategy" in method and method["strategy"] not in ["majority", "minority", "not majority", "not minority", "all", "auto", "custom"]:
            error("Invalid strategy " + colored(method["strategy"], "blue") + ". Must be " + colored("'majority', 'minority', 'not majority', 'not minority', 'all', 'auto', 'custom'", "cyan"))
            return
        if "strategy" not in method or method["strategy"] == "custom":
            if method["sampler"] in ["u","c","e"] and "max_samples" not in method:
                error("strategy 'custom' with undersampling requires 'max_samples' to be specified")
                return
            if method["sampler"] in ["o","k","s"] and "min_samples" not in method:
                error("strategy 'custom' with oversampling requires 'min_samples' to be specified")
                return
        if method["sampler"] in ["u","c","e"] and "strategy" in method and method["strategy"] == "minority":
            error("strategy 'minority' is not a valid strategy for undersampling")
            return
        if method["sampler"] in ["o","k","s"] and "strategy" in method and method["strategy"] == "majority":
            error("strategy 'majority' is not a valid strategy for oversampling")
            return
        for key in method.keys():
            if key not in ["sampler", "strategy", "clusters", "max_samples", "max_decrease_factor", "min_samples", "max_increase_factor"]:
                error("Invalid key " + colored(key, "cyan") + " for method")
                return
        # Check values
        if "max_samples" in method and not check_param(method["max_samples"], "max_samples", [int], expr=method["max_samples"]>=1, expr_msg="'max_samples' must be 1 or higher"): return
        if "max_decrease_factor" in method and not check_param(method["max_decrease_factor"], "max_decrease_factor", [float,int], expr=method["max_decrease_factor"]>1, expr_msg="'max_decrease_factor' must be 1 or higher"): return
        if "min_samples" in method and not check_param(method["min_samples"], "min_samples", [int], expr=method["min_samples"]>=1, expr_msg="'min_samples' must be 1 or higher"): return
        if "max_increase_factor" in method and not check_param(method["max_increase_factor"], "max_increase_factor", [float,int], expr=method["max_increase_factor"]>=1, expr_msg="'max_increase_factor' must be 1 or higher"): return
        if "clusters" in method and not check_param(method["clusters"], "clusters", [int], expr=method["clusters"]>=2, expr_msg="'clusters' must be 2 or higher"): return
        if "strategy" not in method:
            method["strategy"] = "custom"
        
        # All error checks done!
        session["resample"]["methods"].append(method)
        samplers.append(method["sampler"])
        info("Using sampler " + colored(names[method["sampler"]], "cyan") + " with sampling strategy " + colored(method["strategy"], "cyan"))
        
    # Check for multiple over/undersampling
    no = 0
    nu = 0
    for s in samplers:
        if s in ["u","c","e"]:
            nu += 1
        if s in ["o","k","s"]:
            no += 1
    if no > 1:
        warning("multiple oversampling")
    if nu > 1:
        warning("multiple undersampling")
        
    # Reset eval mode (for reload)
    session["eval_mode"] = ""
    
    # Show how resampling affects training data
    if verbose >= 2:
        # Split data (if not already splitted)
        if "X_train" not in session:
            split_data(session)
        ycnt_orig = Counter(session["y_train"])
        X, y = resample(session, session["X_train"], session["y_train"], verbose=0)
        ycnt_rsmp = Counter(y)
        
        # Convert to table and sort
        tab = []
        for cat,n_orig in ycnt_orig.items():
            n_rsmp = ycnt_rsmp[cat]
            if n_orig != n_rsmp:
                tab.append([n_orig, n_rsmp, cat])
        tab = sorted(tab, reverse=True)
        
        if len(tab) == 0:
            warning("Resampling had no effect")
        else:
            t = CustomizedTable(["Category", "Samples original", "Samples resampled", "Diff", "Diff (%)"])
            t.column_style([0], {"color": "id"})
            t.column_style([1,2], {"color": "value"})
            for r in tab:
                diff = r[1]-r[0]
                diff_pct = (r[1]-r[0])/r[0]
                if diff <= 0:
                    diff = f"{diff}"
                    diff_pct = f"{diff_pct*100:.1f}%"
                    col = "green"
                else:
                    diff = f"+{diff}"
                    diff_pct = f"+{diff_pct*100:.1f}%"
                    col = "red"
                if "label_encoder" in session and type(session["label_encoder"]) == LabelEncoder:
                    l = session["label_encoder"].inverse_transform([r[2]])[0]
                    cat = f"{l} ({r[2]})"
                else:
                    cat = r[2]
                t.add_row([cat, r[0], r[1], diff, diff_pct])
                t.cell_style([3,4], -1, {"color": col})
            # Total row
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
            t.add_row(["Total:", tot_orig, tot_rsmp, diff, diff_pct], style={"border": "top", "background": "#eee", "row-toggle-background": 0})
            t.cell_style(0, -1, {"font": "bold", "color": "black"})
            t.cell_style([3,4], -1, {"color": col})
            t.add_row(["Categories affected:", len(tab), f"{len(tab)/len(ycnt_orig)*100:.1f}%", "", ""], style={"background": "#eee", "row-toggle-background": 0})
            t.cell_style(0, -1, {"color": "black"})
            t.add_row(["Categories unchanged:", len(ycnt_orig)-len(tab), f"{(len(ycnt_orig)-len(tab))/len(ycnt_orig)*100:.1f}%", "", ""], style={"background": "#eee", "row-toggle-background": 0})
            t.cell_style(0, -1, {"color": "black"})
            t.add_row(["Training set size:", f"{session['train_size']*100:.1f}%", "", "", ""], style={"border": "bottom", "background": "#eee", "row-toggle-background": 0})
            t.cell_style(0, -1, {"color": "black"})
            print()
            t.display()
            print()
        
        
def clear_resample(session, verbose=1):
    """
    Clears resample settings.

    Args:
        session: Session object (created in load_data())
        verbose (int): Set verbose (output messages) level (0 for no output messages) (default: 1)
    """
    
    # Check params
    if not check_param(session, "session", [dict], expr=session is not None, expr_msg="session is None"): return
    if not check_param(verbose, "verbose", [int], vals=[0,1]): return
    
    if "resample" in session:
        del session["resample"]
        if verbose >= 1:
            info("Removed resample settings")
    else:
        warning("No resample settings found in session")


#
# Wraps a Keras model to have the same functions as a sklearn model.
#
class KerasWrapper:
    def __init__(self, model, epochs, batch_size, loss, optimizer):
        self.model = model
        self.fitted = False
        self.epochs = epochs
        self.batch_size = batch_size
        self.loss = loss
        self.optimizer = optimizer
        self.nout = self.model.layers[-1].output_shape[1]
        
    # Train Keras model
    def fit(self, X, y):
        if type(y[0]) == str:
            error("Keras models require numerical categories. Set " + colored("encode_categories", "cyan") + " to " + colored("True", "blue") + " when calling " + colored("load_data()", "cyan"))
            return
        
        # One-hot encode labels
        if self.nout > 1:
            from tensorflow.keras.utils import to_categorical
            y = to_categorical(y, len(np.unique(y)))
        
        # X must by np array
        if type(X) == list:
            X = np.asarray(np.asarray([xi for xi in X]))
        
        # Compile model
        self.model.compile(loss=self.loss, optimizer=self.optimizer, metrics=["accuracy"])

        # Train model
        self.model.fit(X, y, epochs=self.epochs, batch_size=self.batch_size, verbose=0)
        self.fitted = True
      
    # Predict with Keras model
    def predict(self, X):
        if not self.fitted:
            error("Model has not been trained")
            return None
        
        # X must by np array
        if type(X) == list:
            X = np.asarray(np.asarray([xi for xi in X]))
        
        # Get predictions
        y_pred = self.model(X)
        # Convert back from one-hot
        if self.model.layers[-1].output_shape[1] > 1:
            y_pred = np.argmax(y_pred, axis=1)
        else:
            y_pp = []
            for yi in y_pred:
                y_pp.append(int(round(yi.numpy()[0],0)))
            y_pred = y_pp
        # Return result
        return y_pred
    
    def clone(self):
        from tensorflow.keras.models import clone_model
        # To be absolutely sure the model is cloned we use both ways
        self.model = clone_model(self.model)
        self.model = self.model.__class__.from_config(self.model.get_config())
        self.fitted = False
        

def evaluate_model(model, 
                   session, 
                   reload=False, 
                   mode="CV-5",
                   seed=None,
                   top_n=None,
                   categories=False,
                   max_categories=None,
                   categories_topn=False,
                   sidx=0,
                   max_errors=None,
                   confusionmatrix=False,
                   cm_norm=None,
                   epochs=5,
                   batch_size=32,
                   loss="categorical_crossentropy",
                   optimizer="adam",
                   ):
    
    """
    Builds a classification or regression model and evaluates it.

    Args:
        model: Scikit-learn or Keras classifier/regressor, for example RandomForestClassifier()
        session: Session object (created in load_data())
        reload (bool): Set to True of model shall be rebuilt. If False, previously built model will be used (if any) (default: False) 
        mode (str): Set evaluation mode: 'split' uses train-test split (see split_data()), 'all' trains and evaluates the model on all data and 'CV-n' uses n-fold cross validation (default: 'CV-5')
        seed (int or None): Seed value to be used by the randomizer. If None, no seed will be used and results can differ between runs (default: None)
        top_n (int or None): Set if calculating metrics for top n results instead of only the top result. If None, metrics will only be calculated for the top result (default: None)
        categories (bool): True if table with metrics per category shall be shown (default: False)
        max_categories (int or None): Set to limit the number of categories to be shown in the categories table. If None, all categories are shown (default: None)
        categories_topn (bool): True for using top n predictions for the categories table (default: False)
        sidx (int): When limiting the number of categories to be shown in the categories table, sidx specifies the start index of the first category in the table (default: 0)
        max_errors (int or None): Set to limit the number of errors to be shown for each category in the categories table. If None, all errors are shown (default: None)
        confusionmatrix (bool): True of confusion matrix shall be shown (default: False)
        cm_norm (str or None): Normalization mode for the confusion matrix ('true', 'pred', 'all' or None) (default: None)
        epochs (int): Number of epochs to be used for Keras models (default: 5)
        batch_size (int): Batch size to be used for Keras models (default: 32)
        loss (str): Loss function to be used for Keras models (default: 'categorical_crossentropy')
        optimizer (str or object): Optimizer to be used for Keras models (default: 'adam')
    """
    
    # Check params
    if not check_param(session, "session", [dict], expr=session is not None, expr_msg="session is None"): return
    if model is None:
        error("Model is None")
        return
    if "sklearn." not in str(type(model)) and "keras." not in str(type(model)):
        error("Unsupported model type. Only Scikit-learn and Keras models are supported")
        return
    if not check_param(mode, "mode", [str], expr=mode=="all" or mode=="split" or mode.startswith("CV-"), expr_msg="mode must be split, CV-# or all"): return
    if not check_param(seed, "seed", [int,None], expr=seed is None or seed>=0, expr_msg="seed cannot be negative"): return
    if not check_param(top_n, "top_n", [int,None], expr=top_n is None or top_n>=2, expr_msg="top n must be 2 or higher"): return
    if not check_param(categories, "categories", [bool]): return
    if not check_param(categories_topn, "categories_topn", [bool]): return
    if not check_param(max_categories, "max_categories", [int,None], expr=max_categories is None or max_categories>=1, expr_msg="max categories must be at least 1"): return
    if not check_param(max_errors, "max_errors", [int,None], expr=max_errors is None or max_errors>=1, expr_msg="max errors must be at least 1"): return
    if not check_param(sidx, "sidx", [int], expr=sidx>=0, expr_msg="sidx cannot be negative"): return
    if not check_param(confusionmatrix, "confusionmatrix", [bool]): return
    if not check_param(cm_norm, "cm_norm", [str,None], ["true","pred","all"]): return
    if not check_param(epochs, "epochs", [int], expr=epochs>=1, expr_msg="epochs must be at least 1"): return
    if not check_param(batch_size, "batch_size", [int], expr=batch_size>=1, expr_msg="batch size must be at least 1"): return
    
    # Check if we have a Keras model
    if "keras." in str(type(model)):
        model = KerasWrapper(model, epochs=epochs, batch_size=batch_size, loss=loss, optimizer=optimizer)
        if model.nout > 1 and model.nout != session["categories"]:
            error("Keras model outputs " + colored(f"{model.nout}", "blue") + " does not match " + colored(f"{session['categories']}", "blue") + " categories")
            return
        
    # Check if rebuild model
    if "eval_mode" in session and mode != session["eval_mode"]:
        reload = True
    if "modelid" in session and session["modelid"] != str(model):
        reload = True
    
    # Build model and predict data (if not already built)
    if "y_pred" not in session or reload:
        #
        # Cross-validation
        #
        if mode.lower().startswith("cv"):
            st = time.time()
            cv = 5
            if len(mode) > 2:
                if "-" in mode: 
                    cv = int(mode.split("-")[1])
                elif " " in mode:
                    cv = int(mode.split(" ")[1])
                else:
                    error("Cross validation mode must be " + colored("CV", "cyan") + ", " + colored("CV-#", "cyan") + " or " + colored("CV #", "cyan"))
                    return
                
            # Clones a model
            def cloner(_model):
                if "KerasWrapper" in str(type(_model)):
                    _model.clone()
                    return _model
                else:
                    return clone(_model, safe=True)
            
            # Get folds
            cvm = KFold(n_splits=cv, shuffle=False)
                
            # Run cross validation
            y_pred = []
            y_pred_topn = []
            y_actual = []
            for tf_idx, val_idx in cvm.split(session["X"], session["y"]):
                if type(session["X"]) == list:
                    X_train = [session["X"][i] for i in tf_idx]
                    X_test = [session["X"][i] for i in val_idx]
                else:
                    X_train, X_test = session["X"][tf_idx], session["X"][val_idx]
                    
                if type(session["y"]) == list:
                    y_train = [session["y"][i] for i in tf_idx]
                    y_test = [session["y"][i] for i in val_idx]
                else:
                    y_train, y_test = session["y"][tf_idx], session["y"][val_idx]
                # Resample
                if "resample" in session:
                    X_train, y_train = resample(session, X_train, y_train) 
                # Build model
                model_obj = cloner(model)
                model_obj.fit(X_train, y_train)
                y_pred += list(model_obj.predict(X_test))
                y_actual += list(y_test)
                
                # Top n result
                if top_n is not None:
                    if hasattr(model, "predict_proba"):
                        model_ccv = model_obj
                    else:
                        model_ccv = CalibratedClassifierCV(model_obj, cv="prefit").fit(X_train, y_train)
                    probs = model_ccv.predict_proba(X_test)
                    for py,yi in zip(probs, y_test):
                        tpreds = [[p,l] for p,l in zip(py,model_ccv.classes_)]
                        tpreds = sorted(tpreds, key=lambda x: x[0], reverse=True)
                        tpreds = [l[1] for l in tpreds[:top_n]] 
                        if yi in tpreds:
                            y_pred_topn.append(yi)
                        else:
                            y_pred_topn.append(tpreds[0])
            
            session["y_pred"] = y_pred
            session["y_actual"] = y_actual
            if top_n is not None:
                session["y_pred_topn"] = y_pred_topn
                session["top_n"] = top_n
            
            en = time.time()
            info(f"Building and evaluating model using {cv}-fold cross validaton took " + colored(f"{en-st:.2f}", "blue") + " sec")
            
        #
        # Train-test split
        #
        elif mode in ["train-test", "split"]:
            st = time.time()
            if "X_train" not in session or "y_train" not in session:
                error("Data must be split using function " + colored("split_data()", "cyan") + " before evaluating model using train-test split")
                return
            X_train = session["X_train"]
            y_train = session["y_train"]
            # Resample
            if "resample" in session:
                X_train, y_train = resample(session, X_train, y_train)
            model.fit(X_train, y_train)
            session["y_pred"] = model.predict(session["X_test"])
            session["y_actual"] = session["y_test"]
            
            # Top n result
            if top_n is not None:
                y_pred_topn = []
                if hasattr(model, "predict_proba"):
                    model_ccv = model
                else:
                    model_ccv = CalibratedClassifierCV(model, cv="prefit").fit(X_train, y_train)
                probs = model_ccv.predict_proba(session["X_test"])
                for py,yi in zip(probs, session["y_test"]):
                    tpreds = [[p,l] for p,l in zip(py,model_ccv.classes_)]
                    tpreds = sorted(tpreds, key=lambda x: x[0], reverse=True)
                    tpreds = [l[1] for l in tpreds[:top_n]] 
                    if yi in tpreds:
                        y_pred_topn.append(yi)
                    else:
                        y_pred_topn.append(tpreds[0])
                session["y_pred_topn"] = y_pred_topn
                session["top_n"] = top_n
                
            en = time.time()
            mode = "split"
            info("Building and evaluating model using train-test split took " + colored(f"{en-st:.2f}", "blue") + " sec")
            
        #
        # All data
        #
        elif mode.lower() in ["all", ""]:
            st = time.time()
            X = session["X"]
            y = session["y"]
            # Resample
            if "resample" in session:
                warning("Resampling when using all data for both training and testing can give incorrect accuracy")
                X, y = resample(session, X, y)
            model.fit(X, y)
            session["y_pred"] = model.predict(X)
            session["y_actual"] = y
            
            # Top n result
            if top_n is not None:
                y_pred_topn = []
                if hasattr(model, "predict_proba"):
                    model_ccv = model
                else:
                    model_ccv = CalibratedClassifierCV(model, cv="prefit").fit(X, y)
                probs = model_ccv.predict_proba(session["X"])
                for py,yi in zip(probs, session["y"]):
                    tpreds = [[p,l] for p,l in zip(py,model_ccv.classes_)]
                    tpreds = sorted(tpreds, key=lambda x: x[0], reverse=True)
                    tpreds = [l[1] for l in tpreds[:top_n]] 
                    if yi in tpreds:
                        y_pred_topn.append(yi)
                    else:
                        y_pred_topn.append(tpreds[0])
                session["y_pred_topn"] = y_pred_topn
                session["top_n"] = top_n
            
            en = time.time()
            mode = "all"
            info("Building and evaluating model on all data took " + colored(f"{en-st:.2f}", "blue") + " sec")
        else:
            warning("Invalid mode " + colored(mode, "cyan"))
            return
            
        session["eval_mode"] = mode
        session["modelid"] = str(model)
    
    # Error check
    if session["y_pred"] is None:
        error("No predictions was made. Make sure your model works correctly")
        session["mode"] = ""
        session["modelid"] = ""
        return
    
    # Results (regression)
    if session["mode"] == "regression":
        t = CustomizedTable(["Results", ""])
        t.column_style(1, {"color": "value", "num-format": "int-2"})
        t.add_row(["R^2 score:", float(r2_score(session["y_actual"], session["y_pred"]))])
        t.add_row(["Mean Absolute Error (MAE):", float(mean_absolute_error(session["y_actual"], session["y_pred"]))])
        t.add_row(["Root Mean Squared Error (RMSE):", float(mean_squared_error(session["y_actual"], session["y_pred"]))])
        print()
        t.display()
        
    # Results (classification)
    else:
        t = CustomizedTable(["Results", ""])
        t.column_style(1, {"color": "percent", "num-format": "pct-2"})
        t.add_row(["Accuracy:", float(accuracy_score(session["y_actual"], session["y_pred"]))])
        t.add_row(["F1-score:", float(f1_score(session["y_actual"], session["y_pred"], average="weighted"))])
        t.add_row(["Precision:", float(precision_score(session["y_actual"], session["y_pred"], average="weighted", zero_division=False))])
        t.add_row(["Recall:", float(recall_score(session["y_actual"], session["y_pred"], average="weighted", zero_division=False))])
        if "y_pred_topn" in session and top_n is not None:
            t.add_row([f"Accuracy (top {session['top_n']}):", float(accuracy_score(session["y_actual"], session["y_pred_topn"]))])
            t.add_row([f"F1-score (top {session['top_n']}):", float(f1_score(session["y_actual"], session["y_pred_topn"], average="weighted"))])
        print()
        t.display()
        
        # Results per category
        if categories:
            # Generate sorted list of category results
            cats = np.unique(session["y_actual"])
            if categories_topn and top_n is not None:
                cm = confusion_matrix(session["y_actual"], session["y_pred_topn"])
            else:
                cm = confusion_matrix(session["y_actual"], session["y_pred"])
            tmp = []
            for i,cat,r in zip(range(0,len(cats)),cats,cm):
                # Generate errors
                errs = []
                for j in range(0,len(r)):
                    if i != j and r[j] > 0:
                        errs.append([r[j], cats[j]])
                tmp.append([r[i]/sum(r),cat,r[i],sum(r),errs])
            tmp = sorted(tmp, reverse=True)
            # Show table
            if "descriptions" not in session:
                t = CustomizedTable(["Category", "Accuracy", "Correct", "n"], style={"row-toggle-background": 0})
            else:
                t = CustomizedTable(["Category", "Accuracy", "Correct", "n", "Description"], style={"row-toggle-background": 0})
                t.column_style("Description", {"color": "#666"}) # 05760f
            t.column_style(0, {"color": "#048512"})
            t.column_style(1, {"color": "percent", "num-format": "pct-2"})
            t.column_style([2,3], {"color": "value"})
            if max_categories in [-1,0,None]:
                max_categories = len(tmp)
            for r in tmp[sidx:sidx+max_categories]:
                cat = r[1]
                if "label_encoder" in session:
                    l = session["label_encoder"].inverse_transform([cat])[0]
                    cat = f"{l} ({cat})"
                row = [cat, float(r[0]), r[2], r[3]]
                if "descriptions" in session:
                    if r[1] in session["descriptions"]:
                        row.append(session["descriptions"][r[1]])
                    else:
                        row.append("")
                t.add_row(row, style={"border": "top", "background": "#eee"})
                if len(r[4]) > 0:
                    errs = sorted(r[4], reverse=True)
                    if max_errors in [-1,0,None]:
                        max_errors = len(errs)
                    errs = errs[:max_errors]
                    for err in errs:
                        ecat = err[1]
                        if "label_encoder" in session:
                            l = session["label_encoder"].inverse_transform([ecat])[0]
                            ecat = f"{l} ({ecat})"
                        erow = [f"&nbsp;&nbsp;{ecat}", float(err[0]/r[2]), err[0], ""]
                        if "descriptions" in session:
                            if err[1] in session["descriptions"]:
                                erow.append(session["descriptions"][err[1]])
                            else:
                                erow.append("")
                        t.add_row(erow)
                        if "descriptions" in session:
                            t.cell_style(4,-1, {"color": "#666"}) # fb6d6d
                        t.cell_style(0,-1, {"color": "#fd8e8a"})
                        t.cell_style([1,2],-1, {"color": "#aaa4fa"})
            print()
            t.display()

        # Confusion matrix
        if confusionmatrix:
            print()
            labels = None
            if "label_encoder" in session:
                labels = []
                for cat in cats:
                    l = session["label_encoder"].inverse_transform([cat])[0]
                    labels.append(f"{l} ({cat})")
            ConfusionMatrixDisplay.from_predictions(session["y_actual"], session["y_pred"], normalize=cm_norm, xticks_rotation="vertical", cmap="inferno", values_format=".2f", colorbar=False, display_labels=labels)
            plt.show()
    
    print()


def build_model(model, 
                session, 
                mode="all",
                epochs=5,
                batch_size=32,
                loss="categorical_crossentropy",
                optimizer="adam",
                seed=None,
               ):
    """
    Builds final classification or regression model.

    Args:
        model: Scikit-learn or Keras classifier/regressor, for example RandomForestClassifier()
        session: Session object (created in load_data())
        mode (str): Sets if final model shall be built on training data ('split') or all data ('all') (default: 'all')
        seed (int or None): Seed value to be used by the randomizer. If None, no seed will be used and results can differ between runs (default: None)
        epochs (int): Number of epochs to be used for Keras models (default: 5)
        batch_size (int): Batch size to be used for Keras models (default: 32)
        loss (str): Loss function to be used for Keras models (default: 'categorical_crossentropy')
        optimizer (str or object): Optimizer to be used for Keras models (default: 'adam')
    """
    
    # Check params
    if not check_param(session, "session", [dict], expr=session is not None, expr_msg="session is None"): return
    if model is None:
        error("Model is None")
        return
    if "sklearn." not in str(type(model)) and "keras." not in str(type(model)):
        error("Unsupported model type. Only Scikit-learn and Keras models are supported")
        return
    if not check_param(mode, "mode", [str], vals=["all", "split"]): return
    if not check_param(seed, "seed", [int,None], expr=seed is None or seed>=0, expr_msg="seed cannot be negative"): return
    if not check_param(epochs, "epochs", [int], expr=epochs>=1, expr_msg="epochs must be at least 1"): return
    if not check_param(batch_size, "batch_size", [int], expr=batch_size>=1, expr_msg="batch size must be at least 1"): return
        
    # Check if we have a Keras model
    if "keras." in str(type(model)):
        model = KerasWrapper(model, epochs=epochs, batch_size=batch_size, loss=loss, optimizer=optimizer)
    
    if mode in ["train-test", "split"]:
        if "X_train" not in session or "y_train" not in session:
            error("Building final model with mode " + colored("split", "cyan") + " requires splitting data with " + colored("split_data()", "cyan"))
            return
        st = time.time()
        X = session["X_train"]
        y = session["y_train"]
        # Resample
        if "resample" in session:
            X, y = resample(session, X, y)
        model.fit(X, y)
        y_pred = model.predict(X)
        session["model"] = model
        en = time.time()
        if session["mode"] == "regression":
            info("Building final model on training data took " + colored(f"{en-st:.2f}", "blue") + " sec (MAE " + colored(f"{float(mean_absolute_error(y, y_pred)):.2f}", "blue") + ")")
        else:
            info("Building final model on training data took " + colored(f"{en-st:.2f}", "blue") + " sec (accuracy " + colored(f"{float(accuracy_score(y, y_pred))*100:.2f}%", "blue") + ")")
    elif mode in ["all", ""]:
        st = time.time()
        X = session["X"]
        y = session["y"]
        # Resample
        if "resample" in session:
            X, y = resample(session, X, y)
        model.fit(X, y)
        y_pred = model.predict(X)
        session["model"] = model
        en = time.time()
        if session["mode"] == "regression":
            info("Building final model on all data took " + colored(f"{en-st:.2f}", "blue") + " sec (MAE " + colored(f"{float(mean_absolute_error(y, y_pred)):.2f}", "blue") + ")")
        else:
            info("Building final model on all data took " + colored(f"{en-st:.2f}", "blue") + " sec (accuracy " + colored(f"{float(accuracy_score(y, y_pred))*100:.2f}%", "blue") + ")")
    else:
        error("Invalid mode " + colored(mode, "cyan"))


def save_session(session, sid, verbose=1):
    """
    Saves a session to file.

    Args:
        session: Session object (created in load_data())
        sid (int or str): Session id to store the session in
        verbose (int): Set verbose (output messages) level (0 for no output messages) (default: 1)
    """
    
    # Check params
    if not check_param(session, "session", [dict], expr=session is not None, expr_msg="session is None"): return
    
    # Check if path exists
    fpath = "sessions"
    if not exists(fpath):
        mkdir(fpath)
    
    # Date-time
    session["created"] = timestamp_to_str(None)
    
    # Dump to file
    file = f"sessions/{sid}.gz"
    dump(session, gzip.open(file, "wb"))
    if verbose >= 1:
        info("Session saved to " + colored(file, "cyan"))


def load_session(sid, verbose=1):
    """
    Loads a session from file.

    Args:
        sid (int or str): Session id the session is stored in
        verbose (int): Set verbose (output messages) level (0 for no output messages) (default: 1)
        
    Returns:
        session: Session object
    """
    
    file = f"sessions/{sid}.gz"
    if not exists(file) and not file.endswith(".gz"):
        file += ".gz"
    if not exists(file):
        error("File " + colored(file, "cyan") + " not found")
        return None
    # Load file
    s = load(gzip.open(file, "rb"))
    if verbose >= 1:
        info("Session loaded from " + colored(file, "cyan") + " (created at " + colored(s["created"], "blue") + " from file " + colored(s["file"], "cyan") + ")")
    return s


def prediction_errors_for_category(session, category, predicted_category=None, sidx=0, n=5):
    """
    Dumps n prediction errors. TBD.

    Args:
        session: Session object (created in load_data())
    """
    
    # Check params
    if not check_param(session, "session", [dict], expr=session is not None, expr_msg="session is None"): return
    
    # Check if model has been built
    if "model" not in session:
        error("Final model has not been built. Use the function " + colored("build_model()", "cyan"))
        return
    
    # Find n errors
    ht = f"Actual: <id>{category}</>"
    t = CustomizedTable(["Predicted", tag_text(ht)])
    t.column_style(1, {"color": "#e65205"})
    cidx = 0
    for xi_raw,xi,yi in zip(session["X_original"], session["X"], session["y"]):
        if yi == category:
            y_pred = session["model"].predict(xi)[0]
            if y_pred != yi and (predicted_category is None or predicted_category == y_pred):
                if cidx >= sidx and t.no_rows() < n:
                    t.add_row([y_pred, xi_raw])
                cidx += 1
    if predicted_category is None:
        t.add_subheader(["", tag_text(f"Found {cidx} prediction errors for <id>{category}</>")])
    else:
        t.add_subheader(["", tag_text(f"Found {cidx} prediction errors for <id>{category}</> where predicted category is <id>{predicted_category}</>")])
    
    t.display()
    

def errors_for_predicted_category(session, category, n=None):
    """
    Check actual categories for prediction errors where predicted category matches the specified category. TBD.

    Args:
        session: Session object (created in load_data())
    """
    
    # Check params
    if not check_param(session, "session", [dict], expr=session is not None, expr_msg="session is None"): return
    # Check if model has been built
    if "model" not in session:
        error("Final model has not been built. Use the function " + colored("build_model()", "cyan"))
        return
    # Check if valid category
    if category not in set(session["y"]):
        error("Category " + colored(category, "cyan") + " is not a valid category for the dataset")
        return
    
    # Get test data
    if "X_test" not in session or "y_test" not in session:
        y_preds = session["model"].predict(session["X"])
        y = session["y"]
    else:
        y_preds = session["model"].predict(session["X_test"])
        y = session["y_test"]
    
    # Find errors where predictions match specified account
    cnt = 0
    tot = 0
    inf = {}
    for ypi,yi in zip(y_preds,y):
        if ypi != yi and ypi == category:
            cnt += 1
            if yi not in inf:
                inf.update({yi: 0})
            inf[yi] += 1
        if ypi != yi:
            tot += 1
            
    # Check if we have found errors
    if tot == 0:
        info("No prediction errors were found for category " + colored(category, "cyan"))
        return
    
    # Sort results
    linf = []
    for acc,no in inf.items():
        linf.append([no,acc])
    linf = sorted(linf, reverse=True)
    
    # Result table
    ht = f"Predicted as <id>{category}</>"
    t = CustomizedTable(["Actual", "Errors", tag_text(f"Part of <id>{category}</> errs"), "Part of all errs"])
    t.column_style(0, {"color": "id"})
    t.column_style(1, {"color": "value"})
    t.column_style(2, {"color": "percent"})
    t.column_style(3, {"color": "percent"})
    
    if n is not None:
        linf = linf[:n]
    for e in linf:
        t.add_row([e[1], e[0], f"{e[0]/cnt*100:.1f}%", f"{e[0]/tot*100:.1f}%"])
    
    t.add_subheader(["Total:", cnt, "", tag_text(f"(<percent>{cnt/tot*100:.1f}%</> of all <value>{tot}</> errors are predicted as <id>{category}</>)")])
    t.cell_style(0, -1, {"font": "bold"})
    t.cell_style(1, -1, {"color": "value"})
    t.display()
    

def predict(xi, session):
    """
    Predics an example using the final model.

    Args:
        xi: The example to predict
        session: Session object (created in load_data())
        
    Returns:
        Predicted value for the example
    """
    
    # Check params
    if not check_param(session, "session", [dict], expr=session is not None, expr_msg="session is None"): return
    if xi is None:
        error("example is empty")
        return
    
    # Check if model has been built
    if "model" not in session:
        error("Final model has not been built. Use the function " + colored("build_model()", "cyan"))
        return
    
    # Error checks
    if type(xi) == str and session["preprocess"] not in ["bag-of-words", "word2vec", "embeddings"]:
        error("Example is text but no text preprocessing is specified")
        return
    
    # Bag of words
    if type(xi) == str and session["preprocess"] == "bag-of-words":
        X = session["bow"].transform([xi])
        if "tf-idf" in session:
            X = session["tf-idf"].transform(X)
        pred = session["model"].predict(X)
        res = pred[0]
        if "label_encoder" in session:
            res = f"{session['label_encoder'].inverse_transform([res])[0]} ({res})"
        info("Example is predicted as " + colored(res, "green"))
        return
    
    # Word2vec
    if type(xi) == str and session["preprocess"] == "word2vec":
        X = [word_vector(xi, session)]
        pred = session["model"].predict(X)
        res = pred[0]
        if "label_encoder" in session:
            res = f"{session['label_encoder'].inverse_transform([res])[0]} ({res})"
        info("Example is predicted as " + colored(res, "green"))
        return
    
    # Embeddings
    if type(xi) == str and session["preprocess"] == "embeddings":
        X = embedding(xi, session)
        pred = session["model"].predict(X)
        res = pred[0]
        if "label_encoder" in session:
            res = f"{session['label_encoder'].inverse_transform([res])[0]} ({res})"
        info("Example is predicted as " + colored(res, "green"))
        return
    
    # Numerical/ordinal data
    if "scaler" in session:
        X = session["scaler"].transform([xi])
        pred = session["model"].predict(X)
        res = pred[0]
        if "label_encoder" in session:
            res = f"{session['label_encoder'].inverse_transform([res])[0]} ({res})"
        if session["mode"] == "regression" and type(res) in [float, np.float64]:
            if not float(res).is_integer():
                res = round(res, 2)
        info("Example is predicted as " + colored(res, "green"))
        return
    
    # No pre-processing
    pred = session["model"].predict([xi])
    res = pred[0]
    if "label_encoder" in session:
        res = f"{session['label_encoder'].inverse_transform([res])[0]} ({res})"
    if session["mode"] == "regression" and type(res) in [float, np.float64]:
        if not float(res).is_integer():
            res = round(res, 2)
    info("Example is predicted as " + colored(res, "green"))
    