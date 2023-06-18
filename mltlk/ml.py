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


#
# Load and pre-process data
#
def load_data(file, 
              Xcols=None, 
              ycol=None,
              mode="classification",
              preprocess=None,
              shuffle_data=False,
              seed=None,
              min_samples=None,
              encode_labels=False,
              clean_text="letters digits",
              stopwords=None,
              max_features=None,
              tf_idf=True,
              w2v_vector_size=75,
              w2v_rebuild=False,
              embeddings_size=75,
              embeddings_max_length=None,
              verbose=1):
    session = {}
    
    # Check params
    if not check_param(mode, "mode", [str], ["classification", "regression"]): return None
    session["mode"] = mode
    if not check_param(preprocess, "preprocess", [str,None], ["normalize", "scale", "one-hot", "ordinal", "bag-of-words", "word2vec", "embeddings", None]): return None
    session["preprocess"] = preprocess
    if not check_param(shuffle_data, "shuffle_data", [bool], None): return None
    if not check_param(seed, "seed", [int,None], None): return None
    if not check_param(min_samples, "min_samples", [int,None], None): return None
    if not check_param(encode_labels, "encode_labels", [bool], None): return None
    if not check_param(clean_text, "clean_text", [str,None], ["letters", "letters digits"]): return None
    if not check_param(stopwords, "stopwords", [list,None], None): return None
    if not check_param(max_features, "max_features", [int,None], None): return None
    if not check_param(tf_idf, "tf_idf", [bool], None): return None
    if not check_param(w2v_vector_size, "w2v_vector_size", [int], None): return None
    if not check_param(w2v_rebuild, "w2v_rebuild", [bool], None): return None
    if not check_param(embeddings_size, "embeddings_size", [int], None): return None
    if not check_param(embeddings_max_length, "embeddings_max_length", [int,None], None): return None
    
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
    
    # Encode labels
    if encode_labels:
        session["label_encoder"] = LabelEncoder().fit(session["y"])
        session["y"] = session["label_encoder"].transform(session["y"])
        if verbose >= 1:
            info("Labels encoded")
        
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


#
# Show data stats
#
def data_stats(session, max_rows=None, show_graph=False, descriptions=None):
    if session is None:
        error("Session is empty")
        return
    
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
    
    # Set descriptions (if found)
    if descriptions is not None:
        if type(descriptions) == dict:
            session["descriptions"] = descriptions
        else:
            warning("Invalid type for descriptions (expected " + colored("dict", "cyan") + ")")
            descriptions = None
    
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
    if descriptions is not None:
        t = CustomizedTable(["Category", "No", "%", "Σ%", "Description", "Category", "No", "%", "Σ%", "Description", "Category", "No", "%", "Σ%", "Description"])
        t.column_style([0,5,10], {"color": "id"})
        t.column_style([1,6,11], {"color": "value"})
        t.column_style([2,7,12], {"color": "percent"})
        t.column_style([3,8,13], {"color": "green"})
        t.column_style([4,9,14], {"color": "name"})
    else:
        t = CustomizedTable(["Category", "No", "%", "Σ%", "Category", "No", "%", "Σ%", "Category", "No", "%", "Σ%"])
        t.column_style([0,4,8], {"color": "id"})
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
                if descriptions is not None:
                    desc = ""
                    if tab2[j][i][0] in descriptions:
                        desc = descriptions[tab2[j][i][0]]
                    r.append(desc)
        # Fill row, if not full
        rsize = 15
        if descriptions is None:
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
    if descriptions is not None:
        t.add_row(["Examples:", len(y), "", "", "", "Features:", fts, "", "", "", "Categories:", len(cnt), "", "", ""], style={"row-toggle-background": 0, "background": "#eee", "border": "top"})
        t.cell_style([0,5,10], -1, {"font": "bold"})
    else:
        t.add_row(["Examples:", len(y), "", "", "Features:", fts, "", "", "Categories:", len(cnt), "", ""], style={"row-toggle-background": 0, "background": "#eee", "border": "top"})
        t.cell_style([0,4,8], -1, {"font": "bold"})
    
    t.display()


#
# Split data into train and test sets
#
def split_data(session,
               test_size=0.2,
               seed=None,
               stratify=False,
               verbose=1,
              ):
    if session is None:
        error("Session is empty")
        return
    
    # Check params
    if not check_param(test_size, "test_size", [float], None): return
    if not check_param(seed, "seed", [int,None], None): return
    if not check_param(stratify, "stratify", [bool], None): return

    # Check test size
    if test_size <= 0 or test_size >= 1:
        test_size = 0.2
        warning(colored("test_size", "cyan") + " must be between 0 and 1 (using " + colored("0.2", "blue") + ")")
        
    # Info string
    s = "Split data using " + colored(f"{(1-test_size)*100:.0f}%", "blue") + " training data and " + colored(f"{(test_size)*100:.0f}%", "blue") + " test data"
    
    # Random seed
    if seed is not None:
        s += " with seed " + colored(seed, "blue") 
        
    # Stratify
    if stratify:
        stratify = session["y"]
        s += " and stratify"
    else:
        stratify=None
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(session["X"], session["y"], test_size=test_size, random_state=seed, stratify=stratify)
    
    # Update session
    session["X_train"] = X_train
    session["X_test"] = X_test
    session["y_train"] = y_train
    session["y_test"] = y_test
    session["eval_mode"] = ""
    
    if verbose >= 1:
        info(s)


#
# Sets resampling method to use
#
def set_resample(session, 
                 mode="u",
                 max_samples=500,
                 decrease_limit=0.5,
                 min_samples=50,
                 increase_limit=1.0,
                 auto=False,
                 seed=None,
                ):
    if session is None:
        error("Session is empty")
        return
    
    # Check params
    if not check_param(mode, "mode", [str], None): return
    for mode in list(mode):
        if mode not in ["o","u","s"]:
            error("Unsupported resample mode (must be " + colored("o", "cyan") + ", " + colored("u", "cyan") + " or " + colored("s", "cyan") + ")")
            return
    if len(mode) == 0:
        error("Parameter " + colored("mode", "cyan") + " is empty")
        return
    if not check_param(max_samples, "max_samples", [int], None): return
    if not check_param(decrease_limit, "decrease_limit", [float,int], None): return
    if not check_param(min_samples, "max_samples", [int], None): return
    if not check_param(increase_limit, "decrease_limit", [float,int], None): return
    if not check_param(auto, "auto", [bool], None): return
    if not check_param(seed, "seed", [int,None], None): return
    
    session["resample"] = {
        "mode": mode,
        "seed": seed,
    }
    for mode in list(mode):
        if mode == "u":
            session["resample"]["max_samples"] = max_samples
            session["resample"]["decrease_limit"] = decrease_limit
            info("Using random undersampling with max samples " + colored(max_samples, "blue") + " and decrease limit " + colored(decrease_limit, "blue"))
        elif mode == "o":
            session["resample"]["min_samples"] = min_samples
            session["resample"]["increase_limit"] = increase_limit
            info("Using random oversampling with min samples " + colored(min_samples, "blue") + " and increase limit " + colored(increase_limit, "blue"))
        elif mode == "s":
            if auto:
                session["resample"]["auto"] = 1
                info("Using auto SMOTE oversampling")
            else:
                session["resample"]["min_samples"] = min_samples
                session["resample"]["increase_limit"] = increase_limit
                info("Using SMOTE oversampling with min samples " + colored(min_samples, "blue") + " and increase limit " + colored(increase_limit, "blue"))
    session["eval_mode"] = ""
        
        
# 
# Clear resample settings
#
def clear_resample(session):
    if "resample" in session:
        del session["resample"]
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
            error("Keras models require numerical categories. Set " + colored("encode_labels", "cyan") + " to " + colored("True", "blue") + " when calling " + colored("load_data()", "cyan"))
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
        

#
# Builds and evaluates model
#
def evaluate_model(model, 
                   session, 
                   reload=False, 
                   mode="all",
                   seed=None,
                   top_n=None,
                   categories=False,
                   max_categories=None,
                   sidx=0,
                   max_errors=None,
                   confusionmatrix=False,
                   cm_norm=None,
                   epochs=5,
                   batch_size=32,
                   loss="categorical_crossentropy",
                   optimizer="adam",
                   ):
    # Check params
    if session is None:
        error("Session is empty")
        return
    if model is None:
        error("Model is None")
        return
    if "sklearn." not in str(type(model)) and "keras." not in str(type(model)):
        error("Unsupported model type. Only Scikit-learn and Keras models are supported")
        return
    if not check_param(mode, "mode", [str], None): return None
    if not check_param(seed, "seed", [int,None], None): return None
    if not check_param(top_n, "top_n", [int,None], None): return None
    if not check_param(categories, "categories", [bool], None): return None
    if not check_param(max_categories, "max_categories", [int,None], None): return None
    if not check_param(max_errors, "max_errors", [int,None], None): return None
    if not check_param(sidx, "sidx", [int], None): return None
    if not check_param(confusionmatrix, "confusionmatrix", [bool], None): return None
    if not check_param(cm_norm, "cm_norm", [str,None], ["true","pred","all",None]): return None
    if not check_param(epochs, "epochs", [int], None): return None
    if not check_param(batch_size, "batch_size", [int], None): return None
    
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
            if seed is not None:
                cvm = KFold(n_splits=cv, random_state=seed, shuffle=True)
            else:
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
                    for Xi,yi in zip(X_test, y_test):
                        if hasattr(model, "predict_proba"):
                            Xi = [Xi]
                        probs = model_ccv.predict_proba(Xi)
                        best_codes = np.argsort(-probs, axis=1)[:,:top_n][0]
                        best_prob = np.sort(-probs, axis=1)[:,:top_n][0]
                        codes = model_ccv.classes_
                        ypn = False
                        for i,c in enumerate(best_codes):
                            if codes[c] == yi:
                                ypn = True
                        if ypn:
                            y_pred_topn.append(yi)
                        else:
                            y_pred_topn.append(codes[best_codes[0]])
    
            session["y_pred"] = y_pred
            session["y_actual"] = y_actual
            if top_n is not None:
                session["y_pred_topn"] = y_pred_topn
                session["top_n"] = top_n
            
            en = time.time()
            print(f"Building and evaluating model using {cv}-fold cross validaton took " + colored(f"{en-st:.2f}", "blue") + " sec")
            
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
                for Xi,yi in zip(session["X_test"], session["y_test"]):
                    probs = model_ccv.predict_proba([Xi])
                    best_codes = np.argsort(-probs, axis=1)[:,:top_n][0]
                    best_prob = np.sort(-probs, axis=1)[:,:top_n][0]
                    codes = model_ccv.classes_
                    ypn = False
                    for i,c in enumerate(best_codes):
                        if codes[c] == yi:
                            ypn = True
                    if ypn:
                        y_pred_topn.append(yi)
                    else:
                        y_pred_topn.append(codes[best_codes[0]])
                session["y_pred_topn"] = y_pred_topn
                session["top_n"] = top_n
                
            en = time.time()
            mode = "split"
            print("Building and evaluating model using train-test split took " + colored(f"{en-st:.2f}", "blue") + " sec")
            
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
                for Xi,yi in zip(session["X"], session["y"]):
                    probs = model_ccv.predict_proba([Xi])
                    best_codes = np.argsort(-probs, axis=1)[:,:top_n][0]
                    best_prob = np.sort(-probs, axis=1)[:,:top_n][0]
                    codes = model_ccv.classes_
                    ypn = False
                    for i,c in enumerate(best_codes):
                        if codes[c] == yi:
                            ypn = True
                    if ypn:
                        y_pred_topn.append(yi)
                    else:
                        y_pred_topn.append(codes[best_codes[0]])
                session["y_pred_topn"] = y_pred_topn
                session["top_n"] = top_n
            
            en = time.time()
            mode = "all"
            print("Building and evaluating model on all data took " + colored(f"{en-st:.2f}", "blue") + " sec")
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
        if "y_pred_topn" in session:
            t.add_row([f"Accuracy (top {session['top_n']}):", float(accuracy_score(session["y_actual"], session["y_pred_topn"]))])
            t.add_row([f"F1-score (top {session['top_n']}):", float(f1_score(session["y_actual"], session["y_pred_topn"], average="weighted"))])
        print()
        t.display()
        
        # Results per category
        if categories:
            # Generate sorted list of category results
            cats = np.unique(session["y_actual"])
            cm = confusion_matrix(session["y_actual"], session["y_pred"])
            tmp = []
            for i,cat,r in zip(range(0,len(cats)),cats,cm):
                # Generate errors
                errs = []
                for j in range(0,len(r)):
                    if i != j and r[j] > 0:
                        errs.append([r[j], cats[j]])
                tmp.append([r[i]/sum(r),cat,sum(r),errs])
            tmp = sorted(tmp, reverse=True)
            # Show table
            if "descriptions" not in session:
                t = CustomizedTable(["Category", "Accuracy", "n"], style={"row-toggle-background": 0})
            else:
                t = CustomizedTable(["Category", "Accuracy", "n", "Description"], style={"row-toggle-background": 0})
                t.column_style("Description", {"color": "#05760f"})
            t.column_style(0, {"color": "#048512"})
            t.column_style(1, {"color": "percent", "num-format": "pct-2"})
            t.column_style(2, {"color": "value"})
            if max_categories in [-1,0,None]:
                max_categories = len(tmp)
            for r in tmp[sidx:sidx+max_categories]:
                cat = r[1]
                if "label_encoder" in session:
                    l = session["label_encoder"].inverse_transform([cat])[0]
                    cat = f"{l} ({cat})"
                row = [cat, float(r[0]), r[2]]
                if "descriptions" in session:
                    row.append(session["descriptions"][r[1]])
                t.add_row(row, style={"border": "top", "background": "#eee"})
                if len(r[3]) > 0:
                    errs = sorted(r[3], reverse=True)
                    if max_errors in [-1,0,None]:
                        max_errors = len(errs)
                    errs = errs[:max_errors]
                    for err in errs:
                        ecat = err[1]
                        if "label_encoder" in session:
                            l = session["label_encoder"].inverse_transform([ecat])[0]
                            ecat = f"{l} ({ecat})"
                        erow = [f"&nbsp;&nbsp;{ecat}", float(err[0]/r[2]), err[0]]
                        if "descriptions" in session:
                            erow.append(session["descriptions"][err[1]])
                        t.add_row(erow)
                        if "descriptions" in session:
                            t.cell_style(3,-1, {"color": "#fb6d6d"})
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


#
# Builds final model
#
def build_model(model, 
                session, 
                mode="all",
                epochs=5,
                batch_size=32,
                loss="categorical_crossentropy",
                optimizer="adam",
                seed=None,
               ):
    if session is None:
        error("Session is empty")
        return
    if model is None:
        error("Model is None")
        return
    if "sklearn." not in str(type(model)) and "keras." not in str(type(model)):
        error("Unsupported model type. Only Scikit-learn and Keras models are supported")
        return
    if not check_param(mode, "mode", [str], None): return None
    if not check_param(seed, "seed", [int,None], None): return None
    if not check_param(epochs, "epochs", [int], None): return None
    if not check_param(batch_size, "batch_size", [int], None): return None
        
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


#
# Save session to file
#
def save_session(session, id, verbose=1):
    if session is None:
        error("Session is empty")
        return
    
    # Check if path exists
    fpath = "sessions"
    if not exists(fpath):
        mkdir(fpath)
    
    # Date-time
    session["created"] = timestamp_to_str(None)
    
    # Dump to file
    file = f"sessions/{id}.gz"
    dump(session, gzip.open(file, "wb"))
    if verbose >= 1:
        info("Session saved to " + colored(file, "cyan"))


#
# Load session from file
#
def load_session(id, verbose=1):
    file = f"sessions/{id}.gz"
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


#
# Dump n prediction errors
#
def prediction_errors_for_category(session, category, predicted_category=None, sidx=0, n=5):
    if session is None:
        error("Session is empty")
        return
    
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
    

#
# Check actual categories for prediction errors where predicted category is specified as param
#
def errors_for_predicted_category(session, category, n=None):
    if session is None:
        error("Session is empty")
        return 
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
    
    
#
# Predict example
#
def predict(xi, session):
    if session is None:
        error("Session is empty")
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
    
