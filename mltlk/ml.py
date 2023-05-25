# Basic stuff
from termcolor import colored
import numpy as np
import pandas as pd
from collections import Counter
from customized_table import *
import time
import matplotlib.pyplot as plt
import re
# Pre-processing
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split
from nltk.corpus import stopwords
# Classifiers
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
# Evaluation
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, confusion_matrix, ConfusionMatrixDisplay
# File stuff
from pickle import dump,load
from os.path import exists
from os import makedirs
import gzip
# Resampling
from .resampling import resample
from .word2vec import *


#
# Error message
#
def error(e):
    print(colored("Error: ", "red", attrs=["bold"]) + e)

    
#
# Warning message
#
def warning(e):
    print(colored("Warning: ", "red", attrs=["bold"]) + e)
    

#
# Info message
#
def info(e):
    print(colored("Info: ", "yellow", attrs=["bold"]) + e)


#
# Load and pre-process data
#
def load_data(file, Xcols=[0], ycol=1, verbose=1, conf={}):
    session = {}
    
    # Load data
    data = pd.read_csv(file).values
    session["file"] = file
    
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
    session["X_original"] = X
    session["y_original"] = y
    session["X"] = X.copy()
    session["y"] = y.copy()
    
    # Skip minority categories
    if "min_samples" in conf:
        if type(conf["min_samples"]) != int:
            error("min_samples must be integer")
            return
        cnt = Counter(session["y"])
        X = []
        y = []
        for xi,yi in zip(session["X"], session["y"]):
            if cnt[yi] >= conf["min_samples"]:
                X.append(xi)
                y.append(yi)
        session["X_original"] = X
        session["y_original"] = y
        session["X"] = X.copy()
        session["y"] = y.copy()
        if verbose >= 1:
            s = ""
            for li,ni in cnt.items():
                if ni < conf["min_samples"]:
                    s += li + ", "
            if s != "":
                info("Removed minority categories " + colored(s[:-2], "cyan"))
                
    # Clean inputs
    if "clean_text" in conf:
        if conf["clean_text"] == 2 or conf["clean_text"] == "digits":
            info("Clean texts keeping letters and digits")
        else:
            info("Clean texts keeping letters only")    
        for i,xi in enumerate(session["X"]):
            # Remove new line and whitespaces
            xi = xi.replace("<br>", " ")
            xi = xi.replace("&nbsp;", " ")
            # Remove special chars
            if conf["clean_text"] == 2 or conf["clean_text"] == "digits":
                xi = re.sub("[^a-zA-Z0-9åäöÅÄÖ ]", " ", xi)
            else:
                xi = re.sub("[^a-zA-ZåäöÅÄÖ ]", " ", xi)
            # Remove multiple whitespaces
            xi = " ".join(xi.split())
            # Set to lower case
            xi = xi.lower()
            # Strip trailing/leading whitespaces
            xi = xi.strip()
            session["X"][i] = xi
            
    # Encode labels
    if "encode_labels" in conf and conf["encode_labels"]:
        session["label_encoder"] = LabelEncoder().fit(session["y"])
        session["y"] = session["label_encoder"].transform(session["y"])
        if verbose >= 1:
            info("Labels encoded")
    
    # Bag-of-words representation for input texts
    if "bag-of-words" in conf and conf["bag-of-words"]:
        if "stopwords" in conf:
            sw = list(stopwords.words(conf["stopwords"]))
            if verbose >= 1:
                info("Used bag-of-words with stopwords " + colored(conf["stopwords"], "cyan") + " removed")
        else:
            sw = None
            if verbose >= 1:
                info("Used bag-of-words")
        session["bow"] = CountVectorizer(stop_words=sw).fit(session["X"]) #todo: max_features=max_words, ngram_range=ngram)
        session["X"] = session["bow"].transform(session["X"])
    
        # TF-IDF conversion for bag-of-words
        if "TF-IDF" in conf and conf["TF-IDF"] or "tf-idf" in conf and conf["tf-idf"]:
            session["TF-IDF"] = TfidfTransformer().fit(session["X"])
            session["X"] = session["TF-IDF"].transform(session["X"])
            if verbose >= 1:
                info("Used TF-IDF")
    # Word2vec
    if "word2vec" in conf and conf["word2vec"]:
        load_word2vec_data(session, conf, verbose=verbose)
        
    if verbose >= 1:
        info("Loaded " + colored(f"{len(session['y'])}", "blue") + " examples in " + colored(f"{len(Counter(session['y']))}", "blue") + " categories")
    
    return session


#
# Show data stats
#
def data_stats(session, max_rows=None, show_graph=False, descriptions=None):
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
        t = CustomizedTable(["Acc", "No", "%", "Σ%", "Description", "Acc", "No", "%", "Σ%", "Description", "Acc", "No", "%", "Σ%", "Description"])
        t.column_style([0,5,10], {"color": "id"})
        t.column_style([1,6,11], {"color": "value"})
        t.column_style([2,7,12], {"color": "percent"})
        t.column_style([3,8,13], {"color": "green"})
        t.column_style([4,9,14], {"color": "name"})
    else:
        t = CustomizedTable(["Acc", "No", "%", "Σ%", "Acc", "No", "%", "Σ%", "Acc", "No", "%", "Σ%"])
        t.column_style([0,4,8], {"color": "id"})
        t.column_style([1,5,9], {"color": "value"})
        t.column_style([2,6,10], {"color": "percent"})
        t.column_style([3,7,11], {"color": "green"})
    for i in range(0,s):
        r = []
        for j in range(0,3):
            if i < len(tab2[j]):
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
            for i in range(rsize - len(r)):
                r.append("")
        t.add_row(r)
    t.display()


#
# Split data into train and test sets
#
def split_data(session, verbose=1, conf={}):
    # Test size
    test_size = 0.15
    if "test_size" in conf:
        test_size = conf["test_size"]
    s = "Split data using " + colored(f"{(1-test_size)*100:.0f}%", "blue") + " training data and " + colored(f"{(test_size)*100:.0f}%", "blue") + " test data"
    
    # Random seed
    seed = None
    if "seed" in conf:
        seed = conf["seed"]
        s += " with seed " + colored(seed, "blue") 
        
    # Stratify
    stratify = None
    if "stratify" in conf and conf["stratify"]:
        stratify = session["y"]
        s += " and stratify"
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(session["X"], session["y"], test_size=test_size, random_state=seed, stratify=stratify)
    
    # Update session
    session["X_train"] = X_train
    session["X_test"] = X_test
    session["y_train"] = y_train
    session["y_test"] = y_test
    
    if verbose >= 1:
        info(s)


#
# Sets resampling method to use
#
def set_resample(session, conf={}):
    # Check mode parameter
    if "mode" not in conf:
        error("Missing parameter " + colored("mode", "cyan"))
        return
    if type(conf["mode"]) != str:
        error("Resample mode must be a string")
        return
    
    conf["mode"] = conf["mode"].lower()
    for mode in list(conf["mode"]):
        if mode not in ["o","u","s"]:
            error("Unsupported resample mode (must be " + colored("o", "cyan") + ", " + colored("u", "cyan") + " or " + colored("s", "cyan") + ")")
            return
    if len(conf["mode"]) == 0:
        error("Parameter " + colored("mode", "cyan") + " is empty")
        return
    
    session["resample"] = {
        "mode": conf["mode"]
    }
    for mode in list(conf["mode"]):
        if mode == "u":
            if "max_samples" not in conf:
                warning(colored("max_samples", "cyan") + " not set (using " + colored("500", "blue") + ")")
                conf["max_samples"] = 500
            if "decrease_limit" not in conf:
                warning(colored("decrease_limit", "cyan") + " not set (using " + colored("0.5", "blue") + ")")
                conf["decrease_limit"] = 0.5
            session["resample"]["max_samples"] = conf["max_samples"]
            session["resample"]["decrease_limit"] = conf["decrease_limit"]
            info("Using random undersampling with max samples " + colored(conf["max_samples"], "blue") + " and decrease limit " + colored(conf["decrease_limit"], "blue"))
        elif mode == "o":
            if "min_samples" not in conf:
                warning(colored("min_samples", "cyan") + " not set (using " + colored("50", "blue") + ")")
                conf["min_samples"] = 50
            if "increase_limit" not in conf:
                warning(colored("increase_limit", "cyan") + " not set (using " + colored("1.0", "blue") + ")")
                conf["increase_limit"] = 1.0
            session["resample"]["min_samples"] = conf["min_samples"]
            session["resample"]["increase_limit"] = conf["increase_limit"]
            info("Using random oversampling with min samples " + colored(conf["min_samples"], "blue") + " and increase limit " + colored(conf["increase_limit"], "blue"))
        elif mode == "s":
            if "auto" in conf and conf["auto"]:
                session["resample"]["auto"] = 1
                info("Using auto SMOTE oversampling")
            else:
                if "min_samples" not in conf:
                    warning(colored("min_samples", "cyan") + " not set (using " + colored("50", "blue") + ")")
                    conf["min_samples"] = 50
                if "increase_limit" not in conf:
                    warning(colored("increase_limit", "cyan") + " not set (using " + colored("1.0", "blue") + ")")
                    conf["increase_limit"] = 1.0
                session["resample"]["min_samples"] = conf["min_samples"]
                session["resample"]["increase_limit"] = conf["increase_limit"]
                info("Using SMOTE oversampling with min samples " + colored(conf["min_samples"], "blue") + " and increase limit " + colored(conf["increase_limit"], "blue"))
          
    if "seed" in conf:
        session["resample"]["seed"] = conf["seed"]
    else:
        session["resample"]["seed"] = None
        
    # Reset mode to rebuild model
    if "mode" in session:
        session["mode"] = ""


#
# Builds and evaluates model
#
def evaluate_model(model, session, reload=False, conf={}):
    # Check if rebuild model
    if "mode" in conf and "mode" in session and conf["mode"] != session["mode"]:
        reload = True
    if "modelid" in session and session["modelid"] != str(model):
        reload = True
    
    # Build model and predict data (if not already built)
    if "y_pred" not in session or reload:
        if "mode" in conf and (conf["mode"].startswith("CV") or conf["mode"].startswith("cv")):
            st = time.time()
            cv = 5
            if len(conf["mode"]) > 2:
                if "-" in conf["mode"]: 
                    cv = int(conf["mode"].split("-")[1])
                elif " " in conf["mode"]:
                    cv = int(conf["mode"].split(" ")[1])
                else:
                    error("Cross validation mode must be " + colored("CV", "cyan") + ", " + colored("CV-#", "cyan") + " or " + colored("CV #", "cyan"))
                    return
                
            # Run cross validation
            from sklearn.model_selection import KFold
            from sklearn.base import clone
            if "seed" in conf:
                cvm = KFold(n_splits=cv, random_state=conf["seed"], shuffle=True)
            else:
                cvm = KFold(n_splits=cv, shuffle=False)
            y_pred = []
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
                if "resample" in session:
                    X_train, y_train = resample(session, X_train, y_train) 
                # Build model    
                model_obj = clone(model, safe=True)
                model_obj.fit(X_train, y_train)
                y_pred += list(model_obj.predict(X_test))
                y_actual += list(y_test)
            session["y_pred"] = y_pred
            session["y_actual"] = y_actual
            
            en = time.time()
            print(f"Building and evaluating model using {cv}-fold cross validaton took " + colored(f"{en-st:.4f}", "blue") + " sec")
        elif "mode" in conf and (conf["mode"] == "train-test" or conf["mode"] == "split"):
            st = time.time()
            if "X_train" not in session or "y_train" not in session:
                error("Data must be split using function " + colored("split_data()", "cyan") + " before evaluating model using train-test split")
                return
            X_train = session["X_train"]
            y_train = session["y_train"]
            if "resample" in session:
                X_train, y_train = resample(session, X_train, y_train)
            model.fit(X_train, y_train)
            session["y_pred"] = model.predict(session["X_test"])
            session["y_actual"] = session["y_test"]
            en = time.time()
            print("Building and evaluating model using train-test split took " + colored(f"{en-st:.4f}", "blue") + " sec")
        else:
            st = time.time()
            model.fit(session["X"], session["y"])
            session["y_pred"] = model.predict(session["X"])
            session["y_actual"] = session["y"]
            en = time.time()
            print("Building and evaluating model on all data took " + colored(f"{en-st:.4f}", "blue") + " sec")
            conf["mode"] = "all"
            
        session["mode"] = conf["mode"]
        session["modelid"] = str(model)
    
    # Results
    t = CustomizedTable(["", ""])
    t.column_style(1, {"color": "percent", "num-format": "pct-2"})
    t.add_row(["Accuracy:", float(accuracy_score(session["y_actual"], session["y_pred"]))])
    t.add_row(["F1-score:", float(f1_score(session["y_actual"], session["y_pred"], average="weighted"))])
    t.add_row(["Precision:", float(precision_score(session["y_actual"], session["y_pred"], average="weighted", zero_division=False))])
    t.add_row(["Recall:", float(recall_score(session["y_actual"], session["y_pred"], average="weighted", zero_division=False))])
    print()
    t.display()
    
    # Results per category
    if "categories" in conf and conf["categories"]:
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
        t = CustomizedTable(["Category", "Accuracy", "n"], style={"row-toggle-background": 0})
        t.column_style(0, {"color": "#048512"})
        t.column_style(1, {"color": "percent", "num-format": "pct-2"})
        t.column_style(2, {"color": "value"})
        sidx = 0
        maxcats = len(tmp)
        if "sidx" in conf:
            sidx = conf["sidx"]
        if "max_categories" in conf:
            maxcats = conf["max_categories"]
        for r in tmp[sidx:sidx+maxcats]:
            t.add_row([r[1], float(r[0]), r[2]], style={"border": "top", "background": "#eee"})
            if len(r[3]) > 0:
                errs = sorted(r[3], reverse=True)
                if "max_errors" in conf:
                    errs = errs[:conf["max_errors"]]
                for err in errs:
                    t.add_row(["&nbsp;&nbsp;" + err[1], float(err[0]/r[2]), err[0]])
                    t.cell_style(0,-1, {"color": "#fd8e8a"})
                    t.cell_style([1,2],-1, {"color": "#aaa4fa"})
        print()
        t.display()
        
    # Confusion matrix
    if "confusion_matrix" in conf and conf["confusion_matrix"]:
        print()
        from sklearn.metrics import ConfusionMatrixDisplay
        norm = None
        if type(conf["confusion_matrix"]) == str:
            norm = conf["confusion_matrix"]
        ConfusionMatrixDisplay.from_predictions(session["y_actual"], session["y_pred"], normalize=norm, xticks_rotation="vertical", cmap="inferno", values_format=".2f", colorbar=False)
        plt.show()
    
    print()


#
# Builds final model
#
def build_model(model, session, conf={}):
    if "mode" in conf and (conf["mode"] == "train-test" or conf["mode"] == "split"):
        st = time.time()
        model.fit(session["X_train"], session["y_train"])
        session["model"] = model
        en = time.time()
        print("Building final model on training data took " + colored(f"{en-st:.4f}", "cyan") + " sec")
    else:
        st = time.time()
        model.fit(session["X"], session["y"])
        session["model"] = model
        en = time.time()
        print("Building final model on all data took " + colored(f"{en-st:.4f}", "cyan") + " sec")


#
# Save session to file
#
def save_session(session, file, verbose=1):
    # Check if path exists
    if "/" in file:
        path = file[:file.rfind("/")]
        if not exists(path):
            makedirs(path)
    if not file.endswith(".gz"):
        file += ".gz"
    # Dump to file
    dump(session, gzip.open(file, "wb"))
    if verbose >= 1:
        info("Session saved to " + colored(file, "blue"))


#
# Load session from file
#
def load_session(file, verbose=1):
    if not exists(file) and not file.endswith(".gz"):
        file += ".gz"
    if not exists(file):
        error("File " + colored(file, "cyan") + " not found")
        return None
    # Load file
    s = load(gzip.open(file, "rb"))
    if verbose >= 1:
        info("Session loaded from " + colored(file, "blue"))
    return s


#
# Dump n prediction errors
#
def prediction_errors_for_category(session, category, predicted_category=None, sidx=0, n=5):
    # Check if model has been built
    if "model" not in session:
        error("Final model has not been built. Use the function " + colored("build_model()", "cyan"))
    
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
    # Check if model has been built
    if "model" not in session:
        error("Final model has not been built. Use the function " + colored("build_model()", "cyan"))
    
    # Get test data
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
