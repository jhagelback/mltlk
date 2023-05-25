# Basic stuff
from termcolor import colored
import time
# Pre-processing
from nltk.corpus import stopwords
# Word2Vec
from gensim.models import Word2Vec
# File stuff
from pickle import dump,load
from os.path import exists
from os import mkdir
import gzip


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
# Load or build Word2vec model
#
def load_word2vec_model(session, conf, verbose=1):
    # Check if path exists
    fpath = "word2vec"
    if not exists(fpath):
        mkdir(fpath)

    # Check settings
    if "w2v_vector_size" not in conf:
        warning(colored("w2v_vector_size", "cyan") + " not set (using " + colored("75", "blue") + ")")
        conf["w2v_vector_size"] = 75
    if "rebuild" not in conf:
        conf["rebuild"] = False
        
    # Filename
    fname = session["file"]
    fname = fname[fname.rfind("/")+1:]
    fname = fname.replace(".csv","").replace(".gz","") + f"_{conf['w2v_vector_size']}.w2v"
    
    # Check if stored
    if exists(f"word2vec/{fname}") and not conf["rebuild"]:
        wtovec = load(gzip.open(f"word2vec/{fname}", "rb"))
        if verbose >= 1:
            info("Word2vec model loaded from " + colored(f"word2vec/{fname}", "cyan"))
        return wtovec
    
    # Stopwords
    if "stopwords" in conf:
        stpwrds = set(stopwords.words(conf["stopwords"]))
    
    # Convert to list of list of words
    X = []
    for wds in session["X"]:
        xi = []
        for w in wds.split(" "):
            if w not in stpwrds and w.strip() != "":
                xi.append(w)
        X.append(xi)
        
    # Update data
    session["X"] = X
    
    # Train Word2Vec model
    start = time.time()
    model = Word2Vec(X, vector_size=conf["w2v_vector_size"], min_count=1)

    # Generate dict for each word
    vectors = model.wv
    words = list(model.wv.key_to_index.keys())
    wtovec = {}
    for i in range(0,len(vectors)):
        wtovec.update({words[i]: vectors[i]})

    # Done
    end = time.time()
    if verbose >= 1:
        info("Word2vec model generated in " + colored(f"{end-start:.2f}", "blue") + " sec")

    # Store model
    dump(wtovec, gzip.open(f"word2vec/{fname}", "wb"))
    if verbose >= 1:
        info("Word2vec model stored to " + colored(f"word2vec/{fname}", "cyan"))

    return wtovec


#
# Load and preprocess Word2Vec word vectors data
#
def load_word2vec_data(session, conf, verbose=1):
    # Check if path exists
    fpath = "word2vec"
    if not exists(fpath):
        mkdir(fpath)

    # Check settings
    if "w2v_vector_size" not in conf:
        warning(colored("w2v_vector_size", "cyan") + " not set (using " + colored("75", "blue") + ")")
        conf["w2v_vector_size"] = 75
    if "rebuild" not in conf:
        conf["rebuild"] = False
        
    # Filename
    fname = session["file"]
    fname = fname[fname.rfind("/")+1:]
    fname = fname.replace(".csv","").replace(".gz","") + f"_{conf['w2v_vector_size']}.emb"
    
    # Check if stored
    if exists(f"word2vec/{fname}") and not conf["rebuild"]:
        obj = load(gzip.open(f"word2vec/{fname}", "rb"))
        if verbose >= 1:
            info("Word2vec embeddings loaded from " + colored(f"word2vec/{fname}", "cyan"))
        session["X"] = obj[0]
        session["y"] = obj[1]
        return
    
    # Get Word2Vec model
    wtovec = load_word2vec_model(session, conf, verbose=verbose)
    
    # Generate new word vectors
    start = time.time()
    X = []
    y = []
    for xi,yi in zip(session["X"], session["y"]):
        vec = [0] * conf["w2v_vector_size"]
        nn = 0
        for w in xi:
            # Add vectors for each word
            if w.strip() != "":
                vec = [x1+x2 for x1,x2 in zip(vec,wtovec[w])]
                nn += 1
        if nn > 0:
            # Calculate mean values for the word vectors
            vec = [x/nn for x in vec]
            X.append(vec)
            y.append(yi)
            
    # Update data
    session["X"] = X
    session["y"] = y
    
    # Done
    end = time.time()
    if verbose >= 1:
        info("Word2vec embeddings generated in " + colored(f"{end-start:.2f}", "blue") + " sec")
    
    # Store
    dump([X,y], gzip.open(f"word2vec/{fname}", "wb"))
    if verbose >= 1:
        info("Word2vec embeddings stored to " + colored(f"word2vec/{fname}", "cyan"))
