# Basic stuff
from termcolor import colored
from .utils import *
import time


#
# Load and preprocess Keras embeddings data
#
def load_embeddings_data(session, conf, verbose=1):
    # Check settings
    if "embeddings_size" not in conf:
        warning(colored("embeddings_size", "cyan") + " not set (using " + colored("75", "blue") + ")")
        conf["embeddings_size"] = 75
        
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    from tensorflow.keras.preprocessing.text import Tokenizer

    # Stopwords
    sw = load_stopwords(conf, verbose=verbose)
    if sw is None:
        sw = set()
    else:
        sw = set(sw)
    
    # Convert to list of texts
    X = []
    for wds in session["X_original"]:
        xi = []
        for w in wds.split(" "):
            if w not in sw and w.strip() != "":
                xi.append(w)
        xi = " ".join(xi)
        X.append(xi)
        
    # Tokenize the inputs: Each word is assigned a unique id and the input text is converted to a list of word id integers
    t = Tokenizer(num_words=None, filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n', lower=True, split=" ", char_level=False, oov_token=None, document_count=0)
    t.fit_on_texts(session["X_original"])
    session["tokenizer"] = t
    X = t.texts_to_sequences(session["X_original"])

    # Count number of unique words (vocabulary size)
    vocab_size = len(t.word_counts) + 1
    if verbose >= 1:
        info(f"Vocabulary size is " + colored(f"{vocab_size}", "blue"))
    
    if "max_length" not in conf or conf["max_length"] is None:
        maxlen = max([len(xi) for xi in X])
        if verbose >= 1:
            info("Max length not set, using max length "  + colored(f"{maxlen}", "blue") + " from input examples")
    else:
        maxlen = conf["max_length"]
    
    # Check how many examples that are covered by padding sequences on the specified number of words limit
    if verbose >= 1:
        tmp = [len(xi) for xi in X]
        tmp = [xi for xi in tmp if xi <= maxlen]
        info(colored(f"{len(tmp)/len(X)*100:.2f}%", "blue") + " of sequences covered by max length " + colored(f"{maxlen}", "blue"))

    # Pad input sequences to max length
    X = pad_sequences(X, maxlen=maxlen, padding="post") 

    # Update session
    session["X"] = X
    session["embeddings_size"] = conf["embeddings_size"]
    session["max_length"] = maxlen
    session["vocab_size"] = vocab_size
    
    
#
# Create embedding for example
#
def embedding(xi, session):
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    from tensorflow.keras.preprocessing.text import Tokenizer
    
    X = session["tokenizer"].texts_to_sequences([xi])
    
    # Pad input sequences to max length
    X = pad_sequences(X, maxlen=session["max_length"], padding="post") 
    
    return X
