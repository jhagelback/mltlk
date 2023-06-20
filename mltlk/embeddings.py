# Basic stuff
from termcolor import colored
from .utils import *
import time


#
# Load and preprocess Keras embeddings data
#
def load_embeddings_data(session, embeddings_size, max_length, stopwords, verbose=1):
    """
    Loads and pre-processes Keras embeddings word vectors for the loaded text data.

    Args:
        session: Session object (created in load_data())
        embeddings_size (int): Size of word vectors
        max_length (int): Max size of word vectors
        stopwords (str, list or None): Lists of stopwords to be used for text pre-processing. Can be either languages (from the nltk.corpus package) or paths to csv files, for example ['english', 'data/custom_stopwords.csv']. If None, no stopwords will be used (default: None)
        verbose (int): Set verbose (output messages) level (0 for no output messages) (default: 1)
        
    Returns:
        Word2vec word vectors data
    """
    
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    from tensorflow.keras.preprocessing.text import Tokenizer

    # Stopwords
    sw = load_stopwords(stopwords, verbose=verbose)
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
    
    if max_length is None:
        maxlen = max([len(xi) for xi in X])
        if verbose >= 1:
            info("Max length not set, using max length "  + colored(f"{maxlen}", "blue") + " from input examples")
    else:
        maxlen = max_length
    
    # Check how many examples that are covered by padding sequences on the specified number of words limit
    if verbose >= 1:
        tmp = [len(xi) for xi in X]
        tmp = [xi for xi in tmp if xi <= maxlen]
        info(colored(f"{len(tmp)/len(X)*100:.2f}%", "blue") + " of sequences covered by max length " + colored(f"{maxlen}", "blue"))

    # Pad input sequences to max length
    X = pad_sequences(X, maxlen=maxlen, padding="post") 

    # Update session
    session["X"] = X
    session["embeddings_size"] = embeddings_size
    session["max_length"] = maxlen
    session["vocab_size"] = vocab_size
    

def embedding(xi, session):
    """
    Creates an embedding word vector for an example.

    Args:
        xi: The example of text data
        session: Session object (created in load_data())
        
    Returns:
        Embeddings vector for the example
    """
    
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    from tensorflow.keras.preprocessing.text import Tokenizer
    
    X = session["tokenizer"].texts_to_sequences([xi])
    
    # Pad input sequences to max length
    X = pad_sequences(X, maxlen=session["max_length"], padding="post") 
    
    return X
