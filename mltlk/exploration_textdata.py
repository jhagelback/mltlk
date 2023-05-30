# Basic stuff
from termcolor import colored
import numpy as np
import pandas as pd
from collections import Counter
from customized_table import *
from .utils import *


# Caches to speed up processing
cache_topcats = {}
cache_overlap = {}

#
# Builds a corpus from text data
#
def build_corpus(session):
    if session is None:
        error("Session is empty")
        return None
    if "y_original" not in session or "X_original" not in session:
        error("Data must be loaded using function " + colored("load_data()", "cyan"))
        return None
    if type(session["X_original"][0]) != str:
        error("Input does not seem to be text")
        return None
    
    # Reset cache
    global cache_topcats
    cache_topcats = {}
    
    # Stop words
    sw = session["stopwords"]
    if sw is None:
        sw = set()
    else:
        sw = set(sw)
    
    # Corpus
    corpus = {
        "words": [],
        "word_count": None,
        "max_count": None,
        "total_documents": len(session["X_original"]),
        "total_words": None,
        "unique_words": None,
        "size_documents": [],
        "categories": np.unique(session["y_original"]),
        "documents_per_category": {},
        "words_per_category": {},
        "words_in_category": {},
    }
    
    # Build corpus of all words
    for xi, yi in zip(session["X_original"], session["y_original"]):
        wrds = xi.split()
        corpus["size_documents"].append(len(wrds))
        if yi not in corpus["words_per_category"]:
            corpus["words_per_category"][yi] = []
        for w in wrds:
            if w not in sw:
                corpus["words"].append(w)
                corpus["words_per_category"][yi].append(w)
                if w not in corpus["words_in_category"]:
                    corpus["words_in_category"][w] = set()
                corpus["words_in_category"][w].add(yi)
                
    # Summary and word count
    corpus["total_words"] = len(corpus["words"])
    cnt = Counter(corpus["words"])
    corpus["word_count"] = []
    for wrd,n in cnt.items():
        corpus["word_count"].append([wrd,n])
    corpus["word_count"] = sorted(corpus["word_count"], key=lambda x: x[1], reverse=True)
    corpus["unique_words"] = len(cnt)
    
    # Documents per category
    for yi in session["y_original"]:
        if yi not in corpus["documents_per_category"]:
            corpus["documents_per_category"][yi] = 0
        corpus["documents_per_category"][yi] += 1
    
    # Summary    
    t = CustomizedTable(["", ""], header=False)
    #t.column_style(0, {"font": "bold", "color": "param-key"})
    t.column_style(1, {"color": "value"})
    t.add_subheader([["Corpus summary", 2]])
    t.add_row(["Total documents:", corpus["total_documents"]])
    t.add_row(["Total words:", corpus["total_words"]])
    t.add_row(["Unique words:", corpus["unique_words"]])
    t.add_row(["Categories:", len(corpus["categories"])])
    t.add_row(["Average words/doc:", f"{np.mean(corpus['size_documents']):.2f}"])
    t.add_row(["Max words/doc:", f"{np.max(corpus['size_documents'])}"])
    t.add_row(["Min words/doc:", f"{np.min(corpus['size_documents'])}"])
    t.add_row(["Stdev words/doc:", f"{np.std(corpus['size_documents']):.2f}"]) 
    print()
    t.display()
    print()
    
    return corpus


#
# Show top words in the corpus
#
def top_words(corpus, n=10, sidx=0, ncats=10):
    global cache_topcats
    
    # Convert data
    tab = []
    for wrdcnt in corpus["word_count"]:
        incats = len(corpus["words_in_category"][wrdcnt[0]])
        tab.append(wrdcnt + [f"{incats} ({incats/len(corpus['categories'])*100:.1f}%)"])
    
    # No to categories to show
    tcats = ncats
    if len(corpus["categories"]) < ncats:
        tcats = len(corpus["categories"])
    
    # Create table
    t = CustomizedTable(["","Word","Count", "Appears for<br>categories", f"Top {tcats} categories"])
    t.column_style(0, {"color": "size"})
    t.column_style(2, {"color": "value"})
    t.column_style(3, {"color": "percent"})
    
    # Start and end index
    si = sidx
    if sidx < 0:
        si = len(tab) + sidx
    
    for i,r in enumerate(tab[si:si+n]):
        # Get top categories for word
        wrd = r[0]
        if wrd in cache_topcats:
            cats = cache_topcats[wrd]
        else:
            cats = []
            for yi in corpus["categories"]:
                if wrd in corpus["words_per_category"][yi]:
                    cats.append([yi, corpus["words_per_category"][yi].count(wrd)])
            cats = sorted(cats, key=lambda x: x[1], reverse=True)
            cache_topcats[wrd] = cats
        l = ""
        for c in cats[:tcats]:
            l += f"{c[0]} <font color='#9d93fb'>({c[1]})</font>, "
        l = l[:-2]
        # Add row
        t.add_row([si+i] + r + [l])
    print()
    t.display()
    print()


#
# Check overlapping words between two categories
#
def overlap(corpus, cat1, cat2):
    wrds1 = set(corpus["words_per_category"][cat1])
    wrds2 = set(corpus["words_per_category"][cat2])
    tot = wrds1.union(wrds2)
    overlap = wrds1.intersection(wrds2)
    
    t = CustomizedTable(["Category", "Unique words", "No documents"])
    t.column_style([1,2], {"color": "value"})
    t.add_row([cat1, len(wrds1), corpus["documents_per_category"][cat1]])
    t.cell_style(0, -1, {"color": "name"})
    t.add_row([cat2, len(wrds2), corpus["documents_per_category"][cat2]])
    t.cell_style(0, -1, {"color": "name"})
    t.add_row(["Both:", len(tot), corpus["documents_per_category"][cat1]+corpus["documents_per_category"][cat2]], style={"border": "top"})
    t.add_row(["Overlap:", f"{len(overlap)}&nbsp;&nbsp;&nbsp;{len(overlap)/len(tot)*100:.2f}%", ""])
    print()
    t.display()
    print()
    

#
# Check overlapping words between all categories
#
def overlap_all_categories(corpus, n=10, sidx=0):
    global cache_overlap
    done = set()
    tab = []
    for cat1 in corpus["categories"]:
        for cat2 in corpus["categories"]:
            key1 = f"{cat1}-{cat2}"
            key2 = f"{cat2}-{cat1}"
            if key1 not in done and key2 not in done and key1 != key2:
                if key1 in cache_overlap:
                    pct = cache_overlap[key1]
                elif key2 in cache_overlap:
                    pct = cache_overlap[key2]
                else:
                    wrds1 = set(corpus["words_per_category"][cat1])
                    wrds2 = set(corpus["words_per_category"][cat2])
                    tot = wrds1.union(wrds2)
                    overlap = wrds1.intersection(wrds2)
                    pct = len(overlap)/len(tot)
                    cache_overlap[key1] = pct
                    cache_overlap[key2] = pct
                done.add(key1)
                done.add(key2)
                tab.append([cat1, cat2, pct])
    tab = sorted(tab, key=lambda x: x[2], reverse=True)
    
    # Start and end index
    si = sidx
    if sidx < 0:
        si = len(tab) + sidx
    
    t = CustomizedTable(["", "Category 1", "Category 2", "Overlap"])
    t.column_style(0, {"color": "size"})
    t.column_style([1, 2], {"color": "name"})
    t.column_style(3, {"color": "percent", "num-format": "pct-2"})
    for i,r in enumerate(tab[si:si+n]):
        t.add_row([si+i] + r)
    print()
    t.display()
    print()