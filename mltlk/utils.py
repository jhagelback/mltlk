# Basic stuff
from termcolor import colored
import time
from datetime import datetime
from os.path import exists
import pandas as pd
# Stopwords
from nltk.corpus import stopwords


def error(e):
    """
    Shows an error message.

    Args:
        e (str): The message to be shown 
    """
    print(colored("Error: ", "red", attrs=["bold"]) + e)

    
def warning(e):
    """
    Shows a warning message.

    Args:
        e (str): The message to be shown 
    """
    print(colored("Warning: ", "red", attrs=["bold"]) + e)
    

def info(e):
    """
    Shows an info message.

    Args:
        e (str): The message to be shown 
    """
    print(colored("Info: ", "yellow", attrs=["bold"]) + e)


def check_param(p, pname, types, vals=None, expr=None, expr_msg=None):
    """
    Checks if a parameter value is valid.

    Args:
        p: Value of the parameter
        pname (str): Name of the parameter
        types (list): Valid types for the parameter, for example [str,None]
        vals (list): Valid values for the parameter, for example ['classification','regression']. If None, no check is made against type (default: None)
        expr: Result of expression to check on parameter, for example 'p is not None'
        expr_msg: Message to show if result of expression is False
        
    Returns:
        True of parameter values is valid, False if not
    """
    
    if expr is not None and not expr:
        error(expr_msg)
        return False
    if types not in [None, []]:
        if type(types) != list:
            types = [types]
        if (p is not None and type(p) not in types) or (p is None and None not in types):
            error("invalid type for param " + colored(pname, "yellow") + " (" + colored(", ".join([str(x) for x in types]), "blue") + ")")
            return False
    if vals not in [None, []]:
        if type(vals) != list:
            vals = [vals]
        if (p is None and None not in types) or (p is not None and p not in vals):
            error("Invalid value for param " + colored(pname, "yellow") + " (" + colored(", ".join([str(x) for x in vals]), "blue") + ")")
            return False
    return True


def str_to_timestamp(tsstr, mode="from"):
    """
    Converts a date-time string to timestamp.

    Args:
        tsstr (str): date-time string, for example '2023-06-01'
        mode (str): when time string is a date only, use 'from' to set time to 00:00:00 and 'to' to set time from 23:59:59 (default: from)
        
    Returns:
        Tuple with timestamp (int) and parsed date-time string
    """
    
    # Check params
    if not check_param(tsstr, "tsstr", [str]): return None
    if not check_param(mode, "mode", [str], vals=["to","from"]): return None
    
    if tsstr is None or tsstr == "":
        return 0
    
    # Fill short datetimes
    if len(tsstr) == 10:
        # yyyy-mm-dd
        if mode == "from":
            tsstr += " 00:00:00"
        else:
            tsstr += " 23:59:59"
    if len(tsstr) == 16:
        # yyyy-mm-ddThh:mm
        tsstr += ":00"
    if len(tsstr) == 4:
        # yyyy
        if mode == "from":
            tsstr += "-01-01 00:00:00"
        else:
            tsstr += "-12-31 23:59:59"
    if len(tsstr) == 7:
        # yyyy-mm
        if mode == "from":
            tsstr += "-01 00:00:00"
        else:
            warning("Unknown days of month, setting last day to 30")
            tsstr += "-30 23:59:59"
    
    return (int(time.mktime(datetime.strptime(tsstr, "%Y-%m-%d %H:%M:%S").timetuple())), tsstr)


def timestamp_to_str(ts):
    """
    Converts a timestamp to a date-time string.

    Args:
        ts (int): timestamp
        
    Returns:
        Date-time string
    """
    
    if ts in [None, ""]:
        ts = time.time()
    tarr = time.localtime(int(ts))
    tstr = time.strftime("%Y-%m-%d %H:%M:%S", tarr)
    return tstr


def load_stopwords(stopwordslist, verbose=1):
    """
    Load one or more lists of stopwords.

    Args:
        stopwordslist (str, list or None): Lists of stopwords to be used for text pre-processing. Can be either languages (from the nltk.corpus package) or paths to csv files, for example ['english', 'data/custom_stopwords.csv']. If None, no stopwords will be used (default: None)
        
        
    Returns:
        List of all loaded stopwords
    """
    
    # Check params
    if not check_param(verbose, "verbose", [int], vals=[0,1]): return None
    
    # Check if no stopwords
    if stopwordslist in [None, "", []]:
        return None
    
    # Convert to list (if string)
    if type(stopwordslist) == str:
        stopwordslist = [stopwordslist]
        
    # Iterate over stopwords
    stopwrds = []
    l = ""
    for e in stopwordslist:
        try:
            stopwrds += stopwords.words(e)
            l += f"{e}, "
        except:
            # No nltk language, try to load file
            if exists(e):
                stopwrds += [x[0] for x in pd.read_csv(e, header=None).values]
                l += f"{e}, "
            else:
                # Not found
                warning("Unable to find stopwords file " + colored(e, "cyan"))
            
    if verbose >= 1:
        if l != "":
            info(f"Load {len(stopwrds)} stopwords from " + colored(l[:-2], "cyan"))
        else:
            warning("No stopwords loaded")
    return stopwrds
