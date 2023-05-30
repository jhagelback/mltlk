# Basic stuff
from termcolor import colored
import time
from datetime import datetime
from os.path import exists
import pandas as pd
# Stopwords
from nltk.corpus import stopwords


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
# Converts a date-time string to timestamp
#
def str_to_timestamp(tsstr, mode="from"):
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


#
# Converts a timestamp to a date-time string
#
def timestamp_to_str(ts):
    if ts in [None, ""]:
        ts = time.time()
    tarr = time.localtime(int(ts))
    tstr = time.strftime("%Y-%m-%d %H:%M:%S", tarr)
    return tstr


#
# Load stopwords
#
def load_stopwords(conf, verbose=1):
    # Check if no stopwords
    if "stopwords" not in conf or conf["stopwords"] is None:
        return None
    
    # Convert to list (if string)
    if type(conf["stopwords"]) == str:
        conf["stopwords"] = [conf["stopwords"]]
        
    # Iterate over stopwords
    stopwrds = []
    l = ""
    for e in conf["stopwords"]:
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
