# Basic stuff
from .utils import *
from customized_table import *
from customized_chart import *
import numpy as np
from collections import Counter


#
# Plot numerical/nominal features
#
def plot_data(session, conf={}):
    if session is None:
        error("Session is empty")
        return
    
    # Check config
    if "mode" not in conf:
        conf["mode"] = ""
    if "horizontal" not in conf:
        conf["horizontal"] = True,
    if "category" not in conf:
        conf["category"] = None
    if "lim" not in conf:
        conf["lim"] = None
    if "table" not in conf:
        conf["table"] = True
    if "plot" not in conf:
        conf["plot"] = True
    if "size" not in conf:
        conf["size"] = (14,6)
    
    # Placeholder for numerical features
    num_data = {
        "values": [
        ],
        "series": []
    }
    
    # Placeholder for nominal features
    nom_data = {
        "values": [
        ],
        "series": []
    }
    
    # Use original or preprocessed data
    key = "X_original"
    if conf["mode"] in ["scale", "scaled"]:
        key = "X"
    
    # Categories to include
    if conf["category"] is None:
        cats = set(np.unique(session["y_original"]))
    else:
        cats = set([conf["category"]])
    
    # Iterate over features
    for i,col in enumerate(session["columns"]):
        # Nominal feature
        if type(session[key][i][0]) == str:
            nom_data["series"].append(col)
            nom_data["values"].append([xi[i] for xi,yi in zip(session[key],session["y_original"]) if yi in cats])
        # Numerical feature (update data)
        else:
            num_data["series"].append(col)
            num_data["values"].append([xi[i] for xi,yi in zip(session[key],session["y_original"]) if yi in cats])
    
    # Table (numerical features)
    if len(num_data["series"]) > 0 and conf["table"]:
        t = CustomizedTable(["Feature<br><font style='font-weight: normal'>(numerical)</font>", "Mean", "Median", "Min", "Max", "Stdev"])
        t.column_style(0, {"color": "name"})
        t.column_style([1,2,3,4,5], {"color": "value", "num-format": "dec-4"})
        for label,vals in zip(num_data["series"], num_data["values"]):
            t.add_row([
                label,
                float(np.mean(vals)),
                float(np.median(vals)),
                float(np.min(vals)),
                float(np.max(vals)),
                float(np.std(vals)),
            ])
        print()
        t.display()
        print()
        
    # Title
    title = None
    if conf["category"] is not None:
        title = conf["category"]
        
    # Table (nominal features)
    if len(nom_data["series"]) > 0 and conf["table"]:
        t = CustomizedTable(["Feature<br><font style='font-weight: normal'>(nominal)</font>", "Values (occurences)"])
        t.column_style(0, {"color": "name"})
        for label,vals in zip(nom_data["series"], nom_data["values"]):
            vtxt = ""
            cnt = Counter(vals)
            for val,n in cnt.items():
                vtxt += f"{val} <font color='#7566f9'>({n})</font>, "
            vtxt = vtxt[:-2]
            
            t.add_row([
                label,
                vtxt,
            ])
        if title is not None:
            t.add_colspan_row([[title,2]], style={"color": "#000", "font": "bold", "background": "#ddd", "row-toggle-background": 0, "border": "top bottom"})
        print()
        t.display()
        print()
    
    # Show plot for numerical features
    if len(num_data["series"]) > 0 and conf["plot"]:
        box_plot(num_data, opts={
            "grid": True,
            "font": "Verdana",
            "title_fontsize": 10,
            "fontsize": 10,
            "labels_fontsize": 10,
            "labels_color": "#b40403",
            "horizontal": conf["horizontal"],
            "title": title,
            "size": conf["size"],
            "lim": conf["lim"],
        })
    
    
#
# Plot numerical/nominal features per category
#
def plot_data_per_category(session, conf={}):
    if session is None:
        error("Session is empty")
        return
    
    # Check if classification
    if session["preprocess"] == "regression":
        error("Plot data per category requires classification")
        return
    
    # Check config
    if "mode" not in conf:
        conf["mode"] = ""
    
    # Use original or preprocessed data
    key = "X_original"
    if conf["mode"] in ["scale", "scaled"]:
        key = "X"
    
    nom = False
    num = False
    for i,col in enumerate(session["columns"]):
        # Nominal feature
        if type(session[key][i][0]) == str:
            nom = True
        # Numeric features
        else:
            num = True
    
    # Nummeric features
    if num:
        # Get min/max for all features
        vals = []
        for xi in session[key]:
            for v in xi:
                if type(v) != str:
                    vals.append(v)

        # Categories
        cats = np.unique(session["y_original"])
        for cat in cats:
            plot_data(session, conf={"category": cat, "table": False, "size": (10,4), "lim": (np.min(vals)-0.1,np.max(vals)+0.1), "mode": conf["mode"]})
            
    if nom:
        # Categories
        cats = np.unique(session["y_original"])
        for cat in cats:
            plot_data(session, conf={"category": cat, "plot": False})
