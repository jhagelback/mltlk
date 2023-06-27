# Basic stuff
from .utils import *
from customized_table import *
from customized_chart import *
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.preprocessing import LabelEncoder


def plot_data(session, mode=None, horizontal=True, category=None, lim=None, table=True, plot=True, size=(14,6)):
    """
    Plots numerical and/or nominal features in the loaded dataset.

    Args:
        session: Session object (created in load_data())
        mode (str or None): Set to 'scaled' if used scaled inputs instead of original values (None) (default: None)
        horizontal (bool): Horizontal (True) or vertical (False) plots (default: True)
        category (str, int or None): Set if only plot examples from a specific category. If None, all examples are plotted (default: None)
        lim (tuple of ints): Min and max values for the value axis, for example (0,1000). If None, auto limit is used (default: None)
        table (bool): Set if show table for numerical and/or nominal features.
        plot (bool): Set if show plot for numerical features.
        size (tuple of ints): Size of plot (default (14,6))
    """
    
    # Check params
    if not check_param(mode, "mode", [str,None], vals=["scaled"]): return
    if not check_param(horizontal, "horizontal", [bool]): return
    if not check_param(lim, "lim", [tuple,None]): return
    if not check_param(table, "table", [bool]): return
    if not check_param(plot, "plot", [bool]): return
    if not check_param(size, "size", [tuple]): return
    if session is None:
        error("Session is empty")
        return
    
    
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
    if mode == "scaled":
        key = "X"
    
    # Categories to include
    if category is None:
        cats = set(np.unique(session["y_original"]))
    else:
        cats = set([category])
    
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
    if len(num_data["series"]) > 0 and table:
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
    if category is not None:
        title = category
        
    # Table (nominal features)
    if len(nom_data["series"]) > 0 and table:
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
    if len(num_data["series"]) > 0 and plot:
        box_plot(num_data, opts={
            "grid": True,
            "font": "Verdana",
            "title_fontsize": 10,
            "fontsize": 10,
            "labels_fontsize": 10,
            "labels_color": "#b40403",
            "horizontal": horizontal,
            "title": title,
            "size": size,
            "lim": lim,
        })
    

def plot_data_per_category(session, mode=None):
    """
    Plots numerical and/or nominal features with one plot/table per category in the loaded dataset.

    Args:
        session: Session object (created in load_data())
        mode (str or None): Set to 'scaled' if used scaled inputs instead of original values (None) (default: None)
    """
    
    # Check params
    if not check_param(mode, "mode", [str,None], vals=["scaled"]): return
    if session is None:
        error("Session is empty")
        return
    
    if session is None:
        error("Session is empty")
        return
    
    # Check if classification
    if session["preprocess"] == "regression":
        error("Plot data per category requires classification")
        return
    
    # Use original or preprocessed data
    key = "X_original"
    if mode == "scaled":
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
            plot_data(session, category=cat, table=False, size=(10,4), lim=(np.min(vals)-0.1,np.max(vals)+0.1), mode=mode)
            
    if nom:
        # Categories
        cats = np.unique(session["y_original"])
        for cat in cats:
            plot_data(session, category=cat, plot=False)


def plot_decision_borders(session, title=True, cmap="Spectral", eps=1, markersize=20):
    """
    Plot decision borders for two-dimensional datasets.

    Args:
        session: Session object (created in load_data())
        title (bool or str): Title for the plot. If True, title is generated from the model. If False, no title is used (default: True) 
        cmap (str): Colormap to be used (see https://matplotlib.org/stable/gallery/color/colormap_reference.html) (default: 'Spectal')
        eps (int,float): Extends the min/max values of X in the plot (default: 1.0)
        markersize (int): Size of the markers for the actual data points (default: 20)
    """
    # Check params
    if not check_param(title, "title", [bool,str]): return None
    if not check_param(eps, "eps", [int,float]): return None
    if not check_param(eps, "eps", [int,float], expr=eps>=0, expr_msg="eps must be 0 or higher"): return None
    if not check_param(markersize, "markersize", [int]): return None
    if not check_param(markersize, "markersize", [int], expr=markersize>=1, expr_msg="markersize must be at least 1"): return None
    if not check_param(cmap, "cmap", [str]): return None
    cmaps = plt.colormaps()
    if cmap not in cmaps:
        error(colored(cmap, "cyan") + " is not a valid colormap, see https://matplotlib.org/stable/gallery/color/colormap_reference.html")
        return None

    # Check if two-dimensional
    if type(session["X"]) == list:
        fts = len(session["X"][0])
    else:
        fts = session["X"].shape[1]
    if fts != 2:
        error("Decision borders can only be shown for two-dimensional datasets (found " + colored(fts, "blue") + " dimensions)")
        return
    
    # Check if model has been built
    if "model" not in session:
        error("Final model has not been built. Use the function " + colored("build_model()", "cyan"))
        return
    
    # Targets
    target = session["y"]
    if type(target[0]) != int:
        target = LabelEncoder().fit_transform(target)
    
    disp = DecisionBoundaryDisplay.from_estimator(session["model"], 
                                                  session["X"], 
                                                  response_method="predict", 
                                                  alpha=0.5, 
                                                  grid_resolution=250, 
                                                  eps=eps,
                                                  xlabel=session["columns"][0],
                                                  ylabel=session["columns"][1],
                                                  cmap=cmap
                                                 )
    disp.ax_.scatter(session["X"][:,0], session["X"][:,1], c=target, edgecolor="k", s=markersize, cmap=cmap)
    plt.axis("tight")
    if type(title) == str:
        plt.title(title)
    if title == True:
        mstr = str(session["model"])
        if "(" in mstr:
            plt.title(mstr.split("(")[0])
        else:
            plt.title(mstr)
    plt.show()
