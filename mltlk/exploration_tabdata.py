# Basic stuff
from .utils import *
from customized_table import *
from customized_chart import *
import numpy as np
from collections import Counter


#
#
#
def plot_data(session):
    # Placeholder for numerical features
    num_data = {
        "values": [
        ],
        "series": []
    }
    
    # Placeholder for nomina features
    nom_data = {
        "values": [
        ],
        "series": []
    }
    
    # Iterate over features
    for i,col in enumerate(session["columns"]):
        # Nominal feature
        if type(session["X_original"][i][0]) == str:
            nom_data["series"].append(col)
            nom_data["values"].append([xi[i] for xi in session["X_original"]])
        # Numerical feature (update data)
        else:
            num_data["series"].append(col)
            num_data["values"].append([xi[i] for xi in session["X_original"]])
    
    # Table (numerical features)
    if len(num_data["series"]) > 0:
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
        
    # Table (nominal features)
    if len(nom_data["series"]) > 0:
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
        print()
        t.display()
        print()
    
    # Show plot for numerical features
    if len(num_data["series"]) > 0:
        box_plot(num_data, opts={
            "grid": True,
            "font": "Verdana",
            "fontsize": 12,
            "labels_fontsize": 12,
        })
    