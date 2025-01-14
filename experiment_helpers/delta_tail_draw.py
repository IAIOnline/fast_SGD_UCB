import json

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib import colors

##################################################################################
# set there style and etc parameters of firuges
##################################################################################
LINESTYLES = [
    ("d", "dashdot"),
    ("densely dotted", (0, (1, 1))),
    ("densely dashed", (0, (5, 1))),
    ("d", "solid"),
    ("dashed", (0, (5, 5))),
    ("d", "dashed"),
    ("long dash with offset", (1, (1, 0))),
    ("densely dashed", (0, (2, 2))),
    ("d", "dotted"),
    ("densely dashdotdotted", (0, (3, 1, 1, 1, 1, 1))),
    ("dashdotdotted", (0, (3, 5, 1, 5, 1, 5))),
    ("loosely dashdotdotted", (0, (3, 10, 1, 10, 1, 10))),
    ("densely dashdotted", (0, (3, 1, 1, 1))),
    ("dashdotted", (0, (3, 5, 1, 5))),
    ("long dash with offset", (5, (10, 3))),
    ("loosely dashdotted", (0, (3, 10, 1, 10))),
    ("densely dashdotdotted", (0, (3, 1, 1, 1, 1, 1))),
    ("loosely dashed", (0, (5, 10))),
    ("dashdotdotted", (0, (3, 5, 1, 5, 1, 5))),
    ("loosely dashdotdotted", (0, (3, 10, 1, 10, 1, 10))),
]


COLORMAP_NAME = "tab20"
DPI = 500
FIGSIZE = (17, 8)
FONTSIZE = 20


def get_fig_set_style(lines_count):
    cmap = plt.colormaps.get_cmap(COLORMAP_NAME)
    if lines_count <= 8:
        colors_list = ["blue", "y", "black", "purple", "red", "c", "y", "g"]
    else:
        colors_list = [colors.to_hex(cmap(i)) for i in range(lines_count)]

    sns.set_context("paper", font_scale=1.5, rc={"lines.linewidth": 2.5})
    fig, ax = plt.subplots(dpi=DPI)
    plt.grid(which="both")
    return fig, ax, colors_list


def draw(filename, agent_names, ignore):
    with open(filename, "r") as f:
        arr = json.load(f)
    alg_names = arr["0.0"].keys()
    assert len(alg_names) == len(agent_names), "Provide a pretty name for all algos"

    fig, ax, colors = get_fig_set_style(len(agent_names) - len(ignore) + 1)
    linestyles = LINESTYLES[:len(colors)]

    x_s = arr["rewards_list"] #np.linspace(0.0, 10.0, 25)
    del arr["rewards_list"]

    alg_rez = {alg_name: [] for alg_name in alg_names}
    for key, val in arr.items():
        for alg_name, regret in val.items():
            alg_rez[alg_name].append(regret)
    plotted_num = 0
    for new_name, (name, rez) in zip(agent_names, alg_rez.items()):
        if name in ignore:
            continue
        ax.plot(x_s, rez, label=new_name, color=colors[plotted_num], linestyle= linestyles[plotted_num][1])
        plotted_num += 1
    ax.legend()
    ax.set_xlabel(r"$\Delta$")
    ax.set_ylabel("Expected regret")
    plt.grid(which="both")
    plt.grid()
    plt.legend(loc="upper right")
    return fig
