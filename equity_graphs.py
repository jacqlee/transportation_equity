import matplotlib
import matplotlib.pyplot as plt
import numpy as np


def process_graphs(metric, labels, title="Equity Measure"):
    x = np.arange(len(labels))  # the label locations
    width = 0.35  # the width of the bars

    fig, ax = plt.subplots()
    # print(len(x))
    # print(len(metric))
    ax.bar(x - width / 2, metric, width)

    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)

    plt.savefig(title+".png")
    plt.show()
