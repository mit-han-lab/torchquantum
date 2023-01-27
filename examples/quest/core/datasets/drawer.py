import matplotlib.pyplot as plt
from matplotlib import rc
from torchpack.utils.config import configs


def draw_scatter(
    x, y, x_label="real", y_label="pred", title="scatter", name="scatter.png"
):
    # rc("font", **{"family": "sans-serif", "sans-serif": ["Arial"], "size": 20})
    fig = plt.figure(figsize=(5, 5))
    ax = plt.subplot()
    ax.scatter(x, y)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.savefig("exp/" + configs.exp_name + "/" + name)
    plt.close()


def draw_curve(x, y, x_label="epoch", y_label="loss", title="curve", name="curve.png"):
    # rc("font", **{"family": "sans-serif", "sans-serif": ["Arial"], "size": 20})
    fig = plt.figure(figsize=(5, 5))
    ax = plt.subplot()
    ax.plot(x, y)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.savefig("exp/" + configs.exp_name + "/" + name)
    plt.close()
