import matplotlib.pyplot as plt


def plotgrad(data, text="label", cmap="jet"):
    fig = plt.figure(figsize=(4, 4), frameon=False)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")

    img = plt.imshow(data, cmap=cmap, interpolation="spline16")
    plt.xticks([])
    plt.yticks([])
