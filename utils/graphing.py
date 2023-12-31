import matplotlib.pyplot as plt


def graph(title: str,
          label: tuple[str, str],
          data: tuple[list[int], list[float]],
          file_name: str,
          legend: list[str],
          reset: bool = True):

    plt.plot(data[0], data[1])
    plt.title(title)

    x, y = label
    plt.xlabel(xlabel=x)
    plt.ylabel(ylabel=y)

    plt.legend(legend)

    plt.savefig(file_name)

    if reset:
        plt.clf()
