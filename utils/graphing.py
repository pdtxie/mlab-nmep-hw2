import matplotlib.pyplot as plt


def graph(title: str,
          label: tuple[str, str],
          data: tuple[list[int], list[float]],
          file_name: str):

    plt.plot(data[0], data[1])
    plt.title(title)

    plt.xlabel(xlabel=label[0])
    plt.ylabel(ylabel=label[1])

    plt.savefig(file_name)
