import matplotlib.pyplot as plt


def graph(title: str,
          label: (str, str),
          data: (int[], float[]),
          file_name: str):

    plt.plot(data[0], data[1])
    plt.title(title)

    plt.xlabel(xlabel=label[0])
    plt.ylabel(ylabel=label[1])

    plt.savefig(file_name)
