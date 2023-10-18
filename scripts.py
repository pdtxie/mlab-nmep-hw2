from data.datasets import CIFAR10Dataset


def visualise():
    x = CIFAR10Dataset()
    for a, b in x:
        print(a, b)

visualise()
