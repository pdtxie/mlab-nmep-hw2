import numpy as np
import cv2

from data.datasets import CIFAR10Dataset, MediumImagenetHDF5Dataset


def visualise():
    cifar_dataset = iter(CIFAR10Dataset())
    # mihdf5_dataset = iter(MediumImagenetHDF5Dataset(224))

    for i in range(10):
        x1, y1 = next(cifar_dataset)
        # x2, y2 = next(mihdf5_dataset)

        x1_n = np.transpose(x1.numpy(), [1, 2, 0]) * 255.0
        # x2_n = np.transpose(x2.numpy(), [1, 2, 0]) * 255.0

        cv2.imwrite(f"cifar_{i}.jpg", x1_n)
        # cv2.imwrite(f"mihdf5_{i}.jpg", x2_n)


visualise()
