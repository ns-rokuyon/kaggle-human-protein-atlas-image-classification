import matplotlib.pyplot as plt
import numpy as np


def display_imgs(x):
    columns = 4
    bs = x.shape[0]
    rows = min((bs + 3) // 4, 4)
    fig = plt.figure(figsize=(columns * 4, rows * 4))
    for i in range(rows):
        for j in range(columns):
            idx = i * columns + j
            print(rows, columns, idx + 1)
            fig.add_subplot(rows, columns, idx + 1)
            plt.axis('off')
            plt.imshow((x[i, :, :, :3] * 255).astype(np.int))
    plt.show()