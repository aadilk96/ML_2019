import matplotlib as mp
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def main():
    vectors = np.loadtxt('mfeat-pix.txt')
    print(vectors.shape)

    zeros = vectors[0][:]
    zeros = zeros.reshape(16, 15)
    # ones  = vectors[300][:]
    # ones = ones.reshape(16, 15)

    for i in range(1, 10):
        for j in range(1, 10):
            plt.subplot(10,10,(i-1)* 10 + j);
            plt.imshow(zeros, cmap='gray')
            plt.axis('off')

    plt.show()

if __name__ == '__main__':
    main()
