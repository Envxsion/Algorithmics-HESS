import numpy as np
import matplotlib.pyplot as plt

def mandelbrot(x, y, max_iter):
    c = complex(x, y)
    z = 0
    for i in range(max_iter):
        z = z * z + c
        if abs(z) > 2:
            return i
    return max_iter

def mandelbrot_set(xmin, xmax, ymin, ymax, width, height, max_iter):
    result = np.zeros((height, width))
    for i in range(height):
        for j in range(width):
            x = xmin + (xmax - xmin) * j / (width - 1)
            y = ymin + (ymax - ymin) * i / (height - 1)
            result[i, j] = mandelbrot(x, y, max_iter)
    return result

xmin, xmax = -2.0, 1.0
ymin, ymax = -1.5, 1.5
width, height = 1000, 1000
max_iter = 100
mandelbrot_image = mandelbrot_set(xmin, xmax, ymin, ymax, width, height, max_iter)

plt.imshow(mandelbrot_image, extent=(xmin, xmax, ymin, ymax), cmap='hot', interpolation='nearest')
plt.colorbar()
plt.show()
