import cv2 as c
import matplotlib.pyplot as plt

# open image
img = c.imread("test.jpg", c.IMREAD_GRAYSCALE)
assert img is not None

# apply canny to reduce noise, finding intensity gradient, and apply non maximum supression
edges = c.Canny(img, threshold1=100, threshold2=200, apertureSize=3, L2gradient=True)

# show the image
plt.subplot(121), plt.imshow(img, cmap="gray")
plt.title("Original Image"), plt.xticks([]), plt.yticks([])
plt.subplot(122), plt.imshow(edges, cmap="gray")
plt.title("Edge Image"), plt.xticks([]), plt.yticks([])

plt.show()
