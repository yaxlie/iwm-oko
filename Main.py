import cv2
from skimage.color import rgb2gray
from skimage.exposure import rescale_intensity, adjust_gamma
from PIL import ImageEnhance, Image
from skimage import data, io, filters, exposure
from skimage.measure import moments, moments_central, moments_normalized
import numpy as np
from skimage.filters import gaussian

import numpy as np

g = 2.5
c = 4
m = 50

oko = cv2.imread("oko.jpg")

img2 = rgb2gray(oko)

img = adjust_gamma(oko, g)

# img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
pil_im = Image.fromarray(img)
contrast = ImageEnhance.Contrast(pil_im)
contrast = contrast.enhance(c)
img = np.array(contrast)

# img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

img = cv2.GaussianBlur(img, (5, 5), 0.8)

# retval, thresh = cv2.threshold(img, m, 255, cv2.THRESH_BINARY)

img2 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
sobel = filters.sobel(img2)
# sobel2 = filters.sobel(img2)

# img = cv2.resize(img, (500, 500))
cv2.imshow('Oko', img)
cv2.imshow('Sobel', sobel)

cv2.waitKey(0)
cv2.destroyAllWindows()