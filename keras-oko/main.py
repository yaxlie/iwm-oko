import random

import numpy as np
import cv2
import sys


def classify_img(img):
    # todo zwraca true jesli img jest naczyniem krwionosnym, false wpw.
    return bool(random.getrandbits(1))

CHUNK_SIZE = 64
WIDTH_RESIZED = 876
HEIGHT_RESIZED = 584


img_origin = cv2.imread('./res/01.jpg')
height, width, chan = img_origin.shape

img_result = np.zeros([height,width,3],dtype=np.uint8)

# recognize and set new image
for w in range (0,width-CHUNK_SIZE):
    for h in range(0, height-CHUNK_SIZE):
        img_chunk = img_origin[h:h+CHUNK_SIZE, w:w+CHUNK_SIZE]
        if classify_img(img_chunk):
            img_result[h+int(CHUNK_SIZE/2),w+int(CHUNK_SIZE/2)] = (255, 255, 255)

#     debug
    b = ("Loading [" + "#" * int(w//(width/10)) + " " * (10 - int(w//(width/10))) + "]")
    # \r prints a carriage return first, so `b` is printed on top of the previous line.
    sys.stdout.write('\r' + b)
    sys.stdout.flush()



resized_image = cv2.resize(img_result, (WIDTH_RESIZED, HEIGHT_RESIZED))
cv2.imshow("OKO", resized_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
