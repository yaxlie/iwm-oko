import random

import numpy as np
import cv2
import sys
import recognize


def classify_img(img):
    # todo zwraca true jesli img jest naczyniem krwionosnym, false wpw.
    return recognize.CheckImages(img)

CHUNK_SIZE = 64
WIDTH_RESIZED = 876
HEIGHT_RESIZED = 584


img_origin = cv2.imread('./res/01.jpg')
height, width, chan = img_origin.shape

img_result = np.zeros([height,width,3],dtype=np.uint8)

# recognize and set new image
for w in range (0,width-CHUNK_SIZE):
    images=[]
    for h in range(0, height-CHUNK_SIZE):
        images.append(img_origin[h:h+CHUNK_SIZE, w:w+CHUNK_SIZE])
        # img_chunk = img_origin[h:h+CHUNK_SIZE, w:w+
    np_list = np.array([img for img in images])
    classification = classify_img(np_list)
    for h in range(0, height - CHUNK_SIZE):
        if classification[h]:
            img_result[h+int(CHUNK_SIZE/2),w+int(CHUNK_SIZE/2)] = (255, 255, 255)
#        print("chunk ",w," ",h)
#     print(w)
#     debug
    b = ("Loading [" + "#" * int(w//(width/100)) + " " * (100 - int(w//(width/100))) + "]")
    # \r prints a carriage return first, so `b` is printed on top of the previous line.
    sys.stdout.write('\r' + b)
    sys.stdout.flush()


cv2.imwrite("result.jpg", img_result)
resized_image = cv2.resize(img_result, (WIDTH_RESIZED, HEIGHT_RESIZED))
cv2.imshow("OKO", resized_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
