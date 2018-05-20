import random

import numpy as np
import cv2
import sys
import recognize


def classify_img(img):
    # todo zwraca true jesli img jest naczyniem krwionosnym, false wpw.
    return recognize.CheckImages(img)


for arg in sys.argv:
    CHUNK_SIZE = 64
    WIDTH_RESIZED = 876
    HEIGHT_RESIZED = 584

    image_name = "02.jpg"


    img_origin = cv2.imread('./res/' + image_name)

    # img_origin = cv2.resize(img_origin, (WIDTH_RESIZED, HEIGHT_RESIZED))

    height, width, chan = img_origin.shape

    img_result = np.zeros([height,width,3],dtype=np.uint8)

    # recognize and set new image
    MAX_IMAGES_COUNT = 1000000
    old_w = 0
    images=[]


    for w in range (0,width-CHUNK_SIZE):

        b = ("Loading [" + "#" * int(w // (width / 10)) + " " * (10 - int(w // (width / 10))) + "]")
        # \r prints a carriage return first, so `b` is printed on top of the previous line.
        sys.stdout.write('\r' + b)
        sys.stdout.flush()

        for h in range(0, height-CHUNK_SIZE):
            images.append(img_origin[h:h+CHUNK_SIZE, w:w+CHUNK_SIZE])
            # img_chunk = img_origin[h:h+CHUNK_SIZE, w:w+

        if(len(images) + height <= MAX_IMAGES_COUNT and w < width-CHUNK_SIZE-1):
            continue
        # set pixels
        np_list = np.array([img for img in images])
        classification = classify_img(np_list)
        for w2 in range(old_w, w-1):
            for h in range(0, height - CHUNK_SIZE):
                if classification[(w2*(width-CHUNK_SIZE)) + h]:
                    img_result[h+int(CHUNK_SIZE/2),w2+int(CHUNK_SIZE/2)] = (255, 255, 255)
        old_w = w + 1
        images = []
    #     print(w)

    cv2.imwrite("result_" + image_name, img_result)
#     resized_image = cv2.resize(img_result, (WIDTH_RESIZED, HEIGHT_RESIZED))
#     cv2.imshow(image_name, resized_image)
#
# cv2.waitKey(0)
# cv2.destroyAllWindows()
sys.exit()
