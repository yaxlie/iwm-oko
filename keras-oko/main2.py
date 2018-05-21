import random

import numpy as np
import cv2
import sys
import recognize


def classify_img(img):
    # todo zwraca true jesli img jest naczyniem krwionosnym, false wpw.
    return recognize.CheckImages(img)


for arg in sys.argv[1::]:
    CHUNK_SIZE = 64
    WIDTH_RESIZED = 876
    HEIGHT_RESIZED = 584

    print(sys.argv)

    image_name = arg
    print(image_name, "start")

    img_origin = cv2.imread('./res/' + image_name)

    # img_origin = cv2.resize(img_origin, (WIDTH_RESIZED, HEIGHT_RESIZED))

    height, width, chan = img_origin.shape

    img_result = np.zeros([height,width,3],dtype=np.uint8)

    # recognize and set new image

    images=[]


    for w in range (0,width-CHUNK_SIZE):

        b = ("Loading [" + "#" * int(w // (width / 10)) + " " * (10 - int(w // (width / 10))) + "]")
        # \r prints a carriage return first, so `b` is printed on top of the previous line.
        sys.stdout.write('\r' + b)
        sys.stdout.flush()

        for h in range(0, height-CHUNK_SIZE):
            images.append(img_origin[h:h+CHUNK_SIZE, w:w+CHUNK_SIZE])

        # set pixels
        np_list = np.array([img for img in images])
        classification = classify_img(np_list)
        for h in range(0, height - CHUNK_SIZE):
            if classification[h]:
                img_result[h+int(CHUNK_SIZE/2),w+int(CHUNK_SIZE/2)] = (255, 255, 255)
        images = []
    #     print(w)

    cv2.imwrite("./res/result_" + image_name, img_result)
    resized_image = cv2.resize(img_result, (WIDTH_RESIZED, HEIGHT_RESIZED))
    cv2.imshow(image_name, resized_image)
#
# cv2.waitKey(0)
# cv2.destroyAllWindows()
sys.exit()
