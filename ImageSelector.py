import shutil
import numpy as np
import cv2
import glob
from skimage import data, io, filters, exposure
from skimage.color import rgb2gray
from skimage.exposure import rescale_intensity, adjust_gamma
from PIL import ImageEnhance, Image

import os
from PIL import Image

# left -> 81 (false)
# right -> 83 (true)

IMG_SIZE = 200


def drawLines(img, g, b, r):
    cv2.line(img, (int(IMG_SIZE / 2), 0), (int(IMG_SIZE / 2), IMG_SIZE), (g, b, r), 1, 1)
    cv2.line(img, (0, int(IMG_SIZE / 2)), (IMG_SIZE, int(IMG_SIZE / 2)), (g, b, r), 1, 1)
    return img


def enhanceImg(imgOrigin, gamma, contrast, gausianBlur, medianBlur, bilFilter):
    img = imgOrigin.copy()
    img = adjust_gamma(img, gamma)

    if (gausianBlur == True):
        img = cv2.GaussianBlur(img, (5, 5), 100)
    if (medianBlur == True):
        img = cv2.medianBlur(img, 5)
    if (bilFilter == True):
        img = cv2.bilateralFilter(img, 9, 75, 75)

    pil_im = Image.fromarray(img)
    c_image = ImageEnhance.Contrast(pil_im)
    c_image = c_image.enhance(contrast)
    img = np.array(c_image)

    return img


def showImages(file, img, hsv, no_red):
    cv2.imshow(file.title(), img)
    cv2.imshow("HSV" + file.title(), hsv)
    cv2.imshow("NO_RED" + file.title(), no_red)

    cv2.moveWindow(file.title(), 0, 0)
    cv2.moveWindow("NO_RED" + file.title(), IMG_SIZE + 50, 0)
    cv2.moveWindow("HSV" + file.title(), IMG_SIZE * 2 + 50 * 2, 0)


def main():
    _GAMMA_NORED = 2.5
    _CONTRAST_NORED = 92
    _GAMMA_HSV = 1.5
    _CONTRAST_HSV = 2

    CONTRAST_STEP = 2
    GAMMA_STEP = 0.2

    gamma_nored = 2.5
    contrast_nored = 92
    gamma_hsv = 1.5
    contrast_hsv = 2

    GAUSIAN_BLUR = True
    MEDIAN_BLUR = True
    BIL_FILTER = True

    last_moved = ""

    images = []
    for img in glob.glob("fragments50/*.jpg"):
        images.append(img)

    # images = [cv2.imread(file) for file in glob.glob("fragments50/*.jpg")]

    for file in images:
        gamma_nored = _GAMMA_NORED
        contrast_nored = _CONTRAST_NORED
        gamma_hsv = _GAMMA_HSV
        contrast_hsv = _CONTRAST_HSV

        img = cv2.imread(file)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

        hsv = cv2.applyColorMap(img, cv2.COLORMAP_HSV)

        no_red = img.copy()
        no_red[:, :, 2] = 0

        while (True):
            _hsv = hsv.copy()
            _no_red = no_red.copy()

            _hsv = enhanceImg(_hsv, gamma_hsv, contrast_hsv, GAUSIAN_BLUR, MEDIAN_BLUR, BIL_FILTER)
            _no_red = enhanceImg(_no_red, gamma_nored, contrast_nored, GAUSIAN_BLUR, MEDIAN_BLUR, BIL_FILTER)

            drawLines(_hsv, 0, 0, 255)
            drawLines(_no_red, 0, 0, 255)
            drawLines(img, 255, 0, 0)

            showImages(file, img, _hsv, _no_red)

            key = cv2.waitKey(0)
            print(key)

            if (key == 49):  # cyfry od 0
                GAUSIAN_BLUR = not GAUSIAN_BLUR
            elif (key == 50):
                MEDIAN_BLUR = not MEDIAN_BLUR
            elif (key == 51):
                BIL_FILTER = not BIL_FILTER
            elif (key == 52):
                gamma_hsv = gamma_hsv - GAMMA_STEP
                gamma_nored = gamma_nored - GAMMA_STEP
                if gamma_nored < 0:
                    gamma_nored = 0
                if gamma_hsv < 0:
                    gamma_hsv = 0
                print("gamma set to : ", gamma_nored, ", ", gamma_hsv)
            elif (key == 53):
                gamma_hsv = gamma_hsv + GAMMA_STEP
                gamma_nored = gamma_nored + GAMMA_STEP
                print("gamma set to : ", gamma_nored, ", ", gamma_hsv)
            elif (key == 54):
                contrast_hsv = contrast_hsv - CONTRAST_STEP
                contrast_nored = contrast_nored - CONTRAST_STEP
                if contrast_nored < 0:
                    contrast_nored = 0
                if contrast_hsv < 0:
                    contrast_hsv = 0
                print("contrast set to : ", contrast_nored, ", ", contrast_hsv)
            elif (key == 55):
                contrast_hsv = contrast_hsv + CONTRAST_STEP
                contrast_nored = contrast_nored + CONTRAST_STEP
                print("contrast set to : ", contrast_nored, ", ", contrast_hsv)
            elif key<49 or key > 60:
                break

        if key == 83:  # Right_Arrow
            shutil.move(file.title().lower(), "fragments50/0true/" + file.title().lower().split('/')[-1])
            # os.rename(img.title().lower(), "fragments50/0true/"+img.title().lower())
            print(file.title(), " is TRUE -> moving to ./0true/\n")
            last_moved = "fragments50/0true/" + file.title().lower().split('/')[-1]

        elif key == 81:  # Left_Arrow
            shutil.move(file.title().lower(), "fragments50/0false/" + file.title().lower().split('/')[-1])
            # os.rename(img.title().lower(), "fragments50/0false/"+img.title().lower())
            print(file.title(), " is FALSE -> moving to ./0false/\n")
            last_moved = "fragments50/0false/" + file.title().lower().split('/')[-1]

        elif key == 8:  # Backspace
            cv2.destroyAllWindows()
            shutil.move(last_moved, "fragments50/" + last_moved.split('/')[-1])
            print("Reverting changes for : ", last_moved)

        elif key == 27:  # Escape
            cv2.destroyAllWindows()
            break

        shutil.move(file.title().lower(), "fragments50/0skipped/" + file.title().lower().split('/')[-1])

        cv2.destroyAllWindows()


if __name__ == '__main__': main()
