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

# left -> 81 -> move to 0false folder
# right -> 83 -> move to 0true folder
# backspace -> 8 -> revert last move
# escape -> 27 -> close program
# key1 -> 49 -> turn on/off gausian blur filter
# key2 -> 50 -> turn on/off median blur filter
# key3 -> 51 -> turn on/off bil filter
# key4 -> 52 -> less gamma
# key5 -> 53 -> more gamma
# key6 -> 54 -> less contrast
# key7 -> 55 -> more contrast
# other -> skip photo and move it to 0skipped dir

FOLDER_NAME = "fragments50"
PATH = "healthy/01_h.jpg"
DESTPATH = "res/"

IMG_SIZE = 200
IMG_SIZE_O = 64

KEY_LEFT = 81
KEY_UP = 82
KEY_DOWN = 84
KEY_RIGHT = 83
SPACE = 32
KEY_BACKSPACE = 8
KEY_ESCAPE = 27
KEY_1 = 49
KEY_2 = 50
KEY_3 = 51
KEY_4 = 52
KEY_5 = 53
KEY_6 = 54
KEY_7 = 55

MOVE_PIX = 5


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

    GAUSIAN_BLUR = True
    MEDIAN_BLUR = True
    BIL_FILTER = True

    last_moved = ""

    images = []
    for img in glob.glob(PATH):
        images.append(img)

    # images = [cv2.imread(file) for file in glob.glob("fragments50/*.jpg")]

    height = 700
    width = 1000
    work = True

    for file in images:
        img_origin = cv2.imread(file)
        while(work):
            gamma_nored = _GAMMA_NORED
            contrast_nored = _CONTRAST_NORED
            gamma_hsv = _GAMMA_HSV
            contrast_hsv = _CONTRAST_HSV

            # img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            img_o = img_origin[width:width + IMG_SIZE_O, height:height + IMG_SIZE_O]
            # cv2.imshow("or" + file.title(), img)

            img = cv2.resize(img_o, (IMG_SIZE, IMG_SIZE))

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
                print(key, "width:", width, " height:", height)

                if (key == KEY_1):  # cyfry od 1
                    GAUSIAN_BLUR = not GAUSIAN_BLUR
                elif (key == KEY_2):
                    MEDIAN_BLUR = not MEDIAN_BLUR
                elif (key == KEY_3):
                    BIL_FILTER = not BIL_FILTER
                elif (key == KEY_4):
                    gamma_hsv = gamma_hsv - GAMMA_STEP
                    gamma_nored = gamma_nored - GAMMA_STEP
                    if gamma_nored < 0:
                        gamma_nored = 0
                    if gamma_hsv < 0:
                        gamma_hsv = 0
                    print("gamma set to : ", gamma_nored, ", ", gamma_hsv)
                elif (key == KEY_5):
                    gamma_hsv = gamma_hsv + GAMMA_STEP
                    gamma_nored = gamma_nored + GAMMA_STEP
                    print("gamma set to : ", gamma_nored, ", ", gamma_hsv)
                elif (key == KEY_6):
                    contrast_hsv = contrast_hsv - CONTRAST_STEP
                    contrast_nored = contrast_nored - CONTRAST_STEP
                    if contrast_nored < 0:
                        contrast_nored = 0
                    if contrast_hsv < 0:
                        contrast_hsv = 0
                    print("contrast set to : ", contrast_nored, ", ", contrast_hsv)
                elif (key == KEY_7):
                    contrast_hsv = contrast_hsv + CONTRAST_STEP
                    contrast_nored = contrast_nored + CONTRAST_STEP
                    print("contrast set to : ", contrast_nored, ", ", contrast_hsv)
                elif key<49 or key > 60:
                    break

            if key == KEY_RIGHT:  # Right_Arrow
                height+=MOVE_PIX
            elif key == KEY_LEFT:  # Left_Arrow
                height-=MOVE_PIX
            elif key == KEY_UP:  # Left_Arrow
                width-=MOVE_PIX
            elif key == KEY_DOWN:  # Left_Arrow
                width+=MOVE_PIX
            elif key == SPACE:  # Left_Arrow
                cv2.imwrite(DESTPATH + str(width)+"_"+str(height)+".jpg", img_o)

            elif key == KEY_BACKSPACE:  # Backspace
                cv2.destroyAllWindows()
                shutil.move(last_moved, FOLDER_NAME+"/" + last_moved.split('/')[-1])
                print("Reverting changes for : ", last_moved)

            elif key == KEY_ESCAPE:  # Escape
                cv2.destroyAllWindows()
                work = False
                break

            # else:
            #     shutil.move(file.title().lower(), FOLDER_NAME+"/0skipped/" + file.title().lower().split('/')[-1])

            # cv2.destroyAllWindows()


if __name__ == '__main__': main()
