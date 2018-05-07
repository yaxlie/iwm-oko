import shutil

import cv2
import glob

import os
from PIL import Image

# left -> 81 (false)
# right -> 83 (true)

def main():
    IMG_SIZE = 200

    last_moved = ""

    images = []
    for img in glob.glob("fragments50/*.jpg"):
        images.append(img)

    # images = [cv2.imread(file) for file in glob.glob("fragments50/*.jpg")]

    for file in images:
        img = cv2.imread(file)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

        hsv = cv2.applyColorMap(img, cv2.COLORMAP_HSV)

        cv2.line(hsv, (int(IMG_SIZE/2), 0), (int(IMG_SIZE/2), IMG_SIZE), (0, 0, 255), 1, 1)
        cv2.line(hsv, (0, int(IMG_SIZE/2)), (IMG_SIZE, int(IMG_SIZE/2)), (0, 0, 255), 1, 1)

        cv2.line(img, (int(IMG_SIZE / 2), 0), (int(IMG_SIZE / 2), IMG_SIZE), (255, 0, 0), 1, 1)
        cv2.line(img, (0, int(IMG_SIZE / 2)), (IMG_SIZE, int(IMG_SIZE / 2)), (255, 0, 0), 1, 1)


        cv2.imshow(file.title(), img)
        cv2.imshow("HSV" + file.title(), hsv)
        cv2.moveWindow(file.title(), 0, 0)
        cv2.moveWindow("HSV" + file.title(), IMG_SIZE+50, 0)

        key = cv2.waitKey(0)
        print(key)

        if key== 83:   # Right_Arrow
            shutil.move(file.title().lower(), "fragments50/0true/"+file.title().lower().split('/')[-1])
            # os.rename(img.title().lower(), "fragments50/0true/"+img.title().lower())
            print(file.title(), " is TRUE -> moving to ./0true/\n")
            last_moved = "fragments50/0true/"+file.title().lower().split('/')[-1]

        elif key == 81: # Left_Arrow
            shutil.move(file.title().lower(), "fragments50/0false/"+file.title().lower().split('/')[-1])
            # os.rename(img.title().lower(), "fragments50/0false/"+img.title().lower())
            print(file.title(), " is FALSE -> moving to ./0false/\n")
            last_moved = "fragments50/0false/"+file.title().lower().split('/')[-1]

        elif key == 8:  # Backspace
            cv2.destroyAllWindows()
            shutil.move(last_moved,"fragments50/"+last_moved.split('/')[-1])
            print ("Reverting changes for : ", last_moved)

        elif key == 27:  # Escape
            cv2.destroyAllWindows()
            break

        cv2.destroyAllWindows()


if __name__ == '__main__': main()