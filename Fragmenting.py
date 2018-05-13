from PIL import Image

def crop_image(input_image, output_image, start_x, start_y, width, height):
    """Pass input name image, output name image, x coordinate to start croping, y coordinate to start croping, width to crop, height to crop """
    input_img = Image.open(input_image)
    box = (start_x, start_y, start_x + width, start_y + height)
    output_img = input_img.crop(box)
    output_img.save("fragments64/" + output_image +".jpg")

def main():
    for i in range(1,3):
        for j in range(20,35):
            for k in range(20, 40):
                crop_image("glaucoma/0"+str(i)+"_g.jpg","g_"+str(i)+"_"+str(j)+"_"+str(k),
                           64*k, 64*j, 64, 64)

if __name__ == '__main__': main()
