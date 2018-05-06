from PIL import Image

def crop_image(input_image, output_image, start_x, start_y, width, height):
    """Pass input name image, output name image, x coordinate to start croping, y coordinate to start croping, width to crop, height to crop """
    input_img = Image.open(input_image)
    box = (start_x, start_y, start_x + width, start_y + height)
    output_img = input_img.crop(box)
    output_img.save("fragments50/" + output_image +".jpg")

def main():
    for i in range(1,3):
        for j in range(20,35):
            for k in range(20, 40):
                crop_image("glaucoma/0"+str(i)+"_g.jpg","g_"+str(i)+"_"+str(j)+"_"+str(k),
                           50*k, 50*j, 50, 50)

if __name__ == '__main__': main()