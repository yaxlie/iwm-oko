from PIL import Image

def crop_image(input_image, output_image, start_x, start_y, width, height):
    """Pass input name image, output name image, x coordinate to start croping, y coordinate to start croping, width to crop, height to crop """
    input_img = Image.open(input_image)
    box = (start_x, start_y, start_x + width, start_y + height)
    output_img = input_img.crop(box)
    output_img.save("fragments/" + output_image +".jpg")

def main():
    for i in range(1,3):
        for j in range(0,12):
            for k in range(0, 15):
                crop_image("diabetic_retinopathy/0"+str(i)+"_dr.JPG","dr_"+str(i)+"_"+str(j)+"_"+str(k),
                           150*k, 150*j, 150, 150)

if __name__ == '__main__': main()