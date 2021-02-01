import Seam
import cv2
import numpy as np


def main():
    path_list = []

    img_name = input('Image name: ')
    original = cv2.imread(img_name)
    Img = original

    print('Loading photo..')
    print('Original shape: ', original.shape)


    number_of_pixel = int(input('Number of cutting pixels: '))
    row,col,_ = original.shape
    if (number_of_pixel > col):
        print("Out of image size")
        return -1


    print('Cutting Image..')
    cv2.imshow('Original image', original)
    cv2.waitKey(3)
    i = 1
    while i <= number_of_pixel:
        print(i)

        # Bước 1: Chuyển sang ảnh mức xám
        greyImg = Seam.rgbToGrey(Img)

        # Bước 2: Tìm cạnh của các đối tượng bằng lọc Sobel
        edgeImg = Seam.getEdge(greyImg)

        # Bước 3: Quy hoạch động, tìm ma trận chi trí nhằm giúp lưu lại hướng đi của seam
        cost = Seam.findCostArr(edgeImg)

        # Bước 4: Tìm đường seam nhỏ nhất trên E
        # theo chiều từ trên xuống dưới
        path = Seam.findSeam(cost)

        # Bước 5: Vẽ Seam lên ảnh
        Seam.drawSeam(Img,path)
        cv2.imshow('crop', Img)

        # Bước 6: "chặt" đường seam trên ảnh mức xám
        Img = Seam.removeSeam(Img,path)
        
        cv2.waitKey(3)

        i += 1


    cv2.imshow('crop', Img)

    name_list = img_name.split('.')
    new_name = name_list[0] + '_resize_'+ str(number_of_pixel) + '.' + name_list[1]

    cv2.imwrite(new_name,Img)
    print('Save new image')

    cv2.destroyAllWindows()
    print('Done !!!')
    return 0


#Run 
main()