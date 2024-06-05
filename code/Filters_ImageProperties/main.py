import imageio as imageio
from scipy import fftpack
import numpy as np
from PIL import Image, ImageDraw, ImageOps, ImageChops
import matplotlib.pyplot as plt
import math
import cv2 as cv

def K_Means():
    Org_img = Image.open("Image_to_be_segmented.jpg")
    Org_img.thumbnail((300, 300))
    Org_img = np.array(Org_img)

    w, h, c = Org_img.shape
    print(w, h, c)

    # reshaping the image into 1d array with values r,g,b
    img = Org_img.reshape(w * h, c)

    def Init_Centroids(k):
        idx = [np.random.randint(w * h) for i in range(k)]
        centroids = img[idx, :]
        return centroids

    def Calculate_Distance(p1, p2):
        Distance = 0
        Distance = np.sqrt(((p1[0] - p2[0]) ** 2) + ((p1[1] - p2[1]) ** 2) + ((p1[2] - p2[2]) ** 2))
        return Distance

    def AssignItemCentroid(Centroids):

        centroids_dect = {}
        for center in range(Centroids.shape[0]):
            centroids_dect[center] = []

        for i in range(img.shape[0]):
            distance = [Calculate_Distance(img[i], Centroids[center]) for center in range(Centroids.shape[0])]
            idx = distance.index(min(distance))
            centroids_dect[idx].append(img[i])
            img[i] = Centroids[idx]

        return centroids_dect

    def ComputeCentroidsMeans(Centroids, dict):
        j = 0
        for cluster in dict:
            r, g, b = [0, 0, 0]
            cnt = 1
            for values in dict[cluster]:
                r += values[0]
                g += values[1]
                b += values[2]
                cnt += 1
            r /= cnt
            g /= cnt
            b /= cnt
            Centroids[j] = [r, g, b]
            j += 1
        return Centroids

    def Run_KMeans():
        Centroids = None
        dect = None
        k = 3
        iterations = 1
        Centroids = Init_Centroids(k)
        for i in range(iterations):
            dect = AssignItemCentroid(Centroids)
            Centroids = ComputeCentroidsMeans(Centroids, dect)

    Run_KMeans()
    img2 = img.reshape(w, h, c)
    img2 = Image.fromarray(img2)
    img2.save("Image_After_Segmentation.png")

    print("Image Saved")


def Band_Reject():
    # open image
    Original_Image = Image.open('Image_with_periodic_noise.jpg')

    # convert image to numpy array
    Original_Image_np = np.array(Original_Image)

    # Create a low Reject filter image
    x_position = Original_Image_np.shape[0]
    y_position = Original_Image_np.shape[1]

    # size of circle
    Small_x = 40
    Small_y = 40
    Large_x = 100
    Large_y = 100

    # create a box
    Small_box = ((x_position / 2) - (Small_x / 2), (y_position / 2) - (Small_y / 2), (x_position / 2) + (Small_x / 2),
                 (y_position / 2) + (Small_y / 2))

    Large_box = ((x_position / 2) - (Large_x / 2), (y_position / 2) - (Large_y / 2), (x_position / 2) + (Large_x / 2),
                 (y_position / 2) + (Large_y / 2))

    # create new fill image
    Band_Reject_Filter = Image.new("L", (Original_Image_np.shape[1], Original_Image_np.shape[0]), color=1)

    Band_Reject_Filter_Draw = ImageDraw.Draw(Band_Reject_Filter)

    # draw the filter
    Band_Reject_Filter_Draw.ellipse(Large_box, fill=0)
    Band_Reject_Filter_Draw.ellipse(Small_box, fill=1)

    # change filter to np array
    Band_Reject_Filter_np = np.array(Band_Reject_Filter)

    # plot the filter
    plt.imshow(Band_Reject_Filter)
    plt.show()

    if Original_Image.mode == "RGB":
        # split image into three channels
        Red_Channel, Green_Channel, Blue_Channel = Original_Image.split()
        Red_Channel_np = np.array(Red_Channel)
        Green_Channel_np = np.array(Green_Channel)
        Blue_Channel_np = np.array(Blue_Channel)

        # fft of image
        fft_Red = fftpack.fftshift(fftpack.fft2(Red_Channel_np))
        fft_Green = fftpack.fftshift(fftpack.fft2(Green_Channel_np))
        fft_Blue = fftpack.fftshift(fftpack.fft2(Blue_Channel_np))

        # multiply both the images
        Filtered_Red_Channel = np.multiply(fft_Red, Band_Reject_Filter_np)
        Filtered_Green_Channel = np.multiply(fft_Green, Band_Reject_Filter_np)
        Filtered_Blue_Channel = np.multiply(fft_Blue, Band_Reject_Filter_np)

        # inverse fft to real number
        Inverse_Red_Channel = np.real(fftpack.ifft2(fftpack.ifftshift(Filtered_Red_Channel)))
        Inverse_Green_Channel = np.real(fftpack.ifft2(fftpack.ifftshift(Filtered_Green_Channel)))
        Inverse_Blue_Channel = np.real(fftpack.ifft2(fftpack.ifftshift(Filtered_Blue_Channel)))

        # find min and max color range
        Inverse_Red_Channel_Range = np.maximum(0, np.minimum(Inverse_Red_Channel, 255))
        Inverse_Green_Channel_Range = np.maximum(0, np.minimum(Inverse_Green_Channel, 255))
        Inverse_Blue_Channel_Range = np.maximum(0, np.minimum(Inverse_Blue_Channel, 255))

        # Change array to gray scale image
        Inverse_Red_Channel_Image = Image.fromarray(Inverse_Red_Channel_Range).convert("L")
        Inverse_Green_Channel_Image = Image.fromarray(Inverse_Green_Channel_Range).convert("L")
        Inverse_Blue_Channel_Image = Image.fromarray(Inverse_Blue_Channel_Range).convert("L")

        # merge 3 images
        Final_Image = Image.merge("RGB",
                                  (Inverse_Red_Channel_Image, Inverse_Green_Channel_Image, Inverse_Blue_Channel_Image))

        # save image
        Final_Image.save("'Image_without_periodic_noise_RGB.png'")

        print("Image Saved")
    else:
        # change to gray scale
        Original_Image = ImageOps.grayscale(Original_Image)

        # convert image to numpy array
        Original_Image_np = np.array(Original_Image)

        # fft of image
        fft_Original = fftpack.fftshift(fftpack.fft2(Original_Image_np))

        # multiply both the images
        Filtered_Original = np.multiply(fft_Original, Band_Reject_Filter_np)

        # inverse fft to real number
        Inverse_Original_Real = np.real(fftpack.ifft2(fftpack.ifftshift(Filtered_Original)))

        # find min and max color range
        Inverse_Original_Range = np.maximum(0, np.minimum(Inverse_Original_Real, 255))

        # save the image
        imageio.imsave('Image_without_periodic_noise_GS.png', Inverse_Original_Range.astype(np.uint8))

        print("Image Saved")


def Histogram_Equa():
    # 1. Open The Image & Get Image Pixels Matrix
    original_img = Image.open("Image_before_equalization.jpg")
    grayscale_img = ImageOps.grayscale(original_img)
    # 2. Get Histogram Frequencies
    image_histogram = grayscale_img.histogram()

    # 3. Calculate Cumulative Sequence
    def calc_cumulative(histogram_freq):
        new_list = [0] * len(histogram_freq)
        new_list[0] = histogram_freq[0]
        for i in range(1, len(histogram_freq)):
            new_list[i] = new_list[i - 1] + histogram_freq[i]
        return new_list

    # 4. Apply equalization math rule on histogram
    def apply_equalization(cumulative_list):
        x = 255 / (grayscale_img.width * grayscale_img.height)
        new_list = []
        for i in range(0, len(cumulative_list)):
            new_list.append(x * (cumulative_list[i]))
        return new_list

    # 5. Make the img
    def floor_list(img):
        new_list = []
        for i in range(len(img)):
            new_list.append(math.floor(img[i]))
        return new_list

    img_after_equa = floor_list(apply_equalization(calc_cumulative(image_histogram)))
    equalized_img = np.interp(grayscale_img, range(0, 256), img_after_equa)
    cv.imwrite("Image_After_Equalization.png", equalized_img)

    print("Image Saved")


def is_grayscale(imagee):

    if imagee.mode not in ("L", "RGB"):
        raise ValueError("Unsuported image mode")

    if imagee.mode == "RGB":
        rgb = imagee.split()
        if ImageChops.difference(rgb[0], rgb[1]).getextrema()[1] != 0:
            return False
        if ImageChops.difference(rgb[0], rgb[2]).getextrema()[1] != 0:
            return False
    return True


def display_Histo(img):
    val = is_grayscale(img)

    if (val):
        gray_img = ImageOps.grayscale(img)
        gray_img.thumbnail((400, 400))
        print(gray_img.histogram())
        histoFreq = gray_img.histogram()
        histoIndex = np.arange(256)
        plt.bar(x=histoIndex, height=histoFreq)
        plt.show()
    else:
        r, g, b = immmmg.split()
        hr = r.histogram()
        hg = g.histogram()
        hb = b.histogram()
        histoIR = np.arange(len(hr))
        histoIG = np.arange(len(hg))
        histoIB = np.arange(len(hb))
        plt.figure(figsize=(10, 10))
        plt.subplot(2, 2, 1)
        plt.title("R Histogram")
        plt.bar(histoIR, hr)
        plt.subplot(2, 2, 2)
        plt.title("G Histogram")
        plt.bar(histoIG, hg)
        plt.subplot(2, 2, 3)
        plt.title("B Histogram")
        plt.bar(histoIB, hb)
        plt.show()


def Filter_Function(Filter_Par, Size):
    Task_2_image = Image.open("Filter_Image1.jpg")
    Pixels = Task_2_image.load()
    filtered_image = Image.new("RGB", Task_2_image.size)
    draw_image = ImageDraw.Draw(filtered_image)
    for i in range(Size, Task_2_image.width - Size):
        for j in range(Size, Task_2_image.height - Size):
            colour_array = [0, 0, 0]
            for a in range(Size):
                for b in range(Size):
                    n = i + a - Size
                    m = j + b - Size
                    pixel = Pixels[n, m]
                    colour_array[0] += pixel[0] * Filter_Par[a][b]
                    colour_array[1] += pixel[1] * Filter_Par[a][b]
                    colour_array[2] += pixel[2] * Filter_Par[a][b]
            draw_image.point((i, j), (int(colour_array[0]), int(colour_array[1]), int(colour_array[2])))
    filtered_image.save("Image_After_Filter.png")

    print("Image Saved")


def Brightness_or_Darkness_Function(mode, value):
    Task_3_image = Image.open("Filter_Image1.jpg")
    Pixels = Task_3_image.load()
    Min_Value = 255
    Max_Value = 0
    rows = Task_3_image.size[0]
    columns = Task_3_image.size[1]

    if mode == "brightness":
        offset_value = value
    elif mode == "darkness":
        offset_value = -value

    Final_Image = Image.new("RGB", Task_3_image.size)
    Img_draw = ImageDraw.Draw(Final_Image)

    for ro in range(1, rows):
        for co in range(1, columns):
            pixel = Pixels[ro, co]
            Avarage = (pixel[0] + pixel[1] + pixel[2]) / 3
            New_value = Avarage + offset_value
            New_Point = (int((pixel[0] * New_value) / (Avarage + 1)), int((pixel[1] * New_value) / (Avarage + 1)),
                         int((pixel[2] * New_value) / (Avarage + 1)))
            Img_draw.point((ro, co), New_Point)
    Final_Image.save("Image_After_Brightness_or_Darkness.png")

    print("Image Saved")

while True :
    print("---------Computer Vision Project---------")
    print("1-Image segmentation")
    print("2-Band reject filter")
    print("3-Histogram Equalization")
    print("4-Display the Histogram")
    print("5-Apply a given filter")
    print("6-Brightness or Darkness")
    print("7-Exit")

    Option = input('Please Enter The Option Number: ')

    if Option == "1":
        K_Means()

    elif Option == "2":
        Band_Reject()

    elif Option == "3":
        Histogram_Equa()

    elif Option == "4":
        immmmg = Image.open("Filter_Image1.jpg")
        display_Histo(immmmg)

    elif Option == "5":
        Laplacian_Filter = [[0, -1, 0], [-1, 4, -1], [0, -1, 0]]
        Blur_Filter = [[1 / 9, 1 / 9, 1 / 9], [1 / 9, 1 / 9, 1 / 9], [1 / 9, 1 / 9, 1 / 9]]
        Sharpen_Filter = [[0, -0 / 5, 0], [-0 / 5, 3, -0 / 5], [0, -0 / 5, 0]]
        size = len(Blur_Filter)
        Filter_Function(Blur_Filter, size)

    elif Option == "6":
        Brightness_or_Darkness_Function("darkness", 100)

    elif Option == "7":
        print("Exiting!\n")
        break

    else:
        print("Unavailable Option")
        