import cv2
import matplotlib.pyplot as plt
from matplotlib.pyplot import subplot

first_image = cv2.imread("Tree.png")
first_image = cv2.cvtColor(first_image, cv2.COLOR_BGR2RGB)

first_gray = cv2.imread("Tree.png")
first_gray = cv2.cvtColor(first_gray, cv2.COLOR_BGR2GRAY)

first_gray2 = cv2.imread("Tree.png")
width, height = first_gray2.shape[0:2]
for x in range(width):
    for y in range(height):
        first_gray2[x, y] = round(sum(first_gray2[x, y]) / 3)

plt,subplot(1, 3, 1)
plt.imshow(first_image)
plt.title("Original")
plt.subplot(1, 3, 2)
plt.imshow(first_gray, cmap=plt.get_cmap('gray'))
plt.title("Natural Grayscale")
plt.subplot(1, 3, 3)
plt.imshow(first_gray2)
plt.title("Non-natural Greyscale")
plt.show()

second_image = cv2.imread("Zebra.png")
second_image = cv2.cvtColor(second_image, cv2.COLOR_BGR2RGB)

second_gray = cv2.imread("Zebra.png")
second_gray = cv2.cvtColor(second_gray, cv2.COLOR_BGR2GRAY)

second_gray2 = cv2.imread("Zebra.png")
width, height = second_gray2.shape[0:2]
for x in range(width):
    for y in range(height):
        second_gray2[x, y] = sum(second_gray2[x, y]) / 3

plt,subplot(1, 3, 1)
plt.imshow(second_image)
plt.title("Original")
plt.subplot(1, 3, 2)
plt.imshow(second_gray, cmap=plt.get_cmap('gray'))
plt.title("Natural Grayscale")
plt.subplot(1, 3, 3)
plt.imshow(second_gray2)
plt.title("Non-natural Greyscale")
plt.show()

third_image = cv2.imread("Chinese.png")
third_image = cv2.cvtColor(third_image, cv2.COLOR_BGR2RGB)

third_gray = cv2.imread("Chinese.png")
third_gray = cv2.cvtColor(third_gray, cv2.COLOR_BGR2GRAY)

third_gray2 = cv2.imread("Chinese.png")
width, height = third_gray2.shape[0:2]
for x in range(width):
    for y in range(height):
        third_gray2[x, y] = sum(third_gray2[x, y]) / 3

plt,subplot(1, 3, 1)
plt.imshow(third_image)
plt.title("Original")
plt.subplot(1, 3, 2)
plt.imshow(third_gray, cmap=plt.get_cmap('gray'))
plt.title("Natural Grayscale")
plt.subplot(1, 3, 3)
plt.imshow(third_gray2)
plt.title("Non-natural Greyscale")
plt.show()

fourth_image = cv2.imread("Mountains.png")
fourth_image = cv2.cvtColor(fourth_image, cv2.COLOR_BGR2RGB)

fourth_gray = cv2.imread("Mountains.png")
fourth_gray = cv2.cvtColor(fourth_gray, cv2.COLOR_BGR2GRAY)

fourth_gray2 = cv2.imread("Mountains.png")
width, height = fourth_gray2.shape[0:2]
for x in range(width):
    for y in range(height):
        fourth_gray2[x, y] = sum(fourth_gray2[x, y]) / 3

plt,subplot(1, 3, 1)
plt.imshow(fourth_image)
plt.title("Original")
plt.subplot(1, 3, 2)
plt.imshow(fourth_gray, cmap=plt.get_cmap('gray'))
plt.title("Natural Grayscale")
plt.subplot(1, 3, 3)
plt.imshow(fourth_gray2)
plt.title("Non-natural Greyscale")
plt.show()

fifth_image = cv2.imread("New York.png")
fifth_image = cv2.cvtColor(fifth_image, cv2.COLOR_BGR2RGB)

fifth_gray = cv2.imread("New York.png")
fifth_gray = cv2.cvtColor(fifth_gray, cv2.COLOR_BGR2GRAY)

fifth_gray2 = cv2.imread("New York.png")
width, height = fifth_gray2.shape[0:2]
for x in range(width):
    for y in range(height):
        fifth_gray2[x, y] = sum(fifth_gray2[x, y]) / 3

plt,subplot(1, 3, 1)
plt.imshow(fifth_image)
plt.title("Original")
plt.subplot(1, 3, 2)
plt.imshow(fifth_gray, cmap=plt.get_cmap('gray'))
plt.title("Natural Grayscale")
plt.subplot(1, 3, 3)
plt.imshow(fifth_gray2)
plt.title("Non-natural Greyscale")
plt.show()
