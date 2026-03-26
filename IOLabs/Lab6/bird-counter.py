import os
import numpy as np
import cv2

folder_path = "bird_miniatures"
for file_name in os.listdir(folder_path):
    image = cv2.imread(folder_path + "\\" + file_name)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    normalized_image = cv2.normalize(gray_image, None, 0, 255, cv2.NORM_MINMAX)
    threshold, image_black_white = cv2.threshold(normalized_image, 90, 255, cv2.THRESH_BINARY_INV)
    kernel = np.ones((3, 3), np.uint8)
    dilated_image = cv2.dilate(image_black_white, kernel, iterations=1)
    contours, _ = cv2.findContours(dilated_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filtered_contours = [contour for contour in contours if cv2.contourArea(contour) > 1]
    quantity_of_birds = len(filtered_contours)
    print(f"{file_name} has {quantity_of_birds} birds.")
    output_image = cv2.drawContours(image.copy(), filtered_contours, -1, (0, 255, 0), 1)
    cv2.imshow(file_name, output_image)
    cv2.waitKey(0)
