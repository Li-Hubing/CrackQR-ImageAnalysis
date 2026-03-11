import os
import cv2
from PIL import Image
import numpy as np


def image_crops(image, r, save_path):
    H, W = image.size
    for i in range(int(H / r)):
        for j in range(int(W / r)):
            image_crop = image.crop((i * r, j * r, r * (i + 1), r * (j + 1)))
            image_crop.save(save_path + "/" + str(i) + "_" + str(j) + ".png")


def image_combination(prediction_path, resized_W, resized_H, r):
    w = int(resized_W / r)
    h = int(resized_H / r)
    new_image = np.zeros((resized_H, resized_W), dtype=np.uint8)

    for x in range(w):
        for y in range(h):
            image_path = os.path.join(prediction_path, f"{x}_{y}.png")
            image = Image.open(image_path)
            paste_x = x * r
            paste_y = y * r
            new_image[paste_y:paste_y + r, paste_x:paste_x + r] = image

    new_image_pil = Image.fromarray(new_image)
    return new_image_pil


def mouse_callback(event, x, y, flags, param):
    global clicked_points
    if event == cv2.EVENT_LBUTTONDOWN:
        # print(f"Clicked at ({x}, {y})")
        param.append((x, y))


def measure_distance(image_path, pixel_coefficient):
    def mouse_callback(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            if len(clicked_points) < 2:
                param.append((x, y))

    def reset_points():
        del clicked_points[:]

    name = "Click on two points to measure distance. Press 'R' to reset points."
    image = cv2.imread(image_path)

    # cv2.namedWindow(name)  # It can maintain the image aspect ratio, but may not display the entire image.
    cv2.namedWindow(name,cv2.WINDOW_KEEPRATIO)  # It can fit the entire image within the window, but cannot preserve the original image dimensions.

    clicked_points = []
    cv2.setMouseCallback(name, mouse_callback, clicked_points)

    while True:
        img_copy = image.copy()
        for point in clicked_points:
            cv2.circle(img_copy, point, 0, (0, 0, 255), -1)
        if len(clicked_points) == 2:
            cv2.line(img_copy, clicked_points[0], clicked_points[1], (0, 0, 255), 5)
            point1 = np.array(clicked_points[0])
            point2 = np.array(clicked_points[1])
            distance = np.linalg.norm(point1 - point2)
            lines = [
                f"Pixel calibration coefficient: {pixel_coefficient} mm/pixel",
                f'distance in image: {round(distance, 3)} pixels',
                f'measured distance: {round(distance * pixel_coefficient, 3)} mm'
            ]

            window_name = f'{lines[0]}, {lines[1]}, {lines[2]}.'
            cv2.setWindowTitle(name, window_name)

        cv2.imshow(name, img_copy)
        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            break
        if key == ord('r'):  # Press 'R' to reset points
            reset_points()
    cv2.destroyAllWindows()


def erase_noise_interactively(reference_path, image_path, name, image_downscale_factor, mouse_radius):
    original_image = cv2.imread(image_path)
    original_reference = cv2.imread(reference_path)

    height, width, _ = original_image.shape

    new_width = int(width / image_downscale_factor)
    new_height = int(height / image_downscale_factor)

    image = cv2.resize(original_image, (new_width, new_height), interpolation=cv2.INTER_AREA)
    reference = cv2.resize(original_reference, (new_width, new_height), interpolation=cv2.INTER_AREA)

    height, width, _ = image.shape
    background_color = 255
    spacing = np.ones((height, 20, 3), dtype=np.uint8) * 100
    adjusted_x = 0
    clicked_positions = []

    def reset():
        nonlocal image
        image = cv2.resize(original_image, (new_width, new_height), interpolation=cv2.INTER_AREA)
        clicked_positions.clear()
        cv2.imshow(name, np.hstack((reference, spacing, image)))

    def draw_circle(event, x, y, flags, param):
        nonlocal image
        global mask
        nonlocal adjusted_x
        if event == cv2.EVENT_MOUSEMOVE:
            mask = np.zeros((height, width), dtype=np.uint8)
            adjusted_x = x - reference.shape[1] - spacing.shape[1]

            # cv2.circle(mask, (x, y), mouse_radius, 255, -1)
            cv2.circle(mask, (adjusted_x, y), mouse_radius, 255, -1)
            white = np.ones_like(image) * background_color

            white_circle = cv2.bitwise_and(white, white, mask=mask)

            result = cv2.bitwise_and(image, image, mask=cv2.bitwise_not(mask))
            result = cv2.add(result, white_circle)

            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            result = cv2.drawContours(result, contours, -1, (0, 0, 255), 1)

            # cv2.imshow(name, result)
            cv2.imshow(name, np.hstack((reference, spacing, result)))
        if event == cv2.EVENT_LBUTTONDOWN or event == cv2.EVENT_MOUSEMOVE and flags == cv2.EVENT_FLAG_LBUTTON:
            cv2.circle(mask, (adjusted_x, y), mouse_radius, 255, -1)
            white = np.ones_like(image) * background_color
            white_circle = cv2.bitwise_and(white, white, mask=mask)
            clicked_positions.append((adjusted_x, y))
            image = cv2.bitwise_and(image, image, mask=cv2.bitwise_not(mask))
            image = cv2.add(image, white_circle)

            mask.fill(0)
            # cv2.imshow(name, image)
            cv2.imshow(name, np.hstack((reference, spacing, image)))
    cv2.namedWindow(name)
    cv2.setMouseCallback(name, draw_circle)

    # cv2.imshow(name, image)
    cv2.imshow(name, np.hstack((reference, spacing, image)))
    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            break
        if key == ord('r'):  # Press 'R' to reset points
            reset()

    cv2.destroyAllWindows()
    for pos in clicked_positions:
        x, y = pos
        cv2.circle(original_image,
                    (int(x * image_downscale_factor), int(y * image_downscale_factor)),
                    int(mouse_radius * image_downscale_factor), (255, 255, 255), -1)
    return original_image


def sliding_window(image, step, window_size):
    for y in range(0, image.shape[0], step):
        for x in range(0, image.shape[1], step):
            yield (x, y, image[y:y + window_size[1], x:x + window_size[0]])


from pyzbar.pyzbar import decode

def find_qr_codes_in_image(image_path, window_size, step):
    image = cv2.imread(image_path)
    qr_codes = []
    qr_code_found = False
    for (x, y, window) in sliding_window(image, step, window_size):
        # Convert the window to grayscale
        window_gray = cv2.cvtColor(window, cv2.COLOR_BGR2GRAY)

        # Try to decode any QR codes in the window
        decoded_objects = decode(window_gray)
        if decoded_objects:
            for obj in decoded_objects:
                rect_points = obj.polygon
                src_points = rect_points[:4]
                coordinates = [(point.x, point.y) for point in src_points]
                qr_code_data = obj.data.decode('utf-8')
                qr_codes = [x, y, qr_code_data, coordinates]
            qr_code_found = True
            break
    return qr_codes



