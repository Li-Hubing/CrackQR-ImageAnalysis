import cv2
from PIL import Image
from pyzbar.pyzbar import decode
import numpy as np

def qrcode_decoding(image):
    # image = Image.open(image)
    decoded_objects = decode(image)
    if decoded_objects:
        for obj in decoded_objects:
            rect_points = obj.polygon
            src_points = rect_points[:4]
            coordinates = [(point.x, point.y) for point in src_points]
            return coordinates
        pass
    else:
        print("Decoding failed!")


def correct_image(qrcode_image, close_img, r):
    qrcode_img = Image.open(qrcode_image)
    qr_coordinates = qrcode_decoding(qrcode_img)
    l = qr_coordinates[2][0] - qr_coordinates[0][0]
    ratio = l / qrcode_img.size[0]

    source_coordinates = qrcode_decoding(close_img)
    x_values, y_values = zip(*source_coordinates)
    max_value = max(max(x_values) - min(x_values), max(y_values) - min(y_values))
    s = int(max_value / ratio)

    qrcode_img = qrcode_img.resize((s, s))
    qr_coordinates = qrcode_decoding(qrcode_img)
    while qr_coordinates is None:
        print("Adjusting the QR code scaling size...")
        s = int(s - 1)  # Adjust the reduction factor as needed
        qrcode_img = qrcode_img.resize((s, s))
        qr_coordinates = qrcode_decoding(qrcode_img)

    x_dis = source_coordinates[0][0] - qr_coordinates[0][0]
    y_dis = source_coordinates[0][1] - qr_coordinates[0][1]
    target_coordinates = [(x + x_dis, y + y_dis) for x, y in qr_coordinates]
    w, h = close_img.size
    close_img = cv2.cvtColor(np.array(close_img), cv2.COLOR_RGB2BGR)

    matrix = cv2.getPerspectiveTransform(np.array(source_coordinates, dtype=np.float32),
                                         np.array(target_coordinates, dtype=np.float32))

    image_coordinates = [(0, 0), (0, h), (w, h), (w, 0)]
    image_coordinates = np.array(image_coordinates, dtype=np.float32)
    transformed_coordinates = cv2.perspectiveTransform(image_coordinates.reshape(-1, 1, 2), matrix)
    transformed_coordinates_flat = transformed_coordinates.reshape(-1, 2)
    min_x = np.min(transformed_coordinates_flat[:, 0])
    min_y = np.min(transformed_coordinates_flat[:, 1])

    target_coordinates = [(x - min_x, y - min_y) for x, y in target_coordinates]
    matrix = cv2.getPerspectiveTransform(np.array(source_coordinates, dtype=np.float32),
                                         np.array(target_coordinates, dtype=np.float32))

    transformed_coordinates = cv2.perspectiveTransform(image_coordinates.reshape(-1, 1, 2), matrix)
    transformed_coordinates_flat = transformed_coordinates.reshape(-1, 2)
    max_x = np.max(transformed_coordinates_flat[:, 0])
    max_y = np.max(transformed_coordinates_flat[:, 1])

    nw = max_x // r + 1 if max_x % r > 0 else max_x // r
    nh = max_y // r + 1 if max_y % r > 0 else max_y // r

    corrected_img = cv2.warpPerspective(close_img, matrix, (int(nw * r), int(nh * r)))
    return corrected_img, s, max_x, max_y
