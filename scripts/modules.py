import os
import cv2
import numpy as np
from PIL import ImageDraw, Image
from scripts.batch_predict import batch_predict
from scripts.crack_parameters_calculation import calculation
from scripts.drawing_quadrilateral import points_coordinates_return
from scripts.image_processing import find_qr_codes_in_image,erase_noise_interactively,image_crops,image_combination
from scripts.qrcode_processing import qrcode_decoding,correct_image
from torchvision.transforms import functional as F


def segmentation(medium_image, r, result_dir):
    medium_padding_image = os.path.join(result_dir, 'medium_padding.png')
    image_patches_file = os.path.join(result_dir, 'image_patches')
    if not os.path.exists(image_patches_file):
        os.mkdir(image_patches_file)
    patches_prediction_file = os.path.join(result_dir, 'patches_prediction')
    if not os.path.exists(patches_prediction_file):
        os.mkdir(patches_prediction_file)
    padding_prediction_image = os.path.join(result_dir, 'padding_prediction.png')
    model_prediction_image = os.path.join(result_dir, 'model_prediction.png')

    # (a).crops images
    print('cropping the image...')
    medium_img = Image.open(medium_image)
    original_W, original_H = medium_img.size
    nw = original_W // r + 1 if original_W % r > 0 else original_W // r
    nh = original_H // r + 1 if original_H % r > 0 else original_H // r
    padw = nw * r - original_W
    padh = nh * r - original_H
    medium_img_padding = F.pad(medium_img, (0, 0, padw, padh), fill=(0, 0, 0))
    medium_img_padding.save(medium_padding_image)
    resized_W, resized_H = medium_img_padding.size
    image_crops(medium_img_padding, r, image_patches_file)
    # =================================================================
    # (b).predict
    print('detecting cracks...')
    if not os.path.exists(patches_prediction_file):
        os.mkdir(patches_prediction_file)
    batch_predict(image_patches_file, patches_prediction_file)
    prediction_padding_img = image_combination(patches_prediction_file, resized_W, resized_H, r)
    prediction_padding_img.save(padding_prediction_image)
    model_prediction_img = prediction_padding_img.crop([0, 0, original_W, original_H])
    model_prediction_img.save(model_prediction_image)
    return medium_img


def correction_segmentation(medium_image, r, qrcode_image, result_dir, qr_image_real_length):
    crops_file = os.path.join(result_dir, 'image_patches')
    if not os.path.exists(crops_file):
        os.mkdir(crops_file)
    patches_prediction_file = os.path.join(result_dir, 'patches_prediction')
    if not os.path.exists(patches_prediction_file):
        os.mkdir(patches_prediction_file)

    rectified_image = os.path.join(result_dir, 'correction.png')
    model_prediction_image = os.path.join(result_dir, 'model_prediction.png')
    crop_rectified_image = os.path.join(result_dir, 'cropped_correction.png')

    # # # ************************ Part 1: Image processing ***************************
    print('image processing...')
    medium_img = Image.open(medium_image)
    corrected_img, s, max_x, max_y = correct_image(qrcode_image, medium_img, r)
    cv2.imwrite(rectified_image, corrected_img)
    pixel_coefficient = qr_image_real_length / s
    # # # ******************************* Part 2: Segmentation ***************************
    # print('crops images...')
    corrected_img = Image.fromarray(cv2.cvtColor(corrected_img, cv2.COLOR_BGR2RGB))
    rectified_W, rectified_H = corrected_img.size
    image_crops(corrected_img, r, crops_file)
    crop_rectified_img = corrected_img.crop((0, 0, max_x, max_y))
    crop_rectified_img.save(crop_rectified_image)

    # print('Predicting...')
    batch_predict(crops_file, patches_prediction_file)
    padding_prediction_img = image_combination(patches_prediction_file, rectified_W, rectified_H, r)
    model_prediction_img = padding_prediction_img.crop((0, 0, max_x, max_y))
    model_prediction_img.save(model_prediction_image)

    return crop_rectified_image, crop_rectified_img, pixel_coefficient


def correction(image, r, qrcode_image, result_dir, qr_image_real_length):
    crop_corrected_image = os.path.join(result_dir, 'cropped_correction.png')
    print('image processing...')
    medium_img = Image.open(image)
    corrected_img, s, max_x, max_y = correct_image(qrcode_image, medium_img, r)
    corrected_img = Image.fromarray(cv2.cvtColor(corrected_img, cv2.COLOR_BGR2RGB))

    cropped_corrected_img = corrected_img.crop((0, 0, max_x, max_y))
    cropped_corrected_img.save(crop_corrected_image)
    pixel_coefficient = qr_image_real_length / s
    return crop_corrected_image, pixel_coefficient


def remove_noise(reference, result_dir, image_downscale_factor):
    model_prediction_image = os.path.join(result_dir, 'model_prediction.png')
    inital_remove_noise_image = os.path.join(result_dir, 'first_noise_erasure.png')
    prediction_image = os.path.join(result_dir, 'prediction.png')
    prediction_dilation_image = os.path.join(result_dir, 'dilated_predction.png')

    tips = "Click and drag to erase non-crack noise. Press 'R' to reset. Press 'ESC' to proceed."
    inital_remove_noise_img = erase_noise_interactively(reference, model_prediction_image, tips, 
                                                        image_downscale_factor, mouse_radius=100)
    cv2.imwrite(inital_remove_noise_image, inital_remove_noise_img)
    prediction_img = erase_noise_interactively(reference, inital_remove_noise_image, tips,
                                               image_downscale_factor, mouse_radius=10)
    cv2.imwrite(prediction_image, prediction_img)
    prediction_white_crack_img = 255 - prediction_img 
    kernel = np.ones((5, 5), np.uint8)
    prediction_dilation_img = cv2.dilate(prediction_white_crack_img, kernel, iterations=5)
    prediction_black_crack_img = 255 - prediction_dilation_img
    cv2.imwrite(prediction_dilation_image, prediction_black_crack_img)
    return prediction_white_crack_img, prediction_dilation_img


def localization(low_image, crop_rectified_img, prediction_dilation_img, result_dir, image_downscale_factor):
    window_size = 1000
    step = 800
    qr_patch_image = os.path.join(result_dir, 'qr_patch.jpg')
    outline_image = os.path.join(result_dir, 'outline.png')
    transformed_crack_image = os.path.join(result_dir, 'transformed_crack.png')
    localization_image = os.path.join(result_dir, 'localization.png')

    print('locating cracks...')
    # decode QR code information from low resolution image
    qr_codes = find_qr_codes_in_image(low_image, (window_size, window_size), step)
    cropped_image_coordinates = qr_codes[3]

    left, top, right, bottom = qr_codes[0], qr_codes[1], qr_codes[0] + window_size, qr_codes[1] + window_size
    # print(left, top)
    low_img = Image.open(low_image)
    qr_patch_img = low_img.crop((left, top, right, bottom))  # decoding failed when large image
    qr_patch_img.save(qr_patch_image)

    qr_in_low_img_coordinates = [(x + qr_codes[0], y + qr_codes[1]) for x, y in cropped_image_coordinates]
    qr_in_medium_img_coordinates = qrcode_decoding(crop_rectified_img)

    # draw the contour of the component with cracks
    tips1 = "Select 4 corners of the component in order (TL-BL-BR-TR). Drag to adjust. Press 'ESC' to proceed."
    component_corner_points = points_coordinates_return(low_image, tips1, image_downscale_factor)
    image_width, image_height = low_img.size
    background_color = (255, 255, 255)
    localization_img = Image.new("RGB", (image_width, image_height), background_color)
    draw = ImageDraw.Draw(localization_img)
    for i in range(4):
        start_point = component_corner_points[i]
        end_point = component_corner_points[(i + 1) % 4]
        draw.line([start_point, end_point], fill=(0, 0, 0), width=10)

    tips2 = "Select 4 corners of the window in order (TL-BL-BR-TR). Drag to adjust. Press 'ESC' to proceed."
    opening_corner_points = points_coordinates_return(low_image, tips2, image_downscale_factor)
    if len(opening_corner_points) > 0:
        for i in range(4):
            start_point = opening_corner_points[i]
            end_point = opening_corner_points[(i + 1) % 4]
            draw.line([start_point, end_point], fill=(0, 0, 255), width=10)
    localization_img.save(outline_image)

    qr_in_medium_img_coordinates = np.array(qr_in_medium_img_coordinates, dtype=np.float32)
    qr_in_low_img_coordinates = np.array(qr_in_low_img_coordinates, dtype=np.float32)
    M = cv2.getPerspectiveTransform(qr_in_medium_img_coordinates, qr_in_low_img_coordinates)

    transformed_crack_img = cv2.warpPerspective(prediction_dilation_img, M, (image_width, image_height))
    cv2.imwrite(transformed_crack_image, 255 - transformed_crack_img)
    # # ------------------------------------------------------------------------------------------------
    # draw a contour image with cracks
    transformed_crack_img = np.array(transformed_crack_img)
    for x in range(image_width):
        for y in range(image_height):
            pixel_color = tuple(transformed_crack_img[y, x])
            if pixel_color == (255, 255, 255):
                draw.point((x, y), fill=(255, 0, 0))
    localization_img.save(localization_image)
    return component_corner_points, M, localization_img


def measurement(pixel_coefficient, prediction_image, component_corner_points, M, localization_img, result_dir, rectangle_aspect_ratio):
    print('measuring crack parameters...')
    width, length, area, angle = calculation(prediction_image, component_corner_points, M, localization_img,
                                             result_dir, rectangle_aspect_ratio)
    print('crack measurement results')
    print(f'pixel calibration coefficient: {round(pixel_coefficient, 3)} mm')
    print(f'width: {round(width * pixel_coefficient, 3)} mm')
    print(f'length: {round(length * pixel_coefficient, 3)} mm')
    print(f'area: {round(area * pixel_coefficient ** 2, 3)} mm²')
    print(f"angle: {round(angle, 3)}°")

    with open(os.path.join(result_dir, 'measurement_results.txt'), 'w', encoding='utf-8') as file:
        file.write(f'pixel calibration coefficient: {round(pixel_coefficient, 3)} mm\n')
        file.write(f'width: {round(width * pixel_coefficient, 3)} mm\n')
        file.write(f'length: {round(length * pixel_coefficient, 3)} mm\n')
        file.write(f'area: {round(area * pixel_coefficient ** 2, 3)} mm²\n')
        file.write(f"angle: {round(angle, 3)}°\n")


def wla_measurement(pixel_coefficient, prediction_white_crack_img, result_dir):
    print('measuring crack parameters...')
    grey_pred = cv2.cvtColor(prediction_white_crack_img, cv2.COLOR_BGR2GRAY)
    dist_transform = cv2.distanceTransform(grey_pred, cv2.DIST_L2, 3)
    dist2 = cv2.normalize(dist_transform, None, 255, 0, cv2.NORM_MINMAX, cv2.CV_8UC1)
    heat_img = cv2.applyColorMap(dist2, cv2.COLORMAP_JET)
    cv2.imwrite(os.path.join(result_dir, 'width_heat.png'), heat_img)
    max_dist = np.max(dist_transform)
    max_dist_coords = np.unravel_index(np.argmax(dist_transform), dist_transform.shape)
    output_img = cv2.cvtColor(255 - grey_pred, cv2.COLOR_GRAY2BGR)
    cv2.circle(output_img, (max_dist_coords[1], max_dist_coords[0]), int(max_dist), (0, 0, 255), 2)
    cv2.circle(output_img, (max_dist_coords[1], max_dist_coords[0]), 0, (0, 0, 255), -1)
    cv2.imwrite(os.path.join(result_dir, 'width.png'), output_img)
    width = 2 * max_dist

    img_skeleton = cv2.ximgproc.thinning(grey_pred)
    img_pil = Image.fromarray(255 - img_skeleton).convert("1")
    img_pil.save(os.path.join(result_dir, 'length.png'))

    length = 0
    white_pixels = np.argwhere(img_skeleton)
    for i in range(1, len(white_pixels)):
        length += np.linalg.norm(white_pixels[i] - white_pixels[i - 1])

    image = Image.fromarray(grey_pred)
    w, h = image.size
    image1 = np.array(image).flatten()
    assert h * w == np.sum(image1 == 0) + np.sum(image1 == 255)
    # # ------------------------------------------------------------------------------------------------
    image = image.convert("1")
    image = np.array(image).flatten()
    area = np.sum(image == 1)
    print('crack measurement results')
    print(f'pixel calibration coefficient: {round(pixel_coefficient, 3)} mm')
    print(f'width: {round(width * pixel_coefficient, 3)} mm')
    print(f'length: {round(length * pixel_coefficient, 3)} mm')
    print(f'area: {round(area * pixel_coefficient ** 2, 3)} mm²')

    with open(os.path.join(result_dir, 'measurement_results.txt'), 'w', encoding='utf-8') as file:
        file.write(f'pixel calibration coefficient: {round(pixel_coefficient, 3)} mm\n')
        file.write(f'width: {round(width * pixel_coefficient, 3)} mm\n')
        file.write(f'length: {round(length * pixel_coefficient, 3)} mm\n')
        file.write(f'area: {round(area * pixel_coefficient ** 2, 3)} mm²\n')


def width_measurement(pixel_coefficient, prediction_white_crack_img, result_dir):
    grey_pred = cv2.cvtColor(prediction_white_crack_img, cv2.COLOR_BGR2GRAY)
    dist_transform = cv2.distanceTransform(grey_pred, cv2.DIST_L2, 3)
    dist2 = cv2.normalize(dist_transform, None, 255, 0, cv2.NORM_MINMAX, cv2.CV_8UC1)
    heat_img = cv2.applyColorMap(dist2, cv2.COLORMAP_JET)
    cv2.imwrite(os.path.join(result_dir, 'width_heat.png'), heat_img)
    max_dist = np.max(dist_transform)
    max_dist_coords = np.unravel_index(np.argmax(dist_transform), dist_transform.shape)
    output_img = cv2.cvtColor(255 - grey_pred, cv2.COLOR_GRAY2BGR)
    cv2.circle(output_img, (max_dist_coords[1], max_dist_coords[0]), int(max_dist), (0, 0, 255), 2)
    cv2.circle(output_img, (max_dist_coords[1], max_dist_coords[0]), 0, (0, 0, 255), -1)
    cv2.imwrite(os.path.join(result_dir, 'width.png'), output_img)
    width = 2 * max_dist
    print(f'pixel calibration coefficient: {round(pixel_coefficient, 3)} mm')
    print(f'width: {round(width * pixel_coefficient, 3)} mm')

    with open(os.path.join(result_dir, 'measurement_results.txt'), 'w', encoding='utf-8') as file:
        file.write(f'pixel calibration coefficient: {round(pixel_coefficient, 3)} mm\n')
        file.write(f'crack width: {round(width * pixel_coefficient, 3)} mm\n')


def extract_information_from_txt(processed_file):
    with open(os.path.join(processed_file, 'measurement_results.txt'), 'r', encoding='utf-8') as file:
        for line in file:
            if 'pixel calibration coefficient:' in line:
                accuracy_str = line.split(':')[-1].strip()
                pixel_coefficient = float(accuracy_str.split(' ')[0])
    return pixel_coefficient
