import os.path
import cv2
import numpy as np
from PIL import ImageDraw, Image


def calculation(image_path, corner_points, M, localization_img, save_path, rectangle_aspect_ratio):
    cv_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    cv_image = 255 - cv_image

    dist_transform = cv2.distanceTransform(cv_image, cv2.DIST_L2, 3)
    dist2 = cv2.normalize(dist_transform, None, 255, 0, cv2.NORM_MINMAX, cv2.CV_8UC1)
    heat_img = cv2.applyColorMap(dist2, cv2.COLORMAP_JET)
    cv2.imwrite(os.path.join(save_path, 'width_heat.png'), heat_img)
    max_dist = np.max(dist_transform)
    max_dist_coords = np.unravel_index(np.argmax(dist_transform), dist_transform.shape)
    output_img = cv2.cvtColor(255 - cv_image, cv2.COLOR_GRAY2BGR)
    cv2.circle(output_img, (max_dist_coords[1], max_dist_coords[0]), int(max_dist), (0, 0, 255), 2)
    cv2.circle(output_img, (max_dist_coords[1], max_dist_coords[0]), 0, (0, 0, 255), -1)
    cv2.imwrite(os.path.join(save_path, 'width.png'), output_img)
    width = 2 * max_dist
    # print("Width pixels:", width)

    img_skeleton = cv2.ximgproc.thinning(cv_image)
    img_pil = Image.fromarray(255 - img_skeleton).convert("1")
    img_pil.save(os.path.join(save_path, 'length.png'))

    length = 0
    white_pixels = np.argwhere(img_skeleton)
    for i in range(1, len(white_pixels)):
        length += np.linalg.norm(white_pixels[i] - white_pixels[i - 1])

    image = Image.fromarray(cv_image)
    w, h = image.size
    image1 = np.array(image).flatten()
    assert h * w == np.sum(image1 == 0) + np.sum(image1 == 255)
    # ------------------------------------------------------------------------------------------------
    image = image.convert("1")
    image = np.array(image).flatten()
    area = np.sum(image == 1)

    white_pixels = np.where(cv_image == 255)
    white_pixels = np.transpose(white_pixels)

    x, y = white_pixels[:, 1], white_pixels[:, 0]
    A = np.vstack([x, np.ones(len(x))]).T
    m, c = np.linalg.lstsq(A, y, rcond=None)[0]

    fit_line_y1 = int(m * 0 + c)
    fit_line_y2 = int(m * (cv_image.shape[1] - 1) + c)
    cv_image_with_line = cv2.cvtColor(255 - cv_image, cv2.COLOR_GRAY2BGR)
    cv2.line(cv_image_with_line, (0, fit_line_y1), (cv_image.shape[1] - 1, fit_line_y2), (0, 0, 255), 5)
    cv2.imwrite(os.path.join(save_path, 'fitted_line.png'), cv_image_with_line)
    # ------------------------------------------------------------------------------------------------
    # S1: Rectify the perspective-distorted component into a rectangle.

    # S11: Initial perspective correction
    x_values, y_values = zip(*corner_points)
    if (max(x_values) - min(x_values)) * rectangle_aspect_ratio > (max(y_values) - min(y_values)):
        W_ = max(x_values) - min(x_values)
        H_ = (W_) * rectangle_aspect_ratio
    else:
        H_ = max(y_values) - min(y_values)
        W_ = (H_) / rectangle_aspect_ratio

    target_coordinates1 = [(corner_points[0][0], corner_points[0][1]), 
                          (corner_points[0][0], corner_points[0][1] + H_),
                          (corner_points[0][0] + W_, corner_points[0][1] + H_), 
                          (corner_points[0][0] + W_, corner_points[0][1])]

    matrix1 = cv2.getPerspectiveTransform(np.array(corner_points, dtype=np.float32),
                                         np.array(target_coordinates1, dtype=np.float32))
    
    # S12: Final perspective correction
    w, h = localization_img.size
    image_coordinates = np.array([(0, 0), (0, h), (w, h), (w, 0)], dtype=np.float32)
    transformed_coordinates = cv2.perspectiveTransform(image_coordinates.reshape(-1, 1, 2), matrix1)
    transformed_coordinates_flat = transformed_coordinates.reshape(-1, 2)
    min_x = np.min(transformed_coordinates_flat[:, 0])
    min_y = np.min(transformed_coordinates_flat[:, 1])

    target_coordinates2 = [(x - min_x, y - min_y) for x, y in target_coordinates1]
    matrix2 = cv2.getPerspectiveTransform(np.array(corner_points, dtype=np.float32),
                                         np.array(target_coordinates2, dtype=np.float32))

    transformed_coordinates = cv2.perspectiveTransform(image_coordinates.reshape(-1, 1, 2), matrix2)
    transformed_coordinates_flat = transformed_coordinates.reshape(-1, 2)
    max_x = np.max(transformed_coordinates_flat[:, 0])
    max_y = np.max(transformed_coordinates_flat[:, 1])

    localization_img_bgr = cv2.cvtColor(np.array(localization_img), cv2.COLOR_RGB2BGR)
    corrected_img_bgr = cv2.warpPerspective(localization_img_bgr, matrix2, (int(max_x), int(max_y)))
    corrected_img_rbg = Image.fromarray(cv2.cvtColor(corrected_img_bgr, cv2.COLOR_BGR2RGB))

    # S2: Angle calculation
    component_top_line_points = np.array([corner_points[0], corner_points[3]], dtype=np.float32)
    new_component_top_line_points = cv2.perspectiveTransform(component_top_line_points.reshape(-1, 1, 2), matrix2).reshape(-1, 2)
    new_component_top_line_points = [(int(x), int(y)) for (x,y) in new_component_top_line_points]

    fit_line_points = np.array([[0, fit_line_y1], [cv_image.shape[1] - 1, fit_line_y2]], dtype=np.float32)
    transformed_fit_line_points = cv2.perspectiveTransform(fit_line_points.reshape(-1, 1, 2), M).reshape(2, 2)
    new_fit_line_points = cv2.perspectiveTransform(transformed_fit_line_points.reshape(-1, 1, 2), matrix2).reshape(-1, 2)
    new_fit_line_points = [(int(x), int(y)) for (x,y) in new_fit_line_points]

    draw = ImageDraw.Draw(corrected_img_rbg)
    draw.line([new_component_top_line_points[0], new_component_top_line_points[1]], fill=(112, 173, 71), width=10)
    draw.line([new_fit_line_points[0], new_fit_line_points[1]], fill=(112, 173, 71), width=10)

    corrected_img_rbg.save(os.path.join(save_path, 'angle.png'))

    angle_fitted_transformed = np.arctan2(
        new_fit_line_points[0][1] - new_fit_line_points[1][1],
        new_fit_line_points[1][0] - new_fit_line_points[0][0])

    return width, length, area, np.degrees(angle_fitted_transformed)
