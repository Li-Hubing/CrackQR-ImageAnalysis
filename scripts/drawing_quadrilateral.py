import cv2


def draw_quadrilateral(image, points):
    if len(points) == 4:
        for i in range(4):
            cv2.line(image, points[i], points[(i + 1) % 4], (0, 0, 255), 2)
    for point in points:
        cv2.circle(image, point, 3, (0, 0, 255), -1)


def points_coordinates_return(image_path, tips, image_downscale_factor):
    clicked_points = []
    image = cv2.imread(image_path)

    height, width, _ = image.shape
    new_width = int(width / image_downscale_factor)
    new_height = int(height / image_downscale_factor)

    display_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
    cv2.namedWindow(tips)
    cv2.setMouseCallback(tips,
                         lambda event, x, y, flags, param: mouse_callback(event, x, y, flags, param, clicked_points))

    while True:
        image_copy = display_image.copy()
        draw_quadrilateral(image_copy, clicked_points)
        cv2.imshow(tips, image_copy)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # Press Esc key to exit
            break

    cv2.destroyAllWindows()
    
    actual_points = [(int(x*image_downscale_factor), int(y*image_downscale_factor))for (x,y) in clicked_points]
    return actual_points


def mouse_callback(event, x, y, flags, param, clicked_points):
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(clicked_points) < 4:
            clicked_points.append((x, y))

    elif event == cv2.EVENT_MOUSEMOVE and flags == cv2.EVENT_FLAG_LBUTTON:
        if len(clicked_points) == 4:
            distances = [((x - px) ** 2 + (y - py) ** 2) for px, py in clicked_points]
            moving_point = distances.index(min(distances))
            clicked_points[moving_point] = (x, y)


