from scripts.modules import *
from .image_processing import measure_distance
import datetime


class ImageProcessor:
    def __init__(self, qrcode_image=None, qr_image_real_size=120, result_dir=None, image_downscale_factor=3):
        self.image_downscale_factor = image_downscale_factor
        self.image_dir = 'images'
        self.qrcode_image = qrcode_image
        self.qr_image_real_size = qr_image_real_size
        if result_dir == None:
            record_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            self.result_dir = 'results/{}'.format(record_time)
            os.makedirs(self.result_dir, exist_ok=True) 
        else:
            self.result_dir = result_dir

        self.LR_image = os.path.join(self.image_dir, 'LR.jpg')
        self.MR_image = os.path.join(self.image_dir, 'MR.jpg')
        self.CU_image = os.path.join(self.image_dir, 'CU.jpg')
        self.prediction_image = os.path.join(self.result_dir, 'prediction.png')

    def correction_localization_measurement(self, rectangle_aspect_ratio):
        crop_rectified_image, cropped_corrected_img, pixel_coefficient = correction_segmentation(
            self.MR_image, 512, self.qrcode_image, self.result_dir, self.qr_image_real_size)

        prediction_white_crack_img, prediction_dilation_img = remove_noise(crop_rectified_image, self.result_dir, self.image_downscale_factor)

        component_corner_points, M, localization_img = localization(self.LR_image, cropped_corrected_img,
                                                                    prediction_dilation_img, self.result_dir, self.image_downscale_factor)

        measurement(pixel_coefficient, self.prediction_image, component_corner_points, M, localization_img,
                    self.result_dir, rectangle_aspect_ratio)

    def parameter_measurement(self):
        crop_rectified_image, cropped_corrected_img, pixel_coefficient = correction_segmentation(
            self.MR_image, 512, self.qrcode_image, self.result_dir, self.qr_image_real_size)

        prediction_white_crack_img, _ = remove_noise(crop_rectified_image, self.result_dir, self.image_downscale_factor)

        wla_measurement(pixel_coefficient, prediction_white_crack_img, self.result_dir)

    def width_measurement(self):
        crop_rectified_image, crop_rectified_img, pixel_coefficient = correction_segmentation(
            self.CU_image, 512, self.qrcode_image, self.result_dir, self.qr_image_real_size)

        prediction_white_crack_img, _ = remove_noise(crop_rectified_image, self.result_dir, self.image_downscale_factor)
        width_measurement(pixel_coefficient, prediction_white_crack_img, self.result_dir)

    def manual_prediction_measurement(self):
        pixel_coefficient = extract_information_from_txt(self.result_dir)
        measure_distance(self.prediction_image, round(pixel_coefficient, 3))

    def manual_image_measurement(self):
        crop_rectified_image, pixel_coefficient = correction(self.MR_image, 512, self.qrcode_image,
                                                             self.result_dir, self.qr_image_real_size)
        measure_distance(crop_rectified_image, round(pixel_coefficient, 3))

    def localization(self):
        medium_img = segmentation(self.MR_image, 512, self.result_dir)
        _, prediction_dilation_img = remove_noise(self.MR_image, self.result_dir, self.image_downscale_factor)
        _, _, _ = localization(self.LR_image, medium_img, prediction_dilation_img, self.result_dir, self.image_downscale_factor)

    def segmentation(self):
        _ = segmentation(self.LR_image, 512, self.result_dir)
        _, _ = remove_noise(self.LR_image, self.result_dir, self.image_downscale_factor)
