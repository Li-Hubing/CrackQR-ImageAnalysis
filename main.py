from scripts.image_processor import ImageProcessor


def run(task):
    image_downscale_factor = 3 # Adjust values upward if the image exceeds the display area.
    qrcode_image = 'qr_images/00002-3089035605.png'
    qr_real_size = 120
    
    if task == 'A':
        """LR, MR"""
        component_aspect_ratio = 2.5 / 1.78
        processor = ImageProcessor(qrcode_image, qr_real_size, image_downscale_factor=image_downscale_factor)
        processor.correction_localization_measurement(component_aspect_ratio)
    elif task == 'B':
        """MR"""
        processor = ImageProcessor(qrcode_image, qr_real_size)
        processor.parameter_measurement()
    elif task == 'C':
        """CU"""
        processor = ImageProcessor(qrcode_image, qr_real_size)
        processor.width_measurement()
    elif task == 'D':
        result_dir = 'results/20260310-214852'
        processor = ImageProcessor(qrcode_image, qr_real_size, result_dir)
        processor.manual_prediction_measurement()
    elif task == 'E':
        """MR"""
        processor = ImageProcessor(qrcode_image, qr_real_size)
        processor.manual_image_measurement()
    elif task == 'F':
        """LR, MR"""
        processor = ImageProcessor(qrcode_image, qr_real_size, image_downscale_factor=image_downscale_factor)
        processor.localization()
    elif task == 'G':
        processor = ImageProcessor()
        processor.segmentation()

if __name__ == '__main__':
    run('A')


