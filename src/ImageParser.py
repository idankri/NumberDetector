"""
@author: idankri
python version: 3.7
tensorflow version: 2.3.0
pandas version: 1.0.1
numpy version: 1.18.1

Utility used to parse images (assumed to be of hand written numbers) to be used by the model
"""


import imageio
import numpy as np

#TODO: add parsing features for other images

class ImageParser:
    @staticmethod
    def parse_28_by_28_png_picture(image_path:str):
        """
        Should be used to parse a picture in the expected format
        (black background and white writing) and expected resolution (28 x 28)
        :param image_path: path for image
        :return: numpy array ready to be parsed by the Number Detector Model
        """
        img = imageio.imread(image_path)
        flat = [ImageParser._normalize_mean(x) for y in img for x in y]
        greyscaled_img = np.array(flat).reshape(28, 28)
        return np.array([greyscaled_img])

    @staticmethod
    def _normalize_mean(pixel):
        return (sum(pixel) / 3) / 255

    @staticmethod
    def parse_picture(image_path:str):
        """
        Parse picture with unknown dimensions
        :param image_path:
        :return:
        """
        img = imageio.imread(image_path)
        height, width, _ = img.shape
        flat = [ImageParser._normalize_mean(x) for y in img for x in y]
        #invert picture if needed
        if sum(flat) > (height*width)  * (2 / 3):
            flat = [(1.0 - x) for x in flat]
        greyscaled_img = np.array(flat).reshape(height, width)
        if height != 28 or width != 28:
            #resize (TODO: implement)
            pass
        return np.array([greyscaled_img])

    def resize(self, image_array:np.array, target_width, target_height):
        #TODO: Implement
        pass
