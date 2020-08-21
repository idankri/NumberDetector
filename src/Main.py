"""
@author: idankri
python version: 3.7
tensorflow version: 2.3.0
pandas version: 1.0.1
numpy version: 1.18.1

Command line program that receives a handwritten number image (single digit)
 and tries to predict its value
 The program is also able to build different model than the default one with custom
 hyper-parameters
"""

import logging, argparse
from src.NumberDetectorModel import NumberDetectorModel
from src.ImageParser import ImageParser
from src.GlobalFields import *

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)
logging.basicConfig(level=logging.DEBUG, format=r'%(asctime)s %(levelname)-8s %(message)s',
                    datefmt=r'%a, %d %b %Y %H:%M:%S',
                    filename=r'log.log', filemode='a')


class DetectNonDefault(argparse.Action):
    """
    Class that helps detect if an argument was set from command line
    """
    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, values)
        setattr(namespace, "_nondefault", True)


def parse_args():
    """
    Parse relevant arguments using ArgumentParser module
    :return:
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--image_path', required=False, type=str, default=r'../res/2.png')
    parser.add_argument('-m', '--model_path', required=False, type=str, default=r"../res/my_model")
    parser.add_argument('-lr', '--learning_rate', required=False, action=DetectNonDefault,
                        type=float, default=0.003)
    parser.add_argument('-e', '--epochs', required=False, type=int, action=DetectNonDefault,
                        default=50)
    parser.add_argument('-bs', '--batch_size', required=False, type=int, action=DetectNonDefault,
                        default=4000)
    parser.add_argument('-vs', '--validation_split', required=False, action=DetectNonDefault,
                        type=float, default=0.2)

    return parser.parse_args()


if __name__ == '__main__':
    all_args = parse_args()
    training_data = None
    if hasattr(all_args, '_nondefault'):
        training_data = TrainingData(learning_rate=all_args.learning_rate,
                                     epochs=all_args.epochs,
                                     batch_size=all_args.batch_size,
                                     validation_split=all_args.validation_split)
    model = NumberDetectorModel(all_args.model_path, training_data)
    img = ImageParser.parse_picture(all_args.image_path)
    print(f"predicted number: {model.predict(img)}")
