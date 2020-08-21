"""
@author: Idan Kringel
Contains global fields / classes / utilities
"""

from collections import namedtuple
import sys

# named tuple that is used as training data fields
TrainingData = namedtuple("TrainingData", "learning_rate epochs batch_size validation_split")


class LoggerWriter:
    """
    Used for replacing stdout in specific sections
    """

    def __init__(self, level):
        self.level = level

    def write(self, message):
        self.level(message)

    def flush(self):
        self.level(sys.stderr)


class Singleton(type):
    """
    Singleton to be used as metaclass
    """
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]