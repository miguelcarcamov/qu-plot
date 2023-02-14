from abc import ABCMeta, abstractmethod
from dataclasses import dataclass


@dataclass(init=True, repr=True)
class Io(metaclass=ABCMeta):
    """
    I/O class to handle files

    Parameters
    ----------
    input_name : str
        Input file name
    output_name : str
        Output file name
    """

    input_name: str = None
    output_name: str = None

    @abstractmethod
    def read(self):
        pass

    @abstractmethod
    def write(self):
        pass
