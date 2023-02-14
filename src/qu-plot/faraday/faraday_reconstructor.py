from abc import ABCMeta, abstractmethod
from dataclasses import dataclass


@dataclass(init=True, repr=True)
class FaradayReconstructor(metaclass=ABCMeta):
    @abstractmethod
    def config_fd_space(self):
        pass

    @abstractmethod
    def reconstruct(self):
        pass

    @abstractmethod
    def calculate_second_moment(self):
        pass
