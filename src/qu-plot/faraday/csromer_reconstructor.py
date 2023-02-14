from dataclasses import dataclass

from csromer.base import Dataset

from .faraday_reconstructor import FaradayReconstructor


@dataclass(init=True, repr=True)
class CSROMERReconstructor(FaradayReconstructor):
    dataset: Dataset = None

    def __post_init__(self):
        super().__init__()

    def config_fd_space(self):
        pass

    def reconstruct(self):
        pass

    def calculate_second_moment(self):
        pass
