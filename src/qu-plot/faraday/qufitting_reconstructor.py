from dataclasses import dataclass

from .faraday_reconstructor import FaradayReconstructor


@dataclass
class QUFittingReconstructor(FaradayReconstructor):
    def config_fd_space(self):
        pass

    def reconstruct(self):
        pass

    def calculate_second_moment(self):
        pass
