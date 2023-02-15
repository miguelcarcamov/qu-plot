from dataclasses import dataclass

from csromer.base import Dataset
from csromer.dictionaries import Wavelet
from csromer.reconstruction import Parameter
from csromer.transformers.dfts import NDFT1D, NUFFT1D
from csromer.transformers.flaggers.flagger import Flagger

from .faraday_reconstructor import FaradayReconstructor


@dataclass(init=True, repr=True)
class CSROMERReconstructor(FaradayReconstructor):
    dataset: Dataset = None
    parameter: Parameter = None
    flagger: Flagger = None
    dft: NDFT1D = None
    nufft: NUFFT1D = None
    wavelet: Wavelet = None
    cellsize: float = None
    oversampling: float = None

    def __post_init__(self):
        super().__init__()
        self.parameter = Parameter()

    def flag_dataset(self, flagger: Flagger = None):

        if flagger is None:
            idxs, outliers_idxs = self.flagger.run()
        else:
            idxs, outliers_idxs = flagger.run()

        return idxs, outliers_idxs

    def config_fd_space(self, cellsize: float = None, oversampling: float = None):
        if cellsize is not None and oversampling is not None:
            self.parameter.calculate_cellsize(dataset=self.dataset, cellsize=cellsize)
        elif cellsize is None and oversampling is not None:
            self.parameter.calculate_cellsize(
                dataset=self.dataset, oversampling=oversampling
            )
        elif cellsize is not None and oversampling is None:
            self.parameter.calculate_cellsize(dataset=self.dataset, cellsize=cellsize)
        else:
            raise ValueError(
                "Either cellsize or oversampling cannot be Nonetype values"
            )

    def config_fourier_transforms(self):
        self.dft = NDFT1D(dataset=self.dataset, parameter=self.parameter)
        self.nufft = NUFFT1D(dataset=self.dataset, parameter=self.parameter, solve=True)

    def reconstruct(self):
        pass

    def calculate_second_moment(self):
        pass
