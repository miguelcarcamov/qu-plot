from dataclasses import dataclass

import numpy as np
from csromer.base import Dataset
from csromer.dictionaries import Wavelet
from csromer.objectivefunction import L1, TSV, TV, Chi2, OFunction
from csromer.optimization import FISTA
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
    coefficients: np.ndarray = None
    fd_restored: np.ndarray = None
    fd_model: np.ndarray = None
    fd_residual: np.ndarray = None
    cellsize: float = None
    oversampling: float = None
    lambda_l_norm: float = None

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

    def get_dirty_faraday_depth(self):
        return self.dft.backward(self.dataset.data)

    def get_rmtf(self):
        return self.dft.RMTF()

    def reconstruct(self):
        fd_dirty = self.get_dirty_faraday_depth()
        self.parameter.data = fd_dirty
        self.parameter.complex_data_to_real()

        if self.wavelet is not None:
            lambda_l_norm = (
                np.sqrt(self.dataset.m + 2 * np.sqrt(self.dataset.m))
                * 2.0
                * np.sqrt(2)
                * np.mean(self.dataset.sigma)
            )
        else:
            lambda_l_norm = (
                np.sqrt(self.dataset.m + 2 * np.sqrt(self.dataset.m))
                * np.sqrt(2)
                * np.mean(self.dataset.sigma)
            )

        chi2 = Chi2(dft_obj=self.nufft, wavelet=self.wavelet)
        l1 = L1(reg=lambda_l_norm)

        F_func = [chi2, l1]
        f_func = [chi2]
        g_func = [l1]

        F_obj = OFunction(F_func)
        f_obj = OFunction(f_func)
        g_obj = OFunction(g_func)

        if self.wavelet is not None:
            opt_noise = 2.0 * self.dataset.theo_noise
        else:
            opt_noise = self.dataset.theo_noise

        opt = FISTA(
            guess_param=self.parameter,
            F_obj=F_obj,
            fx=chi2,
            gx=g_obj,
            noise=opt_noise,
            verbose=True,
        )

        obj, X = opt.run()

        if self.wavelet is not None:
            self.coefficients = X.data
            X.data = self.wavelet.reconstruct(X.data)

        self.fd_model = X.real_data_to_complex()

        self.fd_residual = self.dft.backward(
            self.dataset.data - self.dataset.model_data
        )

        self.fd_restored = X.convolve() + self.fd_residual

    def calculate_second_moment(self):
        fd_model_abs = np.abs(self.fd_model)
        k_parameter = np.sum(fd_model_abs)
        first_moment = np.sum(self.parameter.phi * fd_model_abs) / k_parameter
        second_moment = (
            np.sum((self.parameter.phi - first_moment) ** 2 * fd_model_abs)
            / k_parameter
        )

        return second_moment
