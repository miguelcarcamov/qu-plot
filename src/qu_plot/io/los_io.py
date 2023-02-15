from dataclasses import dataclass, field

import numpy as np
from astropy import constants as const
from csromer.base import Dataset

from .config_io import ConfigIo
from .io import Io


@dataclass
class LOSIo(Io):
    cfg: ConfigIo = None
    nu: np.ndarray = field(init=False)
    lambda_squared: np.ndarray = field(init=False)
    stokes_I: np.ndarray = field(init=False)
    stokes_Qn: np.ndarray = field(init=False)
    stokes_Un: np.ndarray = field(init=False)
    stokes_Vn: np.ndarray = field(init=False)
    noise: np.ndarray = field(init=False)

    def __post_init__(self):
        super().__init__()
        if self.cfg is not None:
            self._read_data()

    def read(self):
        if self.cfg is not None:
            self._read_data()
            nu = self.nu
            p = self.stokes_Qn[::-1] + 1j * U[::-1]
            sigma_qu = self.noise[::-1]

            dataset = Dataset(nu=nu, data=p, sigma=sigma_qu, spectral_idx=0.7)

            return dataset
        else:
            raise ValueError("Configuration object has not been instanced")

    def write(self):
        pass

    def _read_data(self) -> None:

        # freq       I        Q        U        V        N
        data = np.loadtxt(self.cfg.data_path + self.cfg.data_file)
        self.nu = data[:, 0] * 1e6
        self.stokes_I = data[:, 1]  # stokes I in file is Russ' model, not the raw data
        self.stokes_Qn = data[:, 2]
        self.stokes_Un = data[:, 3]
        self.stokes_Vn = data[:, 4]
        Q_bkg = data[:, 5]
        U_bkg = data[:, 6]
        V_bkg = data[:, 7]
        self.noise = data[:, 8]

        if self.cfg.bkg_corr:
            self.stokes_Qn -= Q_bkg
            self.stokes_Un -= U_bkg

        if self.cfg.pol_frac:
            self.stokes_Qn *= 100.0 / self.stokes_I
            self.stokes_Un *= 100.0 / self.stokes_I
            self.noise *= 100.0 / self.stokes_I

        # make data in lambda^2:
        self.lambda_squared = (const.c.value / self.nu) ** 2
