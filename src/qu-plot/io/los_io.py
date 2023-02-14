from dataclasses import dataclass, field

import numpy as np
from astropy import constants as const

from ..utils import QUcfg
from .io import Io


@dataclass
class QUdata(Io):
    _cfg: QUcfg = None
    nu: np.ndarray = field(init=False)
    lambda_squared: np.ndarray = field(init=False)
    stokes_I: np.ndarray = field(init=False)
    stokes_Qn: np.ndarray = field(init=False)
    stokes_Un: np.ndarray = field(init=False)
    stokes_Vn: np.ndarray = field(init=False)
    noise: np.ndarray = field(init=False)
    norm: float = field(init=False)

    def __post_init__(self):
        super().__init__()
        if self._cfg is not None:
            self._read_data()

    def read(self):
        if self._cfg is not None:
            self._read_data()
        else:
            raise ValueError("Configuration object has not been instanced")

    def write(self):
        pass

    def _read_data(self) -> None:

        # freq       I        Q        U        V        N
        data = np.loadtxt(self._cfg.data_path + self._cfg.data_file)
        self.nu = data[:, 0] * 1e6
        self.stokes_I = data[:, 1]  # stokes I in file is Russ' model, not the raw data
        self.stokes_Qn = data[:, 2]
        self.stokes_Un = data[:, 3]
        self.stokes_Vn = data[:, 4]
        Q_bkg = data[:, 5]
        U_bkg = data[:, 6]
        V_bkg = data[:, 7]
        self.noise = data[:, 8]

        if self._cfg.bkg_corr:
            self.stokes_Qn -= Q_bkg
            self.stokes_Un -= U_bkg

        if self._cfg.pol_frac:
            self.stokes_Qn *= 100.0 / self.stokes_I
            self.stokes_Un *= 100.0 / self.stokes_I
            self.noise *= 100.0 / self.stokes_I

        # make data in lambda^2:
        self.lambda_squared = (const.c.value / self.nu) ** 2

    def _norm_data(self):

        self.norm = np.max(
            [np.max(np.abs(self.stokes_Qn)), np.max(np.abs(self.stokes_Un))]
        )

        self.stokes_Qn /= self.norm
        self.stokes_Un /= self.norm
        self.noise /= self.norm

    def _unormalize_data(self, in_q, in_u, in_noise):

        out_q = in_q * self.norm
        out_u = in_u * self.norm
        out_noise = in_noise * self.norm

        return out_q, out_u, out_noise
