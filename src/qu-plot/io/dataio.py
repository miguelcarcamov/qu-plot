from dataclasses import dataclass

import numpy as np
from astropy import constants as const

from ..utils import QUcfg


@dataclass
class QUdata:
    cfg: QUcfg = None
    nu: np.ndarray = None
    lambda_squared: np.ndarray = None
    stokes_I: np.ndarray = None
    stokes_Qn: np.ndarray = None
    stokes_Un: np.ndarray = None
    stokes_Vn: np.ndarray = None
    noise: np.ndarray = None
    norm: float = None

    def __post_init__(self):
        pass

    def read_data(self) -> None:

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

    def norm_data(self):

        self.norm = np.max(
            [np.max(np.abs(self.stokes_Qn)), np.max(np.abs(self.stokes_Un))]
        )

        self.stokes_Qn = self.stokes_Qn / self.norm
        self.stokes_Un = self.stokes_Un / self.norm
        self.noise = self.noise / self.norm

        return

    def unormalize_data(self, in_q, in_u, in_noise):

        out_q = in_q * self.norm
        out_u = in_u * self.norm
        out_noise = in_noise * self.norm

        return out_q, out_u, out_noise
