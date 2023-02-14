import ast
import os
from configparser import ConfigParser
from dataclasses import dataclass, field

from .io import Io


@dataclass
class ConfigIo(Io):
    fit_type: str = field(init=False)
    data_path: str = field(init=False)
    data_file: str = field(init=False)
    cat_path: str = field(init=False)
    cat_file: str = field(init=False)
    bkg_corr: bool = field(init=False)
    pol_frac: bool = field(init=False)
    rm_path: str = field(init=False)
    plot_path: str = field(init=False)

    def __post_init__(self):
        super().__init__()

    @staticmethod
    def __parse_config(filename):
        config = ConfigParser(allow_no_value=True)
        config.read(filename)

        # Build a nested dictionary with tasknames at the top level
        # and parameter values one level down.
        taskvals = dict()
        for section in config.sections():

            if section not in taskvals:
                taskvals[section] = dict()

            for option in config.options(section):
                # Evaluate to the right type()
                try:
                    taskvals[section][option] = ast.literal_eval(
                        config.get(section, option)
                    )
                except (ValueError, SyntaxError):
                    err = "Cannot format field '{0}' in config file '{1}'".format(
                        option, filename
                    )
                    err += ", which is currently set to {0}. Ensure strings are in 'quotes'.".format(
                        config.get(section, option)
                    )
                    raise ValueError(err)

        return taskvals, config

    def read(self):
        self.read_cfg(self.input_name)

    def write(self):
        pass

    def read_cfg(self, cfg_file):

        config_dict, config = self.__parse_config(cfg_file)

        self.fit_type = config_dict["data"]["type"]
        self.data_path = config_dict["data"]["path"]
        if self.fit_type == "single":
            self.data_file = config_dict["data"]["file"]
        self.cat_path = config_dict["data"]["catpath"]
        self.cat_file = config_dict["data"]["catfile"]

        self.bkg_corr = config_dict["data"]["bkg_correction"]
        self.pol_frac = config_dict["data"]["pol_frac"]

        if "rmspec" in config_dict:
            self.rm_path = config_dict["rmspec"]["path"]

        if "fitting" in config_dict:
            self.fit_ml = config_dict["fitting"]["ml"]
            self.fit_mcmc = config_dict["fitting"]["mcmc"]
            self.fit_gpm = config_dict["fitting"]["gpm"]

        if "plots" in config_dict:
            self.plot_path = config_dict["plots"]["path"]
            if not os.path.exists(self.plot_path):
                raise FileNotFoundError(
                    "Path to plotting outputs does not exist - please correct path"
                )

            self.plot_raw = config_dict["plots"]["rawdata"]
            self.plot_fd = config_dict["plots"]["fdspec"]
            self.plot_ml = config_dict["plots"]["mlfit"]
            self.plot_corner = config_dict["plots"]["corner"]
            self.plot_mcmc = config_dict["plots"]["mcmcfit"]
            self.plot_gpm = config_dict["plots"]["gpmfit"]

        if "output" in config_dict:
            self.outfile = config_dict["output"]["outfile"]
            self.output = config_dict["output"]["write_output"]
