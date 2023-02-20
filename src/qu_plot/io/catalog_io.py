from dataclasses import dataclass

import astropy.table as tbl
from astropy.io import fits

from .io import Io


@dataclass
class CatalogIo(Io):
    def read(self):
        with fits.open(self.input_name) as hdu_list:
            catalog = tbl.Table.read(hdu_list, format="fits")
        return catalog

    def write(
        self,
        catalog: tbl.Table = None,
        overwrite: bool = True,
        file_format: str = "fits",
    ):
        catalog.write(catalog, overwrite=overwrite, format=file_format)
