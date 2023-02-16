from dataclasses import dataclass

from .io import Io


@dataclass
class CatalogIo(Io):
    def read(self):
        pass

    def write(self):
        pass
