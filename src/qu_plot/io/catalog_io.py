from dataclasses import dataclass

from .io import Io


@dataclass
class CatalogIo(Io):
    def __post_init__(self):
        super().__init__()

    def read(self):
        pass

    def write(self):
        pass
