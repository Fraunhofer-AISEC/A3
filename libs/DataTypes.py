from enum import Enum, auto


class AutoencoderLayers(Enum):
    ENCODER = auto()
    DECODER = auto()
    CODE = auto()
    OUTPUT = auto()

    def __str__(self):
        return self.name
