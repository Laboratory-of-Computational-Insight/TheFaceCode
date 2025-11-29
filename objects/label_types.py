import enum


class Attribute(enum.IntEnum):
    #not all models have the same attributes, some have intermediate data. Thats why its with -1,...
    EMOTION = -3
    VALENCE = -2
    AROUSAL = -1
    IDENTITY = 0
