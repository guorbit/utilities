from enum import Enum


class ImageOrdering(Enum):
    """
    Enum class for image ordering

    Available options
    -----------------
    :CHANNEL_FIRST: channel first ordering
    :CHANNEL_LAST: channel last ordering
    """
    CHANNEL_FIRST = 1
    CHANNEL_LAST = 2
