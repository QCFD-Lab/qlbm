from matplotlib.pylab import Enum


class ABEncodingType(Enum):
    r"""Enumerator for the modes of quantum comparator circuits.

    The modes are as follows:

    * (1, ``ABEncodingType.AB``, The regular AB encoding.);
    * (2, ``ABEncodingType.OH``, The one-hot encoding.).
    """

    AB = (1, )
    OH = (2, )
