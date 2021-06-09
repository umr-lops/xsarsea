try:
    from xsarsea_protected.streaks import *
except ImportError:
    def streaks_direction(*args, **kwargs):
        raise ImportError('missing xsarsea_protected module')
