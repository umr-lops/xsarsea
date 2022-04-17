import os

registered_gmfs = {}

def register_gmf(name=None, inc_range=[17., 50.], wspd_range=[0.2, 50.], phi_range=None):
    """TODO: docstring"""
    def inner(func):
        gmf_name = name or func.__name__

        update = {
            'gmf': func,
            'inc_range': inc_range,
            'wspd_range': wspd_range,
            'phi_range': phi_range,
        }
        update = {k: update[k] for k in update.keys() if update[k] is not None}

        if gmf_name not in registered_gmfs:
            registered_gmfs[gmf_name] = {}

        registered_gmfs[gmf_name].update(update)

        return func

    return inner