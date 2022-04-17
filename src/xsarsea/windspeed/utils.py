import os

cmod_descr = {}

def register_cmod(name=None, inc_range=[17., 50.], wspd_range=[0.2, 50.], phi_range=None):
    def inner(func_or_file):
        func = None
        file = None
        if callable(func_or_file):
            func = func_or_file
        else:
            file = func_or_file
        try:
            cmod_name = name or func.__name__
        except AttributeError:
            cmod_name = os.path.splitext(os.path.basename(file))[0]

        update = {
            'gmf': func,
            'lut_path': file,
            'inc_range': inc_range,
            'wspd_range': wspd_range,
            'phi_range': phi_range,
        }
        update = {k: update[k] for k in update.keys() if update[k] is not None}

        if cmod_name not in cmod_descr:
            cmod_descr[cmod_name] = {}

        cmod_descr[cmod_name].update(update)

        return func

    return inner