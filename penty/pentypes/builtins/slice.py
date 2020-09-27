class frozen_slice(object):
    def __init__(self, lower, upper, step):
        self.slice = slice(lower, upper, step)

def slice_(lower_types, upper_types, step_types):
    if all(len(tys) == 1 for tys in (lower_types, upper_types, step_types)):
        lower_ty = next(iter(lower_types))
        upper_ty = next(iter(upper_types))
        step_ty = next(iter(step_types))
        NoneTy = type(None)

        isstatic = isinstance(lower_ty, (bool, int, NoneTy))
        isstatic &= isinstance(upper_ty, (bool, int, NoneTy))
        isstatic &= isinstance(step_ty, (bool, int, NoneTy))

        if isstatic:
            return {frozen_slice(lower_ty, upper_ty, step_ty)}

    return {slice}
