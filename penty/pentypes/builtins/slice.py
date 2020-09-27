from penty.types import Cst as _Cst

def slice_(lower_types, upper_types, step_types):
    if all(len(tys) == 1 for tys in (lower_types, upper_types, step_types)):
        lower_ty = next(iter(lower_types))
        upper_ty = next(iter(upper_types))
        step_ty = next(iter(step_types))

        isstatic = all(issubclass(ty, _Cst)
                       for ty in (lower_ty, upper_ty, step_ty))

        if isstatic:
            return {_Cst[slice(*(ty.__args__[0]
                                 for ty in (lower_ty, upper_ty, step_ty)))]}

    return {slice}
