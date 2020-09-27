
class CstMeta(type):
    cache = {}

    def __getitem__(self, args):
        if isinstance(args, slice):
            # because slice are unhashable
            key = slice, (args.start, args.stop, args.step)
        else:
            key = type(args), args  # because True and 1 are similar otherwise
        if key not in CstMeta.cache:
            class LocalCst(Cst):
                __args__ = args,

            CstMeta.cache[key] = LocalCst
        return CstMeta.cache[key]

    def __repr__(self):
        return 'Cst[{}]'.format(self.__args__[0])


class Cst(metaclass=CstMeta):
    pass

class FDefMeta(type):
    cache = {}

    def __getitem__(self, args):
        if args not in FDefMeta.cache:
            class LocalFDef(FDef):
                __args__ = args,

            FDefMeta.cache[args] = LocalFDef
        return FDefMeta.cache[args]

    def __repr__(self):
        return 'FDef[{}]'.format(self.__args__[0])


class FDef(metaclass=FDefMeta):
    pass
