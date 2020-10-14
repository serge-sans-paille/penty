def astype(ty):
    return type(ty.__args__[0]) if hasattr(ty, 'mro') and issubclass(ty, Cst) else ty

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

class FunctionTypeMeta(type):
    cache = {}

    def __getitem__(self, args):
        if args not in FunctionTypeMeta.cache:
            class LocalFunctionType(FunctionType):
                __args__ = args,

            FunctionTypeMeta.cache[args] = LocalFunctionType
        return FunctionTypeMeta.cache[args]

    def __repr__(self):
        return 'FunctionType[{}]'.format(self.__args__[0])


class FunctionType(metaclass=FunctionTypeMeta):
    pass

class ModuleMeta(type):
    cache = {}

    def __getitem__(self, args):
        if args not in ModuleMeta.cache:
            class LocalModule(Module):
                __args__ = args,

            ModuleMeta.cache[args] = LocalModule
        return ModuleMeta.cache[args]

    def __repr__(self):
        return 'Module[{}]'.format(self.__args__[0])


class Module(metaclass=ModuleMeta):
    pass

class LambdaMeta(type):
    cache = {}

    def __getitem__(self, args):
        if args not in LambdaMeta.cache:
            class LocalLambda(Lambda):
                __args__ = args,

            LambdaMeta.cache[args] = LocalLambda
        return LambdaMeta.cache[args]

    def __repr__(self):
        return 'Lambda[{}]'.format(self.__args__[0])


class Lambda(metaclass=LambdaMeta):
    pass
