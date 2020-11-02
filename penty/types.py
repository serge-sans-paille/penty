def astype(ty):
    if issubclass(ty, Cst):
        cst = ty.__args__[0]
        if cst is None:
            return ty
        else:
            return type(cst)
    else:
        return ty

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


class FilteringBoolMeta(CstMeta):
    cache = {}

    def __getitem__(self, args):
        if args not in FilteringBoolMeta.cache:
            class LocalFilteringBool(FilteringBool):
                __args__ = args

            FilteringBoolMeta.cache[args] = LocalFilteringBool
        return FilteringBoolMeta.cache[args]

    def __repr__(self):
        return 'FilteringBool[{}]'.format(', '.join(map(str, self.__args__)))

    @staticmethod
    def bindings(ty):
        if issubclass(ty, FilteringBool):
            k, v = ty.__args__[1:]
            return {k: set(v)}
        else:
            return {}


class FilteringBool(Cst, metaclass=FilteringBoolMeta):
    __args__ = ()

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

class FunctionTypeMeta(CstMeta):
    cache = {}

    def __getitem__(self, args):
        if args not in FunctionTypeMeta.cache:
            class LocalFunctionType(FunctionType):
                __args__ = args,

            FunctionTypeMeta.cache[args] = LocalFunctionType
        return FunctionTypeMeta.cache[args]

    def __repr__(self):
        return 'FunctionType[{}]'.format(self.__args__[0])

    def __call__(self, *args):
        return self.__args__[0](*args)


class FunctionType(Cst, metaclass=FunctionTypeMeta):
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

class TypeMeta(type):
    cache = {}

    def __getitem__(self, args):
        if args not in TypeMeta.cache:
            class LocalType(Type):
                __args__ = args,

            TypeMeta.cache[args] = LocalType
        return TypeMeta.cache[args]

    def __repr__(self):
        return 'Type[{}]'.format(self.__args__[0])


class Type(metaclass=TypeMeta):
    pass

class TypeOfMeta(TypeMeta):
    cache = {}

    def __getitem__(self, args):
        if args not in TypeOfMeta.cache:
            class LocalTypeOf(TypeOf):
                __args__ = args

            TypeOfMeta.cache[args] = LocalTypeOf
        return TypeOfMeta.cache[args]

    def __repr__(self):
        return 'TypeOf[{}]'.format(', '.join(map(str, self.__args__)))


class TypeOf(Type, metaclass=TypeOfMeta):
    pass
