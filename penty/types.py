def astype(ty):
    if issubclass(ty, Cst):
        cst = ty.__args__[0]
        if cst is None:
            return ty
        else:
            return type(cst)
    else:
        return ty

def resolve_base_attrs(attrs, registry):
    for base in attrs['__bases__'].__args__:
        for attr, value in registry[base.__args__[0]].items():
            if attr not in attrs:
                attrs[attr] = value
    return attrs


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
        if args[1] is None:
            return Cst[args[0]]

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


class ConstFunctionTypeMeta(FunctionTypeMeta):
    cache = {}

    def __getitem__(self, args):
        if args not in ConstFunctionTypeMeta.cache:
            class LocalConstFunctionType(ConstFunctionType):
                __args__ = args

            ConstFunctionTypeMeta.cache[args] = LocalConstFunctionType
        return ConstFunctionTypeMeta.cache[args]

    def __repr__(self):
        return 'ConstFunctionType[{}, {}]'.format(self.__args__[0],
                                                  self.__args__[1].__name__)


class ConstFunctionType(FunctionType, metaclass=ConstFunctionTypeMeta):
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


class TupleMeta(type):
    cache = {}

    def __getitem__(self, args):
        if not isinstance(args, tuple):
            args = args,
        if args not in TupleMeta.cache:
            class LocalTuple(Tuple):
                __args__ = args

            TupleMeta.cache[args] = LocalTuple
        return TupleMeta.cache[args]

    def __repr__(self):
        return 'Tuple[{}]'.format(', '.join(map(str, self.__args__)))


class Tuple(tuple, metaclass=TupleMeta):
    pass


class ListMeta(type):

    def __getitem__(self, args):
        if not isinstance(args, set):
            args = {args}

        class LocalList(List):
            __args__ = args,
        return LocalList

    def __repr__(self):
        sortedelts = sorted(map(str, self.__args__[0]))
        return 'List[{{{}}}]'.format(', '.join(sortedelts))


class List(list, metaclass=ListMeta):
    pass


class SetMeta(type):

    def __getitem__(self, args):
        if not isinstance(args, set):
            args = {args}

        class LocalSet(Set):
            __args__ = args,

        return LocalSet

    def __repr__(self):
        sortedkeys = sorted(map(str, self.__args__[0]))
        return 'Set[{{{}}}]'.format(', '.join(sortedkeys))


class Set(set, metaclass=SetMeta):
    pass


class SetIteratorMeta(type):

    def __getitem__(self, args):
        class LocalSetIterator(SetIterator):
            __args__ = args,
        return LocalSetIterator

    def __repr__(self):
        sortedkeys = sorted(map(str, self.__args__[0]))
        return 'SetIterator[{}]'.format(', '.join(sortedkeys))


class SetIterator(metaclass=SetIteratorMeta):
    pass


class GeneratorMeta(type):

    def __getitem__(self, args):
        if not isinstance(args, set):
            args = {args}

        class LocalGenerator(Generator):
            __args__ = args,
        return LocalGenerator

    def __repr__(self):
        sortedelts = sorted(map(str, self.__args__[0]))
        return 'Generator[{{{}}}]'.format(', '.join(sortedelts))


class Generator(list, metaclass=GeneratorMeta):
    pass


class DictMeta(type):

    def __getitem__(self, args):
        def as_set(x):
            return x if isinstance(x, set) else {x}

        args = as_set(args[0]), as_set(args[1])

        class LocalDict(Dict):
            __args__ = args
        return LocalDict

    def __repr__(self):
        sortedkeys = sorted(map(str, self.__args__[0]))
        sortedvals = sorted(map(str, self.__args__[1]))
        return 'Dict[{{{}}}, {{{}}}]'.format(', '.join(sortedkeys),
                                             ', '.join(sortedvals))


class Dict(set, metaclass=DictMeta):
    pass
