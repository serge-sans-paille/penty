str_iterator = type(iter(""))

def next_element(self_types):
    result_types = set()
    for o in self_types:
        if o is str_iterator:
            result_types.add(str)
        elif isinstance(o, str_iterator):
            result_types.add(str)
        else:
            raise NotImplementedError
    return result_types
