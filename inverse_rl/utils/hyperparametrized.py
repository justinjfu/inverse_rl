CLSNAME = '__clsname__'
_HYPER_ = '__hyper__'
_HYPERNAME_ = '__hyper_clsname__'


def extract_hyperparams(obj):
    if any([isinstance(obj, type_) for type_ in (int, float, str)]):
        return obj
    elif isinstance(type(obj), Hyperparametrized):
        hypers = getattr(obj, _HYPER_)
        hypers[CLSNAME] = getattr(obj, _HYPERNAME_)
        for attr in hypers:
            hypers[attr] = extract_hyperparams(hypers[attr])
        return hypers
    return type(obj).__name__

class Hyperparametrized(type):
    def __new__(self, clsname, bases, clsdict):
        old_init = clsdict.get('__init__', bases[0].__init__)
        def init_wrapper(inst, *args, **kwargs):
            hyper = getattr(inst, _HYPER_, {})
            hyper.update(kwargs)
            setattr(inst, _HYPER_, hyper)

            if getattr(inst, _HYPERNAME_, None) is None:
                setattr(inst, _HYPERNAME_, clsname)
            return old_init(inst, *args, **kwargs)
        clsdict['__init__'] = init_wrapper

        cls = super(Hyperparametrized, self).__new__(self, clsname, bases, clsdict)
        return cls


class HyperparamWrapper(object, metaclass=Hyperparametrized):
    def __init__(self, **hyper_kwargs):
        pass

if __name__ == "__main__":
    class Algo1(object, metaclass=Hyperparametrized):
        def __init__(self, hyper1=1.0, hyper2=2.0, model1=None):
            pass


    class Algo2(Algo1):
        def __init__(self, hyper3=5.0, **kwargs):
            super(Algo2, self).__init__(**kwargs)


    class Model1(object, metaclass=Hyperparametrized):
        def __init__(self, hyper1=None):
            pass


    def get_params_json(**kwargs):
        hyper_dict = extract_hyperparams(HyperparamWrapper(**kwargs))
        del hyper_dict[CLSNAME]
        return hyper_dict

    m1 = Model1(hyper1='Test')
    a1 = Algo2(hyper1=1.0, hyper2=5.0, hyper3=10.0, model1=m1)

    print( isinstance(type(a1), Hyperparametrized))
    print(get_params_json(a1=a1))
