import shortuuid


def random_uuid(length=22) -> str:
    return str(shortuuid.random(length=length))


def lazy_import(module):
    if module is None:

        class Dummy:

            @classmethod
            def from_engine(cls, engine):
                return cls()

        return Dummy

    module_name, class_name = module.split(":")
    import importlib
    module = importlib.import_module(module_name)
    return getattr(module, class_name)
