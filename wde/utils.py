import importlib

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

    module = importlib.import_module(module_name)
    return getattr(module, class_name)


class LazyLoader:

    def __init__(self, module):
        self.module = module
        self._mod = None

    def _ensure_import(self):
        if self._mod is None:
            self._mod = lazy_import(self.module)

    def __getattr__(self, attr):
        self._ensure_import()

        return getattr(self._mod, attr)

    def __call__(self, *args, **kwargs):
        self._ensure_import()

        return self._mod(*args, **kwargs)


def process_warp(fn, /, *args, **kwargs):
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor

    with ProcessPoolExecutor(1, mp.get_context("spawn")) as executor:
        f = executor.submit(fn, *args, **kwargs)
        return f.result()


def exception_handling(fn, /, *args, **kwargs):
    try:
        return fn(*args, **kwargs)
    except Exception:
        import traceback
        traceback.print_exc()


def process_warp_with_exc(fn, /, *args, **kwargs):
    return process_warp(exception_handling, fn, *args, **kwargs)
