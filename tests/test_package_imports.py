import builtins
import importlib
import sys


def test_lambda_module_import_does_not_require_numpy():
    original_import = builtins.__import__
    original_modules = {
        name: module
        for name, module in sys.modules.items()
        if name == "ifera" or name.startswith("ifera.")
    }

    for name in list(original_modules):
        sys.modules.pop(name, None)

    def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "numpy" or name.startswith("numpy."):
            raise ModuleNotFoundError("No module named 'numpy'")
        return original_import(name, globals, locals, fromlist, level)

    builtins.__import__ = guarded_import
    try:
        module = importlib.import_module("ifera.s3_batch_parquet_migration")
        assert hasattr(module, "lambda_handler")
    finally:
        builtins.__import__ = original_import
        for name in list(sys.modules):
            if name == "ifera" or name.startswith("ifera."):
                sys.modules.pop(name, None)
        sys.modules.update(original_modules)
