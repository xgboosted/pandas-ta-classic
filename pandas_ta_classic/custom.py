import importlib
import importlib.util
import logging
import os
import sys
import types
from glob import glob
from os.path import abspath, basename, exists, join, normcase, realpath, splitext

import pandas_ta_classic
from pandas_ta_classic import AnalysisIndicators

logger = logging.getLogger(__name__)

# The names bind() has bound. Binding one again (an edited file imported again)
# is allowed; any other name the package or df.ta already has is refused.
_BOUND: set[str] = set()


def _taken_by(name):
    """What *name* already is in pandas_ta_classic or on df.ta, or None when it is free.

    A custom file named like a built-in indicator replaced ta.rsi, df.ta.rsi()
    and rsi in strategy() without a word, while stochrsi and ta.momentum.rsi
    kept the built-in; one named Category.py replaced ta.Category and made
    import_dir() fail halfway (#342).
    """
    if name in _BOUND:
        return None
    if any(name in names for names in pandas_ta_classic.Category.values()):
        return "a built-in indicator"
    if hasattr(pandas_ta_classic, name):
        return "an attribute of pandas_ta_classic"
    if hasattr(AnalysisIndicators, name):
        return "an attribute of df.ta"
    return None


def bind(function_name, function, method):
    """
    Helper function to bind the function and class method defined in a custom
    indicator module to the active pandas_ta_classic instance.

    A name that pandas_ta_classic or df.ta already has (a built-in indicator,
    Category, strategy, ...) raises ValueError, unless an earlier bind() bound
    it, so an edited indicator can be bound again.

    Args:
        function_name (str): The name of the indicator within pandas_ta_classic
        function (fcn): The indicator function
        method (fcn): The class method corresponding to the passed function
    """
    taken = _taken_by(function_name)
    if taken is not None:
        raise ValueError(f"bind() function_name {function_name!r} is already {taken}; give the custom indicator a name of its own")
    setattr(pandas_ta_classic, function_name, function)
    setattr(AnalysisIndicators, function_name, method)
    _BOUND.add(function_name)


def create_dir(path, create_categories=True, verbose=True):
    """
    Helper function to setup a suitable folder structure for working with
    custom indicators. You only need to call this once whenever you want to
    setup a new custom indicators folder.

    Args:
        path (str): Full path to where you want your indicator tree
        create_categories (bool): If True create category sub-folders
        verbose (bool): If True print verbose output of results
    """

    # ensure that the passed directory exists / is readable
    if not exists(path):
        os.makedirs(path)
        if verbose:
            logger.info("Created main directory '%s'.", path)

    # optionally add any missing category subdirectories
    if create_categories:
        for sd in [*pandas_ta_classic.Category]:
            d = abspath(join(path, sd))
            if not exists(d):
                os.makedirs(d)
                if verbose:
                    dirname = basename(d)
                    logger.info("Created an empty sub-directory '%s'.", dirname)


def get_module_functions(module):
    """
     Helper function to get the functions of an imported module as a dictionary.

    Args:
        module: python module

    Returns:
        dict: module functions mapping
        {
            "func1_name": func1,
            "func2_name": func2,...
        }
    """
    module_functions = {}

    for name, item in vars(module).items():
        if isinstance(item, types.FunctionType):
            module_functions[name] = item

    return module_functions


def _resolve(name):
    """The spec an import of *name* would load now, found the way reload() finds it: through sys.meta_path, ignoring sys.modules."""
    for finder in sys.meta_path:
        find_spec = getattr(finder, "find_spec", None)
        spec = find_spec(name, None) if find_spec is not None else None
        if spec is not None:
            return spec
    return None


def _load_module(module_path, category_dir, added_paths, replaced_modules):
    """Load one indicator module from *module_path*; bind nothing.

    The function executes the Python file at *module_path* as a new module and
    returns its name with the mandatory ``<name>`` and ``<name>_method``
    callables; :func:`import_dir` binds them once every file has loaded. A module that
    fails to import, lacks either callable, or whose name already resolves to
    another module (a stdlib or installed module, or a file of the same name
    in another category) raises ImportError.

    Args:
        module_path (str): Absolute path to the ``.py`` file to load.
        category_dir (str): Absolute path to the category directory; added to
            :data:`sys.path` when not already present.
        added_paths (list): Collects the directories this call added to
            :data:`sys.path`, so :func:`import_dir` can take them back.
        replaced_modules (list): Collects ``(name, previous sys.modules entry)``
            for the same purpose.
    """
    module_name = splitext(basename(module_path))[0]
    if not module_name.isidentifier():
        # ab.cd.py would import a package 'ab'; an empty name has nothing to import
        raise ImportError(f"custom indicator '{module_path}': '{module_name}' is not a valid module name. Rename the file.")

    # The module is imported by name, and category_dir comes last on sys.path, so
    # a name that is already taken (a stdlib or installed module, or the same file
    # name in another category) used to reload that other module and bind it.
    # Resolve it the way reload() does, ignoring the sys.modules cache, and do so
    # before category_dir joins sys.path, so a refusal leaves sys.path unchanged.
    spec = _resolve(module_name)
    if spec is not None:  # None: not found yet; category_dir is added below
        origin = spec.origin  # a path, 'built-in', 'frozen', or None for a namespace package
        if origin is None or not exists(origin) or normcase(realpath(origin)) != normcase(realpath(module_path)):
            raise ImportError(
                f"custom indicator '{module_path}' cannot be imported as '{module_name}': that name already resolves to "
                f"{origin or spec}. Rename the file and its two functions."
            )
    taken = _taken_by(module_name)
    if taken is not None:
        raise ImportError(
            f"custom indicator '{module_path}' cannot be bound as '{module_name}': that name is already {taken}. "
            "Rename the file and its two functions."
        )

    if category_dir not in sys.path:
        sys.path.append(category_dir)
        added_paths.append(category_dir)

    # A new module object, not importlib.reload(): reload re-executes the live
    # module in place, so a file refused later in the same import_dir left an
    # indicator bound earlier half on the new code (its *_method looks the
    # function up in the module at call time, the bound function stays old).
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"custom indicator '{module_path}' cannot be loaded as a module")
    module = importlib.util.module_from_spec(spec)
    # in sys.modules while it runs, as an import has it (dataclasses look the module up there)
    replaced_modules.append((module_name, sys.modules.get(module_name)))
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception as ex:
        # user module code can raise anything
        raise ImportError(f"Unable to load the custom indicator module '{module_name}': {ex}") from ex
    module_functions = get_module_functions(module)

    fcn_callable = module_functions.get(module_name)
    fcn_method_callable = module_functions.get(f"{module_name}_method")

    for required, found in ((module_name, fcn_callable), (f"{module_name}_method", fcn_method_callable)):
        if found is None:
            raise ImportError(f"custom indicator module '{module_path}' has no function named '{required}'")

    return module_name, fcn_callable, fcn_method_callable


def import_dir(path, verbose=True):
    # ensure that the passed directory exists / is readable
    if not exists(path):
        # a wrong path used to be logged and import nothing
        raise FileNotFoundError(f"import_dir() path {path!r} does not exist")

    # list the contents of the directory
    dirs = glob(abspath(join(path, "*")))

    # Load every module first and bind only when all of them loaded: binding as
    # it went left the modules before a refused file bound and listed in
    # Category, and the refused file's directory on sys.path.
    loaded = []
    added_paths: list[str] = []
    replaced_modules: list = []
    try:
        for d in dirs:
            dirname = basename(d)

            # only look in directories which are valid pandas_ta_classic categories
            if dirname not in [*pandas_ta_classic.Category]:
                if verbose:
                    logger.info(
                        "Skipping the sub-directory '%s' since it's not a valid pandas_ta_classic category.",
                        dirname,
                    )
                continue

            for module in glob(abspath(join(path, dirname, "*.py"))):
                loaded.append((module, dirname, *_load_module(module, d, added_paths, replaced_modules)))
    except BaseException:
        for added in added_paths:
            if added in sys.path:
                sys.path.remove(added)
        for name, previous in reversed(replaced_modules):
            if previous is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = previous
        raise

    for module, dirname, module_name, function, method in loaded:
        # bind() first: it reads Category to tell a built-in name from a free one.
        bind(module_name, function, method)
        if module_name not in pandas_ta_classic.Category[dirname]:
            pandas_ta_classic.Category[dirname].append(module_name)
        if verbose:
            logger.info(
                "Successfully imported the custom indicator '%s' into category '%s'.",
                module,
                dirname,
            )


import_dir.__doc__ = """
Import a directory of custom indicators into pandas_ta_classic

All or nothing: every module is loaded first, and they are bound only when
none was refused. A refused file raises ImportError and leaves
pandas_ta_classic, Category, sys.path and the indicator modules in sys.modules
as they were, including indicators bound by an earlier call. A module that a
file imported itself stays loaded, as after any failed import.

Args:
    path (str): Full path to your indicator tree
    verbose (bool): If True verbose output of results

This method allows you to experiment and develop your own technical analysis
indicators in a separate local directory of your choice but use them seamlessly
together with the existing pandas_ta_classic functions just like if they were part of
pandas_ta_classic.

If you at some late point would like to push them into the pandas_ta_classic library
you can do so very easily by following the step by step instruction here
https://github.com/xgboosted/pandas-ta-classic/issues.

A brief example of usage:

1. Loading the 'ta' module:
>>> import pandas as pd
>>> import pandas_ta_classic as ta

2. Create an empty directory on your machine where you want to work with your
indicators. Invoke pandas_ta_classic.custom.import_dir once to pre-populate it with
sub-folders for all available indicator categories, e.g.:

>>> import os
>>> from os.path import abspath, join, expanduser
>>> from pandas_ta_classic.custom import create_dir, import_dir
>>> ta_dir = abspath(join(expanduser("~"), "my_indicators"))
>>> create_dir(ta_dir)

3. You can now create your own custom indicator e.g. by copying existing
ones from pandas_ta_classic core module and modifying them.

IMPORTANT: Each custom indicator needs a unique name, which must not be the
name of a built-in indicator, of anything else pandas_ta_classic or df.ta
already has (Category, strategy, ...), or of a stdlib or installed module
(import_dir raises ImportError otherwise; an indicator that an earlier
import_dir bound may be imported again), and must have both
a) a function named exactly as the module, e.g. 'ni' if the module is ni.py
b) a matching method used by AnalysisIndicators named as the module but
   ending with '_method'. E.g. 'ni_method'

In essence these modules should look exactly like the standard indicators
available in categories under the pandas_ta_classic-folder. The only difference will
be an addition of a matching class method.

For an example of the correct structure, look at the example ni.py in the
examples folder.

The ni.py indicator is a trend indicator so therefore we drop it into the
sub-folder named trend. Thus we have a folder structure like this:

~/my_indicators/
│
├── candles/
.
.
└── trend/
.      └── ni.py
.
└── volume/

4. We can now dynamically load all our custom indicators located in our
designated indicators directory like this:

>>> import_dir(ta_dir)

If your custom indicator(s) loaded successfully then it should behave exactly
like all other native indicators in pandas_ta_classic, including help functions.
"""


def load_indicator_module(name):
    """
     Helper function to (re)load an indicator module.

    Returns:
        dict: module functions mapping
        {
            "func1_name": func1,
            "func2_name": func2,...
        }

    """
    # load module
    try:
        module = importlib.import_module(name)
    except Exception as ex:
        # user module code can raise anything; this used to sys.exit(1) and kill the caller's interpreter
        raise ImportError(f"Unable to load the custom indicator module '{name}': {ex}") from ex

    # reload to refresh previously loaded module
    module = importlib.reload(module)
    return get_module_functions(module)
