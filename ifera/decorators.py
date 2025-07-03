import threading
import functools
import inspect
from typing import Any, Callable, Optional, TypeVar, Generic, ParamSpec, cast
from readerwriterlock import rwlock


def singleton(cls):
    """
    A decorator that turns a class into a Singleton, ensuring only one instance
    is created and its __init__ method is called only once.
    """
    # Store the original __init__ method
    cls._original_init = cls.__init__
    cls._lock = threading.Lock()

    # Override __new__ to control instance creation
    def new(cls, *args, **kwargs):
        with cls._lock:
            if not hasattr(cls, "_instance"):  # Check if instance exists
                cls._instance = object.__new__(cls)  # Create a new instance
            return cls._instance

    # Wrap __init__ to ensure it runs only once
    def init(self, *args, **kwargs):
        with cls._lock:
            if not hasattr(self, "_initialized"):  # Check if already initialized
                cls._original_init(self, *args, **kwargs)  # Call original __init__
                self._initialized = True  # Set flag to prevent re-initialization

    # Assign the new methods to the class
    cls.__new__ = staticmethod(new)
    cls.__init__ = init

    # Ensure the instance is hashable
    cls.__hash__ = lambda self: id(self)

    return cls


P = ParamSpec("P")
R = TypeVar("R")


class ThreadSafeCache(Generic[P, R]):
    """
    A thread-safe caching decorator optimized for read-heavy scenarios.
    Compatible with class methods and uses a reader-writer lock for global
    synchronization, along with parameter-specific locks for computation.
    Normalizes arguments to handle positional and keyword arguments uniformly.

    Args:
        maxsize (int, optional): Maximum number of cached results. If None, no limit.
        ignore_self (bool): If True, excludes 'self' from cache key for class-level caching.
    """

    def __init__(
        self,
        func: Callable[P, R] | None = None,
        maxsize: Optional[int] = None,
        ignore_self: bool = False,
    ) -> None:
        """
        Initialize the ThreadSafeCache decorator.
        Args:
            func (Callable): The function to cache.
            maxsize (Optional[int]): Maximum size of the cache.
            ignore_self (bool): Whether to ignore 'self' in class methods.
        """
        self.func: Optional[Callable[P, R]] = func
        self.maxsize = maxsize
        self.ignore_self = ignore_self
        self.cache = {}
        self.lock_dict = {}  # Maps keys to their specific locks
        self.global_rwlock = (
            rwlock.RWLockFair()
        )  # Fair reader-writer lock for global access
        self.signature = inspect.Signature()  # For type checking and argument binding

        # only wrap if func was given
        if func is not None:
            functools.update_wrapper(self, func)
            self.signature = inspect.signature(func)

    def __get__(self, instance: Any, owner: Optional[type] = None) -> Callable[P, R]:
        """Bind instance methods to ensure 'self' is passed correctly."""
        if instance is None:
            return cast(Callable[P, R], self)
        return cast(Callable[P, R], functools.partial(self.__call__, instance))

    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> R:
        """Call the cached function with arguments, managing locks and cache."""
        if self.func is None and len(args) == 1 and callable(args[0]) and not kwargs:
            return cast(
                Any,
                ThreadSafeCache(
                    args[0], maxsize=self.maxsize, ignore_self=self.ignore_self
                ),
            )

        if self.func is None:
            raise ValueError(
                "Function to cache must be provided either as a decorator or as the first argument."
            )

        # Create a normalized key from arguments
        key = self._create_normalized_key(args, kwargs)

        # Step 1: Try to get the result from cache (read-only)
        with self.global_rwlock.gen_rlock():
            if key in self.cache:
                return cast(R, self.cache[key])

        # Step 2: Check if there's a lock for this key (read-only)
        lock = None
        with self.global_rwlock.gen_rlock():
            lock = self.lock_dict.get(key)

        if lock is not None:
            # Wait on the parameter-specific lock without holding the global lock
            with lock:
                with self.global_rwlock.gen_rlock():
                    return cast(R, self.cache[key])

        # Step 3: Key not in lock_dict; create a lock for it (write operation)
        with self.global_rwlock.gen_wlock():
            if key not in self.lock_dict:
                self.lock_dict[key] = threading.Lock()

        # Step 4: Acquire the parameter-specific lock and compute if necessary
        with self.lock_dict[key]:
            # Double-check cache in case another thread computed it
            with self.global_rwlock.gen_rlock():
                if key in self.cache:
                    return cast(R, self.cache[key])

            # Compute the result
            if self.func is None:
                raise RuntimeError("Function to cache must be provided.")
            func = cast(Callable[..., R], self.func)
            result = func(*args, **kwargs)

            # Store in cache (write operation)
            with self.global_rwlock.gen_wlock():
                self.cache[key] = result
                # Handle maxsize (simple eviction, not LRU)
                if self.maxsize is not None and len(self.cache) > self.maxsize:
                    del self.cache[next(iter(self.cache))]

            return cast(R, result)

    def _create_normalized_key(self, args, kwargs):
        """
        Create a normalized cache key that treats positional and keyword arguments uniformly.
        """
        # Handle self for class methods
        if self.ignore_self and args and hasattr(args[0], "__class__"):
            # For methods with self, exclude it from the key
            bound_args = self.signature.bind_partial(*args[1:], **kwargs)
        else:
            # For non-methods or if not ignoring self
            if (
                args
                and hasattr(args[0], "__class__")
                and not hasattr(args[0], "__hash__")
            ):
                # Handle non-hashable self by using its id
                modified_args = (id(args[0]),) + args[1:]
                bound_args = self.signature.bind_partial(*modified_args, **kwargs)
            else:
                bound_args = self.signature.bind_partial(*args, **kwargs)

        bound_args.apply_defaults()
        # Create a key from sorted parameter names and values
        param_items = sorted(bound_args.arguments.items())
        return tuple(param_items)

    # Utility function to clear the cache
    def cache_clear(self) -> None:
        with self.global_rwlock.gen_wlock():
            self.cache.clear()
            self.lock_dict.clear()

    # Utility function to remove a specific key
    def cache_remove(self, *args, **kwargs) -> None:
        key = self._create_normalized_key(args, kwargs)

        with self.global_rwlock.gen_wlock():
            if key in self.cache:
                del self.cache[key]
            if key in self.lock_dict:
                del self.lock_dict[key]
