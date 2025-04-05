def singleton(cls):
    """
    A decorator that turns a class into a Singleton, ensuring only one instance
    is created and its __init__ method is called only once.
    """
    # Store the original __init__ method
    cls._original_init = cls.__init__

    # Override __new__ to control instance creation
    def new(cls, *args, **kwargs):
        if not hasattr(cls, "_instance"):  # Check if instance exists
            cls._instance = object.__new__(cls)  # Create a new instance
        return cls._instance

    # Wrap __init__ to ensure it runs only once
    def init(self, *args, **kwargs):
        if not hasattr(self, "_initialized"):  # Check if already initialized
            cls._original_init(self, *args, **kwargs)  # Call original __init__
            self._initialized = True  # Set flag to prevent re-initialization

    # Assign the new methods to the class
    cls.__new__ = staticmethod(new)
    cls.__init__ = init

    return cls
