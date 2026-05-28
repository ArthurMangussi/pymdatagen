try:
    from importlib.metadata import PackageNotFoundError, version

    __version__ = version(__name__)
except PackageNotFoundError:
    # package is not installed
    pass
