__all__ = ["ResultsWidget"]


def __getattr__(name):
    if name == "ResultsWidget":
        from .results import ResultsWidget

        return ResultsWidget
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
