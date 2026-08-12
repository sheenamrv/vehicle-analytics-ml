from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from src.frontend.main_window import AnalyticsWindow

__all__ = ["AnalyticsWindow"]


def __getattr__(name: str) -> Any:
    if name != "AnalyticsWindow":
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    from src.frontend.main_window import AnalyticsWindow

    return AnalyticsWindow
