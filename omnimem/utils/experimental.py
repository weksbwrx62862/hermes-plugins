from __future__ import annotations

import functools
import logging
import threading
from collections.abc import Callable
from typing import Any

logger = logging.getLogger(__name__)

_warned: set[str] = set()
_warn_lock = threading.Lock()

def experimental(func: Callable[..., Any]) -> Callable[..., Any]:
    _original_doc = func.__doc__ or ""
    func.__doc__ = f"[Experimental] {_original_doc}".strip()

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        key = f"{func.__module__}.{func.__qualname__}"
        with _warn_lock:
            if key not in _warned:
                _warned.add(key)
                logger.warning(
                    "%s is experimental, API may change without notice", key
                )
        return func(*args, **kwargs)

    return wrapper


def experimental_class(cls: type) -> type:
    _original_doc = cls.__doc__ or ""
    cls.__doc__ = f"[Experimental] {_original_doc}".strip()

    original_init = cls.__init__

    @functools.wraps(original_init)
    def new_init(self: Any, *args: Any, **kwargs: Any) -> None:
        key = f"{cls.__module__}.{cls.__qualname__}"
        with _warn_lock:
            if key not in _warned:
                _warned.add(key)
                logger.warning(
                    "%s is experimental, API may change without notice", key
                )
        original_init(self, *args, **kwargs)

    cls.__init__ = new_init
    return cls
