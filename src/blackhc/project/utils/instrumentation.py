"""
Instrumentation Module

This module provides utilities for instrumenting code to collect and manage runtime data.
It offers a flexible and thread-safe way to record, organize, and transform instrumentation
data across different scopes of execution.

Key components:
- Context: A dict-like class for storing and manipulating instrumentation data
- Instrumentation: A class that manages the collection and scoping of instrumentation data
- NullInstrumentation: A no-op version of Instrumentation for when instrumentation is disabled

Usage:
```
    from blackhc.project.utils.instrumentation import instrumentation

    with instrumentation.collect() as run_info:
        instrumentation.log(metric="value")
        with instrumentation.scope("subscope"):
            instrumentation.log(another_metric="another_value")
```

The module also supports disabling instrumentation via the NO_INSTRUMENTATION environment variable.
"""
import contextlib
import os
import threading
import types
import typing
from dataclasses import dataclass

try:
    import torch
except ImportError:
    raise ImportError("Package 'torch' is not installed. Please install it using 'pip install torch'")

T = typing.TypeVar("T")


class Context(dict[str, list]):
    """
    Wrapper class for scopes .
    """

    def walk(self) -> typing.Iterable[tuple[str, typing.Any]]:
        """Replay the collected instrumentation data.

        This returns an iterable of (name, value) pairs.
        """
        for key, values in self.items():
            for value in values:
                yield key, value

    def tree_walk(self) -> typing.Iterable[tuple[tuple[str], typing.Any]]:
        """Replay the collected instrumentation data.

        This returns an iterable of (path, value) pairs, where path is a tuple of keys.
        """
        for key, values in self.items():
            for value in values:
                yield (key,), value

                if isinstance(value, Context):
                    for sub_path, sub_value in value.tree_walk():
                        yield (key,) + sub_path, sub_value

    def transform_values(self, t_value: typing.Callable[[T], T]) -> "Context":
        """Transform the collected instrumentation data.

        This returns a new context with all values transformed by the given function.
        Recursively applies to sub-contexts.
        """
        return Context(
            {
                k: [
                    v.transform_values(t_value)
                    if isinstance(v, Context)
                    else t_value(v)
                    for v in vs
                ]
                for k, vs in self.items()
            }
        )

    def to_dict(self) -> dict:
        """Convert the collected instrumentation data to a dict.

        This returns a new context with all values transformed by the given function.
        Recursively applies to sub-contexts.
        """
        return {
            k: [v.to_dict() if isinstance(v, Context) else v for v in vs]
            for k, vs in self.items()
        }

    def to_numpy(self) -> "Context":
        """Convert the collected instrumentation data to a dict.

        This returns a new context with all values transformed by the given function.
        Recursively applies to sub-contexts.
        """
        return self.transform_values(
            t_value=lambda x: x.numpy(force=True) if isinstance(x, torch.Tensor) else x,
        )

    def to_simple_namespace(self) -> types.SimpleNamespace:
        """Convert the collected instrumentation data to a SimpleNamespace.

        This returns a new context with all values transformed by the given function.
        Recursively applies to sub-contexts.
        """
        return types.SimpleNamespace(
            **{
                k: [
                    v.to_simple_namespace() if isinstance(v, Context) else v for v in vs
                ]
                for k, vs in self.items()
            }
        )


@dataclass
class Instrumentation:
    """A simple helper that allows you to easily add instrumentation to your code.

    Example usage:

        ```
        from utils.instrumentation import instrument

        with instrument.collect() as run_info:
            # your code here
            instrument.record("loss", loss)

            with instrument("my_scope"):
                # your code here
                instrument.record("info", info)
        ```

        You can also use the `instrument` function as a decorator:

        ```
        @instrument("my_scope")
        def my_func():
            # your code here
            instrument.record("info", info)
        ```

        All scopes are stored as lists of SimpleNamespace objects
    """

    _thread_local: threading.local = threading.local()

    @property
    def _context(self) -> list[Context] | None:
        """Get the context instrumentation data"""
        return getattr(self._thread_local, "context", None)

    @property
    def is_active(self) -> bool:
        """Check if instrumentation is currently active."""
        return self._context is not None

    @contextlib.contextmanager
    def scope(self, key: str):
        """Create a new scope.

        If no data is being collected currently, this is a no-op.
        """
        outer = self._context
        if outer is None:
            yield
        else:
            inner = Context()
            self._thread_local.context = inner
            try:
                yield
                outer.setdefault(key, []).append(inner)
            finally:
                self._thread_local.context = outer

    def log(self, **kwargs):
        """Record a set of values.

        If no data is being collected, this is a no-op.
        """
        context = self._context
        if context is not None:
            for key, value in kwargs.items():
                context_value = value
                # detach if value is a torch.Tensor
                if isinstance(value, torch.Tensor):
                    context_value = value.detach()
                context.setdefault(key, []).append(context_value)

    def spy(self, **kwargs):
        """Record a single value.

        If no data is being collected, this is a no-op.
        """
        assert len(kwargs) == 1
        self.log(**kwargs)
        return next(iter(kwargs.values()))

    @contextlib.contextmanager
    def collect(self, enabled=True) -> Context:
        """Collect instrumentation data.

        This is a context manager that collects all instrumentation data
        that is recorded inside the context.
        """
        context = Context()

        if not enabled:
            yield context
            return

        outer = self._context
        self._thread_local.context = context
        try:
            yield context
            # If there is an outer, merge scope into it.
            if outer is not None:
                # Merge each key separately.
                for key, values in context.items():
                    outer.setdefault(key, []).extend(values)
        finally:
            self._thread_local.context = outer


class NullInstrumentation(Instrumentation):
    class _NullScope:
        def __enter__(self):
            pass

        def __exit__(self, *args):
            pass

        def __call__(self, func):
            return func

    _null_scope = _NullScope()

    @property
    def is_active(self) -> bool:
        return False

    def log(self, **kwargs):
        pass

    def spy(self, **kwargs):
        if len(kwargs) == 1:
            return next(iter(kwargs.values()))

    def scope(self, key: str):
        return NullInstrumentation._null_scope

    @contextlib.contextmanager
    def collect(self, enabled=True) -> Context:
        # Log a warning if enabled.
        if enabled:
            print("Warning: collecting data with NullInstrumentation")
        yield Context()


if os.getenv("NO_INSTRUMENTATION"):
    instrumentation = NullInstrumentation()
else:
    instrumentation = Instrumentation()
