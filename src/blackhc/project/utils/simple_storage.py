"""
Simple file-based object storage with automatic path generation and metadata tracking.

This module provides a caching and storage system that automatically generates
hierarchical file paths from function arguments, tracks metadata (git commit,
timestamps, wandb run info), and supports multiple serialization formats.

Key Features
------------
- Prefix schema for key:value path segments (e.g., dataset:mnist_split:train/)
- Object identity mapping for complex objects (models, datasets)
- Versioning with timestamps
- Auto-format detection (JSON, pickle, PyTorch)
- Caching decorator with `.load()` and `.recompute()` accessors
- Direct path loading with `load_from_path()` and `load_metadata_from_path()`

Example Usage
-------------
>>> from blackhc.project.utils.simple_storage import (
...     Storage, prefix_schema, identify, load_from_path
... )
>>>
>>> storage = Storage("cache")
>>>
>>> # Direct save/load
>>> storage.save(result, "experiments", {"model": "cnn", "epochs": 10})
>>> result = storage.load("experiments", {"model": "cnn", "epochs": 10})
>>>
>>> # Load from a known path
>>> data = load_from_path("cache/experiments/model:cnn/2024-01-01T12:00:00")
>>>
>>> # With prefix_schema - creates key:value path segments
>>> @storage.cache(prefix_schema("dataset", "split", "/", "model"))
... def compute_metrics(dataset: str, split: str, model: str, threshold: float = 0.5):
...     return {"accuracy": 0.95}
>>> # Path: dataset:X_split:Y/model:Z/compute_metrics/threshold:0.5/
>>>
>>> result = compute_metrics("mnist", "train", "cnn")  # Computes and caches
>>> result = compute_metrics("mnist", "train", "cnn")  # Loads from cache
>>> result = compute_metrics("mnist", "train", "cnn", _force_refresh=True)  # Force recompute
>>> result = compute_metrics.load("mnist", "train", "cnn")  # Load directly
"""

from __future__ import annotations

import enum
import functools
import inspect
import json
import os
import pickle
import urllib.parse
import weakref
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Literal, TypeVar

import jsonpickle

from blackhc.project.utils.collections.weakref_utils import WeakKeyIdMap
from blackhc.project.utils.collections.bimap import MappingBimap
from blackhc.project.experiment import get_git_head_commit_and_url

try:
    import wandb
except ImportError:
    wandb = None

try:
    import torch
except ImportError:
    torch = None


T = TypeVar("T")


# =============================================================================
# Value Formatting
# =============================================================================


def _escape(part: str) -> str:
    """Escape a path part for filesystem safety."""
    return urllib.parse.quote(part, safe=" +(,){:}[]%")


def _format_value(value: Any) -> str:
    """Convert a value to a path fragment."""
    if value is None:
        return "None"

    if isinstance(value, float) and value.is_integer():
        value = int(value)

    match value:
        case bool() | int() | str():
            value = str(value)
        case float():
            value = format(value, ".6g")
        case enum.Enum():
            value = value.value if isinstance(value.value, str) else value.name
        case list():
            inner = ",".join(_format_value(v) for v in value)
            value = f"[{inner}]"
        case tuple():
            inner = ",".join(_format_value(v) for v in value)
            value = f"({inner})"
        case set() | frozenset():
            sorted_values = sorted(_format_value(v) for v in value)
            value = "{" + ",".join(sorted_values) + "}"
        case dict():
            pairs = [_format_kv(k, v) for k, v in value.items()]
            value = "{" + ",".join(pairs) + "}"
        case _:
            raise TypeError(
                f"Cannot format value of type {type(value).__name__}: {value!r}"
            )
    
    return _escape(value)


def _format_kv(key: Any, value: Any) -> str:
    """Format a key=value pair for path."""
    formatted_key = _format_value(key)
    if value is None:
        return f"~{formatted_key}"
    if isinstance(value, bool):
        return f"+{formatted_key}" if value else f"-{formatted_key}"
    return f"{formatted_key}:{_format_value(value)}"


def _format_kvs(kwargs: dict[Any, Any]) -> str:
    """Format a dict of kwargs as underscore-separated pairs."""
    kvs = []
    for key, value in kwargs.items():
        # Special case: None key is shortened.
        if key is None:
            kvs.append(_format_value(value))
        else:
            kvs.append(_format_kv(key, value))
    return "_".join(kvs)


def _format_arg(arg: Any) -> str:
    if arg is None:
        return "__"
    elif isinstance(arg, dict):
        return _format_kvs(arg)
    elif isinstance(arg, list):
        return "_".join(_format_value(v) for v in arg)
    else:
        return _format_value(arg)


# =============================================================================
# Object Identity Registry
# =============================================================================


class ObjectIdentityRegistry:
    """
    Maps objects to path fragments for cache key generation.

    Useful when you want to pass complex objects (models, datasets) as function
    arguments but have them represented as simple strings in cache paths.

    Example:
        registry = ObjectIdentityRegistry()

        model = load_pretrained_model("gpt2")
        registry.identify(model, "gpt2")

        # When building cache paths, model becomes "gpt2"
    """

    def __init__(self):
        self._bimap: MappingBimap[object, str] = MappingBimap(
            WeakKeyIdMap(), weakref.WeakValueDictionary()
        )

    def identify(self, obj: T, path_fragment: str) -> T:
        """Associate an object with a path fragment. Returns the object for chaining."""
        self._bimap.update(obj, path_fragment)
        return obj

    def get(self, obj: Any, default: Any = None) -> str | None:
        """Get the path fragment for an object, or default if not registered."""
        result = self._bimap.get_value(obj)
        return result if result is not None else default

    def resolve(self, obj: Any) -> Any:
        """Return the path fragment if registered, otherwise the original object."""
        result = self._bimap.get_value(obj)
        return result if result is not None else obj

    def __contains__(self, obj: Any) -> bool:
        return obj in self._bimap

    def __getitem__(self, obj: Any) -> str:
        result = self._bimap.get_value(obj)
        if result is None:
            raise KeyError(obj)
        return result


# Global registry
object_identities = ObjectIdentityRegistry()


def identify(obj: T, path_fragment: str) -> T:
    """Register an object with a path fragment in the global registry."""
    return object_identities.identify(obj, path_fragment)


# =============================================================================
# Metadata
# =============================================================================


@dataclass
class Metadata:
    """Metadata for a stored artifact."""

    timestamp: str
    git_commit: str = ""
    git_url: str = ""
    wandb_id: str = ""
    wandb_url: str = ""
    path_parts: list[str] = field(default_factory=list)


def _collect_metadata(parts: list[str]) -> Metadata:
    """Collect metadata for current environment."""
    git_commit, git_url = get_git_head_commit_and_url(os.getcwd())

    wandb_id = wandb_url = None
    if wandb and wandb.run:
        wandb_id = wandb.run.id
        wandb_url = wandb.run.get_url()

    return Metadata(
        timestamp=datetime.now().isoformat(),
        git_commit=git_commit,
        git_url=git_url,
        wandb_id=wandb_id,
        wandb_url=wandb_url,
        path_parts=list(parts),
    )


# =============================================================================
# Path Building
# =============================================================================


@dataclass
class PrefixPathSpec:
    """
    Specifies how to build a cache path from function arguments using prefix groups.

    Use prefix_schema() to create instances.
    """

    prefix_args: tuple[str | list | set | tuple, ...]
    include_remaining: bool = True

    def build(
        self,
        bound_args: dict[str, Any],
        identifier: str,
        identity_registry: ObjectIdentityRegistry | None = None,
    ) -> list[str | dict | list]:
        """Build path parts from bound arguments using prefix schema."""
        # Resolve object identities
        resolved_args = {}
        for key, value in bound_args.items():
            if identity_registry and value in identity_registry:
                resolved_args[key] = identity_registry.get(value)
            else:
                resolved_args[key] = value

        parts: list[str | dict | list] = []
        prefix_dict: dict[str, Any] = {}
        used_args: set[str] = set()

        for arg in self.prefix_args:
            if arg == "/":
                if prefix_dict:
                    parts.append(prefix_dict)
                    prefix_dict = {}
            elif isinstance(arg, str):
                prefix_dict[arg] = resolved_args[arg]
                used_args.add(arg)
            elif isinstance(arg, (set, tuple)):
                if prefix_dict:
                    parts.append(prefix_dict)
                    prefix_dict = {}
                for key in arg:
                    prefix_dict[key] = resolved_args[key]
                    used_args.add(key)
                parts.append(prefix_dict)
                prefix_dict = {}
            elif isinstance(arg, list):
                if prefix_dict:
                    parts.append(prefix_dict)
                    prefix_dict = {}

                prefix_list = []
                for key in arg:
                    prefix_list.append(resolved_args[key])
                    used_args.add(key)
                parts.append(prefix_list)
            else:
                raise ValueError(f"Invalid prefix arg: {arg}")

        if prefix_dict:
            parts.append(prefix_dict)

        parts.append(identifier)

        if self.include_remaining:
            suffix_dict = {
                arg: resolved_args[arg]
                for arg in resolved_args
                if arg not in used_args
            }
            if suffix_dict:
                parts.append(suffix_dict)

        return parts


def prefix_schema(
    *prefix_args: str | list | set | tuple, include_remaining: bool = True
) -> PrefixPathSpec:
    """
    Build a path schema from prefix arguments.

    Any "/" in the prefix args is treated as a separator between different parts
    of the path. Arguments are grouped into dicts (for key:value pairs) or lists.

    Args:
        *prefix_args: Sequence of:
            - str: argument name to include in current dict group
            - "/": separator to start a new path segment
            - set/tuple: group of keys to include as their own dict segment
            - list: group of keys whose values form a list segment
        include_remaining: If True (default), args not in prefix_args are appended
            as a suffix dict. If False, only explicitly listed args are included.

    Returns:
        PrefixPathSpec that can be used with @storage.cache()

    Example:
        >>> schema = prefix_schema("dataset", "split", "/", "model")
        >>> parts = schema.build(
        ...     {"dataset": "mnist", "split": "train", "model": "cnn", "epochs": 10},
        ...     "my_func"
        ... )
        >>> # parts = [{"dataset": "mnist", "split": "train"}, {"model": "cnn"},
        >>> #          "my_func", {"epochs": 10}]
        >>> # Creates path: dataset:mnist_split:train/model:cnn/my_func/epochs:10/

        >>> @storage.cache(prefix_schema("dataset", "/", "model"))
        ... def train(dataset: str, model: str, epochs: int = 10):
        ...     return result
        >>> # Path: dataset:mnist/model:cnn/train/epochs:10/

        Using tuples/sets for explicit grouping:
        >>> schema = prefix_schema(("dataset", "split"), "/", ("model",))
        >>> # Same as above but more explicit about boundaries

        Using lists for value-only paths (no key names):
        >>> schema = prefix_schema(["dataset", "split"], "/", "model")
        >>> # Creates path: mnist_train/model:cnn/my_func/epochs:10/

        Without remaining args:
        >>> schema = prefix_schema("dataset", "/", "model", include_remaining=False)
        >>> # Only dataset and model in path, other args ignored
    """
    return PrefixPathSpec(prefix_args=prefix_args, include_remaining=include_remaining)


# =============================================================================
# Storage Class
# =============================================================================


class Storage:
    """
    File-based storage with automatic path generation and metadata.

    Features:
    - Hierarchical paths from function arguments
    - Object identity resolution for complex objects
    - Automatic metadata (git, wandb, timestamp)
    - Versioning with timestamps
    - Multiple serialization formats (auto-detected)

    Example:
        storage = Storage("cache/experiments")

        # Direct save/load
        storage.save(result, "mnist", {"model": "cnn", "epochs": 10})
        result = storage.load("mnist", {"model": "cnn", "epochs": 10})

        # With decorator
        @storage.cache(prefix_schema("dataset", "/", "model"))
        def train(dataset: str, model: str, epochs: int = 10):
            return expensive_computation()
    """

    def __init__(
        self,
        root: str | Path,
        identity_registry: ObjectIdentityRegistry | None = None,
        *,
        verbose: bool = True,
    ):
        self.root = Path(root)
        self.identity_registry = identity_registry or object_identities
        self.verbose = verbose

    def get_path(
        self,
        *parts: str | dict | list | None,
        version: bool | str | datetime = False,
        for_save: bool = False,
    ) -> Path:
        """Get the full path for given parts and version.

        Args:
            *parts: Path components
            version: False=no versioning, True=auto (latest for load, now for save),
                     str/datetime=explicit version
            for_save: If True, version=True means new timestamp; if False, means latest
        """
        nonempty_parts = [_format_value(part) for part in parts if part]
        base = self.root.joinpath(*nonempty_parts)

        if version is False:
            return base
        elif version is True:
            if for_save:
                return base / datetime.now().isoformat()
            else:
                # Load: get latest
                if not base.exists():
                    raise FileNotFoundError(f"No versions at {base}")
                subdirs = sorted(d for d in base.iterdir() if d.is_dir())
                if not subdirs:
                    raise FileNotFoundError(f"No version subdirs in {base}")
                return subdirs[-1]
        elif isinstance(version, datetime):
            return base / version.isoformat()
        else:
            return base / str(version)

    def _find_data_file(self, path: Path) -> Path:
        """Find the data file in a storage directory."""
        candidates = list(path.glob("data.*"))
        if not candidates:
            raise FileNotFoundError(f"No data file in {path}")
        if len(candidates) > 1:
            raise RuntimeError(f"Multiple data files: {candidates}")
        return candidates[0]

    def _detect_format(
        self, obj: Any
    ) -> tuple[Literal["json", "pkl", "pt"], bytes | None]:
        """
        Detect best format for object, returning (format, optional pre-serialized bytes).

        Returns pre-serialized pickle bytes when available to avoid double serialization.
        """
        if torch and isinstance(obj, torch.Tensor):
            return "pt", None
        try:
            pickled_bytes = pickle.dumps(obj)
            if len(pickled_bytes) < 256 * 1024:
                if json.loads(json.dumps(obj)) == obj:
                    return "json", pickled_bytes
            return "pkl", pickled_bytes
        except (TypeError, OverflowError, AssertionError, AttributeError):
            return "pkl", pickle.dumps(obj)

    def prepare_output(
        self,
        *parts: str | dict | list | None,
        version: bool | str | datetime = False,
        fmt: Literal["pkl", "json", "pt", "auto"] = "pkl",
    ) -> tuple[Path, Metadata]:
        """
        Prepare output path without writing data.

        Useful for streaming writes or custom serialization.

        Returns:
            Tuple of (data_file_path, metadata)
        """
        path = self.get_path(*parts, version=version, for_save=True)
        path.mkdir(parents=True, exist_ok=True)

        metadata = _collect_metadata(list(parts))
        (path / "meta.json").write_text(
            jsonpickle.encode(asdict(metadata), unpicklable=False)
        )

        return path / f"data.{fmt}", metadata

    def save(
        self,
        obj: Any,
        *parts: str | dict | list | None,
        version: bool | str | datetime = False,
        fmt: Literal["json", "pkl", "pt", "auto"] = "auto",
    ) -> tuple[Path, Metadata]:
        """
        Save an object with automatic path generation and metadata.

        Args:
            obj: Object to save
            *parts: Path components (strings, dicts, lists)
            version: False=no version dir, True=new timestamp, str/datetime=explicit
            fmt: Serialization format (auto-detected if "auto")

        Returns:
            Tuple of (path, metadata)
        """
        pickled_bytes = None
        if fmt == "auto":
            fmt, pickled_bytes = self._detect_format(obj)

        path, metadata = self.prepare_output(*parts, version=version, fmt=fmt)

        if fmt == "json":
            path.write_text(jsonpickle.encode(obj, indent=2, keys=True))
        elif fmt == "pkl":
            if pickled_bytes is None:
                pickled_bytes = pickle.dumps(obj)
            path.write_bytes(pickled_bytes)
        elif fmt == "pt":
            if not torch:
                raise ImportError("torch required for .pt format")
            torch.save(obj, path)

        return path, metadata

    def load(
        self,
        *parts: str | dict | list | None,
        version: bool | str | datetime = False,
    ) -> Any:
        """
        Load an object, auto-detecting format.

        Args:
            *parts: Path components
            version: False=no version, True=latest, or explicit str/datetime

        Returns:
            The loaded object
        """
        path = self.get_path(*parts, version=version)
        data_file = self._find_data_file(path)

        suffix = data_file.suffix
        if suffix == ".json":
            return jsonpickle.decode(data_file.read_text(), keys=True)
        elif suffix == ".pkl":
            return pickle.loads(data_file.read_bytes())
        elif suffix == ".pt":
            if not torch:
                raise ImportError("torch required for .pt format")
            return torch.load(data_file)
        else:
            raise ValueError(f"Unknown format: {suffix}")

    def load_metadata(
        self,
        *parts: str | dict | list | None,
        version: bool | str | datetime = False,
    ) -> Metadata:
        """Load only metadata without loading the data."""
        path = self.get_path(*parts, version=version)
        meta_file = path / "meta.json"
        return Metadata(**json.loads(meta_file.read_text()))

    def exists(
        self,
        *parts: str | dict | list | None,
        version: bool | str | datetime = False,
    ) -> bool:
        """Check if cached data exists."""
        try:
            path = self.get_path(*parts, version=version)
            self._find_data_file(path)
            return True
        except FileNotFoundError:
            return False

    def load_all_metadata(self) -> dict[str, dict]:
        """Scans for meta.json files and loads all metadata."""
        meta_data = {}
        for dirpath, _, filenames in os.walk(self.root):
            if "meta.json" in filenames:
                meta_file = Path(dirpath) / "meta.json"
                path = str(meta_file.parent) + "/"
                meta_data[path] = json.loads(meta_file.read_text())
        return meta_data

    def cache(
        self,
        path_spec: PrefixPathSpec,
        *,
        fmt: Literal["json", "pkl", "pt", "auto"] = "auto",
    ):
        """
        Decorator to cache function results based on arguments.

        Args:
            path_spec: PrefixPathSpec created via prefix_schema()
            fmt: Forced format, or "auto" to detect

        Example:
            @storage.cache(prefix_schema("dataset", "/", "model"))
            def train(dataset: str, model: str, epochs: int = 10):
                # epochs automatically included in path suffix
                return expensive_computation()

            result = train("mnist", "cnn", epochs=20)  # Compute & cache
            result = train("mnist", "cnn", epochs=20)  # Load from cache
            result = train("mnist", "cnn", _force_refresh=True)  # Force recompute
            result = train.load("mnist", "cnn")  # Load directly
            result = train.recompute("mnist", "cnn")  # Force recompute
        """

        def decorator(fn: Callable[..., T]) -> Callable[..., T]:
            sig = inspect.signature(fn)
            identifier = fn.__qualname__

            def _build_parts(args: tuple, kwargs: dict) -> list:
                bound = sig.bind(*args, **kwargs)
                bound.apply_defaults()
                return path_spec.build(
                    bound.arguments,
                    identifier,
                    self.identity_registry,
                )

            @functools.wraps(fn)
            def wrapper(*args, _force_refresh: bool = False, **kwargs) -> T:
                parts = _build_parts(args, kwargs)

                if not _force_refresh:
                    try:
                        result = self.load(*parts, version=True)
                        if self.verbose:
                            print("📦 Loaded from cache")
                        return result
                    except FileNotFoundError:
                        pass

                result = fn(*args, **kwargs)
                path, _ = self.save(result, *parts, version=True, fmt=fmt)
                if self.verbose:
                    print(f"📦 Cached to {path}")
                return result

            new_params = [
                *sig.parameters.values(),
                inspect.Parameter(
                    "_force_refresh",
                    inspect.Parameter.KEYWORD_ONLY,
                    default=False,
                ),
            ]
            wrapper.__signature__ = sig.replace(parameters=new_params)

            def load_cached(
                *args, _version: bool | str | datetime = True, **kwargs
            ) -> T:
                parts = _build_parts(args, kwargs)
                return self.load(*parts, version=_version)

            load_params = [
                *sig.parameters.values(),
                inspect.Parameter(
                    "_version",
                    inspect.Parameter.KEYWORD_ONLY,
                    default=True,
                ),
            ]
            load_cached.__signature__ = sig.replace(parameters=load_params)

            def recompute(*args, **kwargs) -> T:
                parts = _build_parts(args, kwargs)
                result = fn(*args, **kwargs)
                self.save(result, *parts, version=True, fmt=fmt)
                return result

            def get_prefix_path(*args, _version=False, **kwargs) -> Path:
                parts = _build_parts(args, kwargs)
                return self.get_path(*parts, version=_version)

            wrapper.load = load_cached
            wrapper.recompute = recompute
            wrapper.get_prefix_path = get_prefix_path

            return wrapper

        return decorator


# =============================================================================
# Convenience Functions
# =============================================================================


def load_from_path(path: str | Path) -> Any:
    """
    Load data directly from a storage directory path.

    Convenience function for loading from a known path without creating
    a Storage instance with parts.

    Args:
        path: Path to a storage directory containing data.* and meta.json

    Returns:
        The loaded object

    Example:
        data = load_from_path("cache/experiments/model:cnn/2024-01-01T12:00:00")
    """
    return Storage(path, verbose=False).load()


def load_metadata_from_path(path: str | Path) -> Metadata:
    """
    Load metadata directly from a storage directory path.

    Convenience function for loading metadata from a known path without
    loading the full data.

    Args:
        path: Path to a storage directory containing meta.json

    Returns:
        Metadata object

    Example:
        metadata = load_metadata_from_path("cache/experiments/model:cnn/2024-01-01T12:00:00")
        print(metadata.timestamp, metadata.git_commit)
    """
    return Storage(path, verbose=False).load_metadata()
