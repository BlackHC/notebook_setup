"""
Simple file-based object storage with automatic path generation and metadata tracking.

This module provides a caching and storage system that automatically generates
hierarchical file paths from function arguments, tracks metadata (git commit,
timestamps, wandb run info), and supports multiple serialization formats.

Key Features
------------
- Template-based path generation with automatic suffix for unlisted arguments
- Object identity mapping for complex objects (models, datasets)
- Versioning with timestamps
- Auto-format detection (JSON, pickle, PyTorch)
- Caching decorator with `.load()` and `.recompute()` accessors

Example Usage
-------------
>>> from blackhc.project.utils.simple_storage import Storage, template, identify
>>>
>>> storage = Storage("cache")
>>>
>>> # Direct save/load
>>> storage.save(result, "experiments", {"model": "cnn", "epochs": 10})
>>> result = storage.load("experiments", {"model": "cnn", "epochs": 10})
>>>
>>> # With caching decorator - remaining args auto-appended to path
>>> @storage.cache(template("{dataset}/{identifier}"))
... def train_model(dataset: str, model: str, epochs: int = 10):
...     return {"accuracy": 0.95}
>>>
>>> result = train_model("mnist", "cnn")  # Computes and caches
>>> result = train_model("mnist", "cnn")  # Loads from cache
>>> result = train_model("mnist", "cnn", _force_refresh=True)  # Force recompute
>>> result = train_model.load("mnist", "cnn")  # Load directly
"""

from __future__ import annotations

import enum
import functools
import inspect
import json
import os
import pickle
import string
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


def _escape_path_fragment(part: str) -> str:
    """Escape a path part for filesystem safety."""
    return urllib.parse.quote(part, safe=" +(,){:}[]%")


def _format_value(value: Any) -> str:
    """Convert a value to a path fragment."""
    if value is None:
        return "None"

    if isinstance(value, float) and value.is_integer():
        value = int(value)

    match value:
        case bool():
            return str(value)
        case int():
            return str(value)
        case float():
            return format(value, ".6g")
        case str():
            return _escape_path_fragment(value)
        case enum.Enum():
            return value.value if isinstance(value.value, str) else value.name
        case list():
            inner = ",".join(_format_value(v) for v in value)
            return f"[{inner}]"
        case tuple():
            inner = ",".join(_format_value(v) for v in value)
            return f"({inner})"
        case set() | frozenset():
            sorted_values = sorted(_format_value(v) for v in value)
            return "{" + ",".join(sorted_values) + "}"
        case dict():
            pairs = [_format_kwarg(k, v) for k, v in value.items()]
            return "{" + ",".join(pairs) + "}"
        case _:
            raise TypeError(
                f"Cannot format value of type {type(value).__name__}: {value!r}"
            )


def _format_kwarg(key: Any, value: Any) -> str:
    """Format a key=value pair for path."""
    formatted_key = _format_value(key)
    if value is None:
        return f"~{formatted_key}"
    if isinstance(value, bool):
        return f"+{formatted_key}" if value else f"-{formatted_key}"
    return f"{formatted_key}:{_format_value(value)}"


def _format_kwargs_fragment(kwargs: dict[Any, Any]) -> str:
    """Format a dict of kwargs as underscore-separated pairs."""
    kwarg_fragments = []
    for key, value in kwargs.items():
        # Special case: None key is shortened.
        if key is None:
            kwarg_fragments.append(_format_value(value))
        else:
            kwarg_fragments.append(_format_kwarg(key, value))
    return _list_to_path_fragment(kwarg_fragments)


def _list_to_path_fragment(args: list) -> str:
    """Convert a list of arguments to a path fragment."""
    return "_".join(_format_value(v) for v in args)


def _format_arg(arg: Any) -> str:
    if arg is None:
        return "__"
    elif isinstance(arg, dict):
        return _format_kwargs_fragment(arg)
    elif isinstance(arg, list):
        return _list_to_path_fragment(arg)
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
class PathSpec:
    """
    Specifies how to build a cache path from function arguments.

    Use template() to create instances.
    """

    template: str
    include_remaining: bool = True

    def build(
        self,
        bound_args: dict[str, Any],
        identifier: str,
        identity_registry: ObjectIdentityRegistry | None = None,
    ) -> list[str | dict]:
        """Build path parts from bound arguments."""
        resolved_args = {}
        for key, value in bound_args.items():
            if identity_registry and value in identity_registry:
                resolved_args[key] = identity_registry.get(value)
            else:
                resolved_args[key] = value

        format_dict = {k: _format_arg(v) for k, v in resolved_args.items()}
        format_dict["identifier"] = _escape_path_fragment(identifier)

        try:
            formatted = self.template.format(**format_dict)
        except KeyError as e:
            raise ValueError(f"Template references unknown argument: {e}") from e

        parts: list[str | dict] = [
            urllib.parse.unquote(p) for p in formatted.split("/") if p
        ]

        if self.include_remaining:
            formatter = string.Formatter()
            used_keys = {
                field_name.split(".")[0].split("[")[0]
                for _, field_name, _, _ in formatter.parse(self.template)
                if field_name is not None
            }
            if "identifier" not in used_keys:
                parts.append(format_dict["identifier"])
                used_keys.add("identifier")

            remaining = {k: v for k, v in format_dict.items() if k not in used_keys}
            if remaining:
                parts.append(remaining)

        return parts


def template(spec: str, *, include_remaining: bool = True) -> PathSpec:
    """
    Create a path specification from a template string.

    Args:
        spec: Format string with {arg} placeholders and optional {identifier}
        include_remaining: If True (default), args not in template are appended as suffix

    Examples:
        # Remaining args auto-appended
        template("{dataset}/{model}/{identifier}")
        # With dataset="mnist", model="cnn", epochs=10:
        # -> "mnist/cnn/my_func/epochs:10/"

        # Only use listed args
        template("{dataset}/{identifier}", include_remaining=False)
    """
    return PathSpec(template=spec, include_remaining=include_remaining)


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
        @storage.cache(template("{dataset}/{identifier}"))
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

    def _parts_to_path(self, *parts: str | dict | list | None) -> str:
        """Convert parts to a path string."""
        return "/".join(_format_value(part) for part in parts if part) + "/"

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
        base = self.root / self._parts_to_path(*parts)

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
        path_spec: PathSpec | str,
        *,
        fmt: Literal["json", "pkl", "pt", "auto"] = "auto",
    ):
        """
        Decorator to cache function results based on arguments.

        Args:
            path_spec: PathSpec or template string (converted to PathSpec)
            fmt: Forced format, or "auto" to detect

        Example:
            @storage.cache(template("{dataset}/{model}/{identifier}"))
            def train(dataset: str, model: str, epochs: int = 10):
                # epochs automatically included in path suffix
                return expensive_computation()

            result = train("mnist", "cnn", epochs=20)  # Compute & cache
            result = train("mnist", "cnn", epochs=20)  # Load from cache
            result = train("mnist", "cnn", _force_refresh=True)  # Force recompute
            result = train.load("mnist", "cnn")  # Load directly
            result = train.recompute("mnist", "cnn")  # Force recompute
        """
        if isinstance(path_spec, str):
            path_spec = template(path_spec)

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
