"""
Simple file-based object storage.
"""

import enum
import functools
import inspect
import json
import jsonpickle
import os
import pickle
import sys
import typing
import urllib.parse

from dataclasses import dataclass
from datetime import datetime

import blackhc.project
from blackhc.project.experiment import get_git_head_commit_and_url


try:
    import wandb
except ImportError:
    wandb = None


try:
    import torch
except ImportError:
    torch = None


def get_module_name(f):
    """Get the name of the module of an object that has a __module__."""
    module = sys.modules[f.__module__]
    return module.__spec__.name if hasattr(module, "__spec__") else module.__name__


def get_callable_full_name(f: typing.Callable):
    """Get the full name of a callable, including the module name."""
    # return f"{get_module_name(f)}:{f.__qualname__}"
    return f.__qualname__


def escape_path_part(part: str):
    """Escape a path part."""
    return urllib.parse.quote(part, safe=" +(,){:}")


def value_to_path_fragment(
    value: float | str | int | list | dict | tuple | enum.Enum | None,
) -> str:
    """Convert a value to a path part."""
    # Convert to simpler types if possible.
    if value is None:
        value = str(value)
    elif isinstance(value, float) and value.is_integer():
        value = int(value)
    elif isinstance(value, enum.Enum):
        value = value.value if isinstance(value.value, str) else value.name

    if isinstance(value, float):
        value = format(
            value, ".6g"
        )  # Use general format with up to 6 significant digits
    elif isinstance(value, list):
        value = "+".join(map(value_to_path_fragment, value))
    elif isinstance(value, tuple):
        value = "(" + ",".join(map(value_to_path_fragment, value)) + ")"
    elif isinstance(value, dict):
        value = (
            "{"
            + ",".join(
                f"{value_to_path_fragment(k)}:{value_to_path_fragment(v)}"
                for k, v in value.items()
            )
            + "}"
        )
    elif isinstance(value, int):
        value = str(value)
    elif isinstance(value, str):
        pass
    else:
        raise ValueError(f"Unsupported value type: {type(value)}")
    return escape_path_part(str(value))


def dict_to_path_fragment(kwargs: dict):
    """Convert a dictionary of keyword arguments to a path fragment.

    Args:
        kwargs (dict): The dictionary of keyword arguments to convert to a path fragment.
        incl_keys (bool): Whether to include the keys in the path fragment.
    Returns:
        str: The path fragment.
    """
    kwarg_fragments = []
    for key, value in kwargs.items():
        if key is not None:
            kwarg_fragments.append(f"{key}:{value_to_path_fragment(value)}")
        else:
            kwarg_fragments.append(f"{value_to_path_fragment(value)}")
    return "_".join(kwarg_fragments)


def list_to_path_fragment(args: list):
    """Convert a list of arguments to a path fragment."""
    return "_".join(map(value_to_path_fragment, args))

def generate_path(*parts) -> str:
    """
    Generates a path based on the given parts.

    Args:
        identifier (str or callable): The identifier to be used in the path.
        **kwargs: Additional keyword arguments to be included in the path.

    Returns:
        str: The generated path.
    """
    path_parts = []
    for part in parts:
        if part is None:
            fragment = ""
        elif isinstance(part, dict):
            fragment = dict_to_path_fragment(part)
        elif isinstance(part, list):
            fragment = list_to_path_fragment(part)
        else:
            fragment = value_to_path_fragment(part)
        if fragment != "":
            path_parts.append(fragment)

    if parts and parts[-1] is None:
        path_parts.append("")

    return "/".join(path_parts)


def collect_metadata(*parts) -> dict[str]:
    """Collect metadata for the current run."""
    head_commit, github_url = get_git_head_commit_and_url(os.getcwd())
    # If wandb is running, get the wandb id and url
    wandb_id = None
    wandb_url = None
    if wandb is not None and wandb.run is not None:
        wandb_id = wandb.run.id
        wandb_url = wandb.run.get_url()

    metadata = dict(
        timestamp=datetime.now().isoformat(),
        git=dict(commit=head_commit, url=github_url),
        wandb=dict(id=wandb_id, url=wandb_url),
        parts=parts,
    )
    return metadata


class Timestamp(enum.Enum):
    """Whether to use a timestamp or not (and if so, whether to use the current timestamp or the latest one)."""

    NONE = "none"
    NOW = "now"
    LATEST = "latest"


def get_prefix_path(
    *parts, root: str = "", timestamp: Timestamp | str | datetime = Timestamp.NONE
) -> str:
    """Get the prefix path for the given parts, root, and timestamp. The path is used as base path for saving and loading.
    If timestamp != Timestamp.NONE, we create a directory and use timestamps as subdirectories.

    For Timestamp.LATEST, we find the latest subdirectory in the prefix path (last by sorting).

    Args:
        parts (list): The parts of the path.
        root (str): The root of the path.
        timestamp (Timestamp | str | datetime): The timestamp of the path.
    Returns:
        str: The prefix path.
    """
    base_prefix_path = os.path.join(root, generate_path(*parts))
    match timestamp:
        case Timestamp.NONE:
            prefix_path = base_prefix_path
        case Timestamp.LATEST:
            # Find all subdirs in the prefix path
            subdirs = sorted(
                [
                    d
                    for d in os.listdir(base_prefix_path)
                    if os.path.isdir(os.path.join(base_prefix_path, d))
                ]
            )
            if not subdirs:
                raise FileNotFoundError(
                    "No subdirectories found in the prefix path", base_prefix_path
                )
            latest_subdir = subdirs[-1]
            prefix_path = os.path.join(base_prefix_path, latest_subdir, "")
        case Timestamp.NOW:
            timestamp = datetime.now().isoformat()
            prefix_path = os.path.join(base_prefix_path, timestamp, "")
        case str():
            prefix_path = os.path.join(base_prefix_path, timestamp, "")
        case datetime():
            prefix_path = os.path.join(base_prefix_path, timestamp.isoformat(), "")
        case _:
            raise ValueError(f"Invalid timestamp: {timestamp}")
    return prefix_path


def _combine_path(prefix_path, ext) -> str:
    if prefix_path.endswith("/"):
        return f"{prefix_path}{ext}"
    return f"{prefix_path}.{ext}"


def _sync_timestamp(
    timestamp: Timestamp | str | datetime, master_timestamp: str | datetime
) -> str | datetime:
    match timestamp:
        case Timestamp.NOW:
            return (
                master_timestamp.isoformat()
                if isinstance(master_timestamp, datetime)
                else master_timestamp
            )
        case _:
            return timestamp


def _save_metadata(
    *parts, root: str = "", timestamp: Timestamp | str | datetime = Timestamp.NONE
) -> str:
    metadata = collect_metadata(*parts)
    prefix_path = get_prefix_path(
        *parts, root=root, timestamp=_sync_timestamp(timestamp, metadata["timestamp"])
    )
    metadata_string = jsonpickle.encode(metadata, unpicklable=False)
    os.makedirs(os.path.dirname(prefix_path), exist_ok=True)
    with open(_combine_path(prefix_path, "meta.json"), "wt", encoding="utf-8") as f:
        f.write(metadata_string)
    return prefix_path


def load_metadata(
    *parts, root: str = "", timestamp: Timestamp | str | datetime = Timestamp.NONE
) -> dict:
    assert timestamp != Timestamp.NOW
    prefix_path = get_prefix_path(*parts, root=root, timestamp=timestamp)
    with open(_combine_path(prefix_path, "meta.json"), "rt", encoding="utf-8") as f:
        return json.load(f)


def save_pkl(
    obj, *parts, root: str = "", timestamp: Timestamp | str | datetime = Timestamp.NONE
) -> str:
    prefix_path = _save_metadata(*parts, root=root, timestamp=timestamp)
    with open(_combine_path(prefix_path, "data.pkl"), "wb") as f:
        pickle.dump(obj, f)
    return prefix_path


def load_pkl(
    *parts, root: str = "", timestamp: Timestamp | str | datetime = Timestamp.NONE
):
    assert timestamp != Timestamp.NOW
    prefix_path = get_prefix_path(*parts, root=root, timestamp=timestamp)
    with open(_combine_path(prefix_path, "data.pkl"), "rb") as f:
        return pickle.load(f)


def save_pt(
    obj, *parts, root: str = "", timestamp: Timestamp | str | datetime = Timestamp.NONE
) -> str:
    prefix_path = _save_metadata(*parts, root=root, timestamp=timestamp)
    with open(_combine_path(prefix_path, "data.pt"), "wb") as f:
        torch.save(obj, f)
    return prefix_path


def load_pt(
    *parts, root: str = "", timestamp: Timestamp | str | datetime = Timestamp.NONE
):
    assert timestamp != Timestamp.NOW
    prefix_path = get_prefix_path(*parts, root=root, timestamp=timestamp)
    with open(_combine_path(prefix_path, "data.pt"), "rb") as f:
        return torch.load(f)


def save_json(
    obj, *parts, root: str = "", timestamp: Timestamp | str | datetime = Timestamp.NONE
) -> str:
    prefix_path = _save_metadata(*parts, root=root, timestamp=timestamp)
    with open(_combine_path(prefix_path, "data.json"), "wt", encoding="utf-8") as f:
        json.dump(obj, f)
    return prefix_path


def load_json(
    *parts, root: str = "", timestamp: Timestamp | str | datetime = Timestamp.NONE
):
    assert timestamp != Timestamp.NOW
    prefix_path = get_prefix_path(*parts, root=root, timestamp=timestamp)
    with open(_combine_path(prefix_path, "data.json"), "rt", encoding="utf-8") as f:
        return json.load(f)


def save_pkl_or_json(
    obj, *parts, root: str = "", timestamp: Timestamp | str | datetime = Timestamp.NONE
) -> str:
    prefix_path = _save_metadata(*parts, root=root, timestamp=timestamp)

    # Pickle the object into bytes
    pickled_obj = pickle.dumps(obj)

    # Check if the pickled size is less than 256KB
    if len(pickled_obj) < 256 * 1024:
        try:
            # Try to save as JSON
            json_obj = json.loads(json.dumps(obj))
            assert json_obj == obj
            with open(
                _combine_path(prefix_path, "data.json"), "wt", encoding="utf-8"
            ) as f:
                json.dump(obj, f)
            return prefix_path
        except (TypeError, OverflowError, AssertionError):
            # If it fails, save as pickle instead.
            pass

    with open(_combine_path(prefix_path, "data.pkl"), "wb") as f:
        f.write(pickled_obj)
    return prefix_path


def load(
    *parts, root: str = "", timestamp: Timestamp | str | datetime = Timestamp.NONE
):
    assert timestamp != Timestamp.NOW
    prefix_path = get_prefix_path(*parts, root=root, timestamp=timestamp)

    # Find the *data.* file (can either end in pkl, json or ot)
    data_files = [
        f
        for f in os.listdir(os.path.dirname(prefix_path))
        if f.startswith(os.path.basename(prefix_path))
        and f.endswith(("data.pt", "data.pkl", "data.json"))
    ]

    if len(data_files) == 1:
        data_file = os.path.join(os.path.dirname(prefix_path), data_files[0])
        if data_file.endswith(".pkl"):
            with open(data_file, "rb") as f:
                return pickle.load(f)
        elif data_file.endswith(".json"):
            with open(data_file, "rt", encoding="utf-8") as f:
                return json.load(f)
        elif data_file.endswith(".pt"):
            with open(data_file, "rb") as f:
                return torch.load(f)
        else:
            raise ValueError(f"Unsupported file type: {data_file}")
    elif len(data_files) > 1:
        raise RuntimeError(
            "Multiple data files found for the same prefix path", data_files
        )
    else:
        raise FileNotFoundError("No data file found for the prefix path", prefix_path)


@dataclass
class PartLiteral:
    """A literal part of the path."""

    value: str


@dataclass
class PartKW:
    """A literal keyword argument part of the path."""

    key: str
    value: str


class PartIdentifier:
    """The identifier part of the path, which gets filled in by the identifier argument."""
    
    pass


def part_schema(*schema_parts: list):
    """Build a path schema from a list of schema parts.

    Example:
        schema = part_schema(
            PartSchemaLiteral("fixed_dataset"),
            ("arg1", PartSchemaKW("split", "test"), "arg2")),
            PartSchemaIdentifier
        )

        assert schema({"arg1": "value1", "arg2": "value2"}, "test_id") == [
            "fixed_dataset",
            {"arg1": "value1", "split": "test", "arg2": "value2"},
            "test_id"
        ]

        And creates "{root}/fixed_dataset/arg1:{arg1}_split:test_arg2:arg{2}" as prefix path (excl timestamps).
    """
    
    def params_to_parts(bound_arguments: dict, identifier: str) -> list[str]:
        """Convert a list of arguments and a dictionary of keyword arguments to a list of path parts using the given schema."""
        def missing_arg_error(arg):
            raise ValueError(
                "Missing schema arg:",
                arg,
                " in ",
                schema_parts,
                " for ",
                bound_arguments,
                "(or use $identifier for the identifier"
                "or PartSchemaLiteral or PartSchemaKW for literal parts)."
            )
        
        parts = []
        for schema_part in schema_parts:
            if isinstance(schema_part, PartLiteral):
                part = schema_part.value
            elif isinstance(schema_part, PartKW):
                part = {schema_part.key: schema_part.value}
            elif schema_part == PartIdentifier or schema_part == "$identifier":
                part = identifier
            elif isinstance(schema_part, str):
                if schema_part in bound_arguments:
                    part = bound_arguments[schema_part]
                else:
                    missing_arg_error(schema_part)
            elif isinstance(schema_part, (set, tuple)):
                part = {}
                for arg in schema_part:
                    if isinstance(arg, PartLiteral):
                        if None in part:
                            raise ValueError("Multiple PartLiteral (or Identifier) not allowed in set schema parts")
                        part[None] = arg.value
                    elif isinstance(arg, PartKW):
                        part[arg.key] = arg.value
                    elif arg == PartIdentifier or arg == "$identifier":
                        part[None] = identifier
                    elif arg in bound_arguments:
                        part[arg] = bound_arguments[arg]
                    else:
                        missing_arg_error(schema_part)
            elif isinstance(schema_part, list):
                part = []
                for arg in schema_part:
                    if isinstance(arg, PartLiteral):
                        part.append(arg.value)
                    elif arg in bound_arguments:
                        part.append(bound_arguments[arg])
                    elif isinstance(arg, PartKW):
                        raise ValueError("PartKW is not allowed in list schema parts")
                    else:
                        missing_arg_error(schema_part)
            else:
                raise ValueError("Invalid schema part:", schema_part)
            parts.append(part)
        return parts

    return params_to_parts


def prefix_schema(*prefix_args: list[str]) -> typing.Callable[[dict, dict], list[str]]:
    """Build a path schema from a list of prefix arguments.
    
    Any "/" in the prefix args is treated as a separator between different parts of the path.

    Example:
        schema = prefix_schema(["arg1", "/" "arg2"])
        parts = schema({"arg1": "value1", "arg2": "value2", "arg3": "value3"}, "test_id")
        assert parts == [
            {"arg1": "value1"}, 
            {"arg2": "value2"},
            "test_id",
            {"arg3": "value3"}
        ]

        And creates "{root}/arg1:{arg1}/arg2:arg{2}/test_id/arg3:value3" as prefix path (excl timestamps).
    """

    def params_to_parts(bound_arguments: dict, identifier: str) -> list[str]:
        """Build a path using prefix args, identifier, and remaining args."""
        parts = []
        prefix_dict = {}
        used_args = set()
        for arg in prefix_args:
            if arg == "/":
                if prefix_dict:
                    parts.append(prefix_dict)
                    prefix_dict = {}
            elif isinstance(arg, str):
                prefix_dict[arg] = bound_arguments[arg]
                used_args.add(arg)
            elif isinstance(arg, (set, tuple)):
                if prefix_dict:
                    parts.append(prefix_dict)
                    prefix_dict = {}
                for key in arg:
                    prefix_dict[key] = bound_arguments[key]
                    used_args.add(key)
                parts.append(prefix_dict)
                prefix_dict = {}
            elif isinstance(arg, list):
                if prefix_dict:
                    parts.append(prefix_dict)
                    prefix_dict = {}
                
                prefix_list = []
                for key in arg:
                    prefix_list.append(bound_arguments[key])
                    used_args.add(key)
                parts.append(prefix_list)
            else:
                raise ValueError(f"Invalid prefix arg: {arg}")
        if prefix_dict:
            parts.append(prefix_dict)
        parts.append(identifier)
        suffix_dict = {
            arg: bound_arguments[arg]
            for arg in bound_arguments
            if arg not in used_args
        }
        parts.append(suffix_dict)
        return parts

    return params_to_parts


def template_schema(template: str) -> typing.Callable[[dict, dict], list[str]]:
    """Format the parts of the path using the given template by calling `.format` on it with the bound arguments and the identifier.

    Example:
        schema = template_schema("{arg1}-{arg2}/{identifier}")
        parts = schema({"arg1": "value1", "arg2": "value2"}, "test_id")
        assert parts == [
            "value1:value2",
            "test_id"
        ]

        And creates "{root}/value1-value2/test_id" as prefix path (excl timestamps).
    """

    def params_to_parts(bound_arguments: dict, identifier: str) -> list[str]:
        prefix_path = template.format(
            **{
                key: value_to_path_fragment(value)
                for key, value in bound_arguments.items()
            },
            identifier=escape_path_part(identifier),
        )
        parts = prefix_path.split("/")
        unquoted_parts = [urllib.parse.unquote(part) for part in parts]
        return [
            unquoted_part for unquoted_part in unquoted_parts
        ]

    return params_to_parts


def cache(
    f=None,
    *,
    path_schema: typing.Callable[[dict, dict], list[str]],
    root: str = None,
    force_format: typing.Literal["json", "pkl", "pt"] | None = None,
):
    """Cache the result of a function call.

    Example:
        @cache(path_schema=template_schema("{arg1}-{arg2}/{identifier}"))
        def my_func(arg1, arg2):
            return arg1 + arg2

        my_func automatically caches the result of the function call based on the arguments and identifier and loads from cache when possible.

        my_func.get_prefix_path(..., __timestmap=...) with the same args and kwargs returns the given prefix path (and __timestamp specifies the timestamp to use).
        my_func.load(..., __timestamp=...) tries to load from the cache or fail
        my_func.recompute(...) computes and writes to the cache regardless of prior results

        my_func(..., __force_refresh=True) is the same as .recompute but this makes it easy to force a refresh from the commandline when using e.g. Typer.
    """
    if f is None:
        return functools.partial(
            cache, path_schema=path_schema, root=root, force_format=force_format
        )

    if root is None:
        root = os.path.join(blackhc.project.project_dir, "cache")

    sig = inspect.signature(f)

    def _get_cache_path(args, kwargs, timestamp: Timestamp | str | datetime):
        # Apply defaults
        bound_args = sig.bind(*args, **kwargs)
        bound_args.apply_defaults()

        parts = path_schema(bound_args.arguments, get_callable_full_name(f))
        return get_prefix_path(*parts, root=root, timestamp=timestamp)

    @functools.wraps(f)
    def f_get_prefix_path(
        *args, timestamp: Timestamp | str | datetime = Timestamp.NONE, **kwargs
    ):
        return _get_cache_path(args, kwargs, timestamp)

    @functools.wraps(f)
    def f_load(
        *args, __timestamp: Timestamp | str | datetime = Timestamp.LATEST, **kwargs
    ):
        assert __timestamp != Timestamp.NOW
        cache_path = _get_cache_path(args, kwargs, timestamp=__timestamp)
        result = load(root=cache_path)
        print(f"📦 Loaded from cache in {cache_path}")
        return result

    @functools.wraps(f)
    def f_recompute(*args, **kwargs):
        cache_path = _get_cache_path(args, kwargs, timestamp=Timestamp.NOW)
        result = f(*args, **kwargs)

        match force_format:
            case "json":
                save_fn = save_json
            case "pkl":
                save_fn = save_pkl
            case "pt":
                save_fn = save_pt
            case None:
                if torch is not None and isinstance(result, torch.Tensor):
                    save_fn = save_pt
                else:
                    save_fn = save_pkl_or_json
            case _:
                raise ValueError(f"Unsupported force_format: {force_format}")

        cache_path = save_fn(result, root=cache_path)
        print(f"📦 Cached result in {cache_path}")
        return result

    @functools.wraps(f)
    def f_wrapper(
        *args,
        __force_refresh: bool = False,
        **kwargs,
    ):
        if not __force_refresh:
            try:
                result = f_load(*args, **kwargs, __timestamp=Timestamp.LATEST)
                return result
            except FileNotFoundError:
                pass

        return f_recompute(*args, **kwargs)

    new_wrapper_sig = sig.replace(
        parameters=[
            *sig.parameters.values(),
            inspect.Parameter(
                name="__force_refresh",
                kind=inspect.Parameter.KEYWORD_ONLY,
                default=False,
            ),
        ]
    )
    f_wrapper.__signature__ = new_wrapper_sig

    new_load_sig = sig.replace(
        parameters=[
            *sig.parameters.values(),
            inspect.Parameter(
                name="__timestamp",
                kind=inspect.Parameter.KEYWORD_ONLY,
                default=Timestamp.LATEST,
            ),
        ]
    )
    f_load.__signature__ = new_load_sig

    f_wrapper.get_prefix_path = f_get_prefix_path
    f_wrapper.load = f_load
    f_wrapper.recompute = f_recompute
    return f_wrapper
