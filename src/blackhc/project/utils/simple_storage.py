"""
Simple file-based object storage.
"""
import functools
import inspect
import json
import os
import pickle
import sys
import typing
import urllib.parse

from datetime import datetime

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
    # return f"{get_module_name(f)}:{f.__qualname__}"
    return f.__qualname__


def escape_path_part(part: str):
    return urllib.parse.quote(part, safe=" ")


def arg_to_path_fragment(value: float | str | int | list[float | int | str]) -> str:
    """Convert a value to a path part."""
    if isinstance(value, float) and value.is_integer():
        value = int(value)
    if isinstance(value, float):
        value = format(
            value, ".6g"
        )  # Use general format with up to 6 significant digits
    elif isinstance(value, list):
        value = "[" + ",".join(map(arg_to_path_fragment, value)) + "]"
    elif isinstance(value, tuple):
        value = "(" + ",".join(map(arg_to_path_fragment, value)) + ")"
    elif isinstance(value, dict):
        value = (
            "{"
            + ",".join(
                f"{arg_to_path_fragment(k)}:{arg_to_path_fragment(v)}"
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


def kwargs_to_fragment(kwargs: dict, incl_keys: bool = True):
    kwarg_fragments = []
    for key in sorted(kwargs.keys()):
        if incl_keys:
            kwarg_fragments.append(f"{key}:{arg_to_path_fragment(kwargs[key])}")
        else:
            kwarg_fragments.append(f"{arg_to_path_fragment(kwargs[key])}")
    return "_".join(kwarg_fragments)


def generate_path(*parts) -> str:
    """
    Generates a path based on the given identifier and keyword arguments.

    Args:
        identifier (str or callable): The identifier to be used in the path.
        **kwargs: Additional keyword arguments to be included in the path.

    Returns:
        str: The generated path.
    """
    path_parts = []
    for part in parts:
        if part is None:
            fragment = ''
        elif isinstance(part, dict):
            fragment = kwargs_to_fragment(part)
        else:
            fragment = arg_to_path_fragment(part)
        if fragment != '':
            path_parts.append(fragment)
            
    if parts and parts[-1] is None:
        path_parts.append('')
        
    return "/".join(path_parts)


def collect_metadata(*parts) -> dict[str]:
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


def get_prefix_path(*parts, root: str = "") -> str:
    return os.path.join(root, generate_path(*parts))


def _combine_path(prefix_path, ext) -> str:
    if prefix_path.endswith("/"):
        return f"{prefix_path}{ext}"
    return f"{prefix_path}.{ext}"


def _save_metadata(*parts, root: str = "") -> str:
    prefix_path = get_prefix_path(*parts, root=root)
    metadata = collect_metadata(*parts)
    os.makedirs(os.path.dirname(prefix_path), exist_ok=True)
    with open(_combine_path(prefix_path, "meta.json"), "wt", encoding="utf-8") as f:
        json.dump(metadata, f)
    return prefix_path


def load_metadata(*parts, root: str = ""):
    prefix_path = get_prefix_path(*parts, root=root)
    with open(_combine_path(prefix_path, "meta.json"), "rt", encoding="utf-8") as f:
        return json.load(f)


def save_pkl(obj, *parts, root: str = "") -> str:
    prefix_path = _save_metadata(*parts, root=root)
    with open(_combine_path(prefix_path, "data.pkl"), "wb") as f:
        pickle.dump(obj, f)
    return prefix_path


def load_pkl(*parts, root: str = ""):
    prefix_path = get_prefix_path(*parts, root=root)
    with open(_combine_path(prefix_path, "data.pkl"), "rb") as f:
        return pickle.load(f)


def save_pt(obj, *parts, root: str = "") -> str:
    prefix_path = _save_metadata(*parts, root=root)
    with open(_combine_path(prefix_path, "data.pt"), "wb") as f:
        torch.save(obj, f)
    return prefix_path


def load_pt(*parts, root: str = ""):
    prefix_path = get_prefix_path(*parts, root=root)
    with open(_combine_path(prefix_path, "data.pt"), "rb") as f:
        return torch.load(f)


def save_json(obj, *parts, root: str = "") -> str:
    prefix_path = _save_metadata(*parts, root=root)
    with open(_combine_path(prefix_path, "data.json"), "wt", encoding="utf-8") as f:
        json.dump(obj, f)
    return prefix_path


def load_json(*parts, root: str = ""):
    prefix_path = get_prefix_path(*parts, root=root)
    with open(_combine_path(prefix_path, "data.json"), "rt", encoding="utf-8") as f:
        return json.load(f)
    
    
def save_pkl_or_json(obj, *parts, root: str = "") -> str:
    prefix_path = _save_metadata(*parts, root=root)
    
    # Pickle the object into bytes
    pickled_obj = pickle.dumps(obj)
    
    # Check if the size is less than 256KB
    if len(pickled_obj) < 256 * 1024:
        try:
            # Try to save as JSON
            json_obj = json.loads(json.dumps(obj))
            assert json_obj == obj
            with open(_combine_path(prefix_path, "data.json"), "wt", encoding="utf-8") as f:
                json.dump(obj, f)
            return prefix_path
        except (TypeError, OverflowError, AssertionError):
            # If it fails, save as pickle
            pass
   
    with open(_combine_path(prefix_path, "data.pkl"), "wb") as f:
        f.write(pickled_obj)
    return prefix_path


def load(*parts, root: str = ""):
    prefix_path = get_prefix_path(*parts, root=root)
    
    # Find the *data.* file (can either end in pkl, json or ot)
    data_files = [f for f in os.listdir(os.path.dirname(prefix_path)) if f.endswith(("data.pt", "data.pkl", "data.json"))]
    
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
        raise RuntimeError("Multiple data files found for the same prefix path", data_files)
    else:
        raise FileNotFoundError("No data file found for the prefix path", prefix_path)


def cache(f=None, *, prefix_args: list[str], root:str, force_format:typing.Literal["json", "pkl", "pt"] | None = None):
    if f is None:
        return functools.partial(cache, prefix_args=prefix_args, root=root, force_format=force_format)
    
    @functools.wraps(f)
    def f_get_prefix_path(*args, **kwargs):
         # Apply defaults
        sig = inspect.signature(f)
        bound_args = sig.bind(*args, **kwargs)
        bound_args.apply_defaults()
        
        # Extract prefix args into one dictionary and all other args into another
        prefix_dict = {arg: bound_args.arguments[arg] for arg in bound_args.arguments if arg in prefix_args}
        suffix_dict = {arg: bound_args.arguments[arg] for arg in bound_args.arguments if arg not in prefix_args}
        
        # Generate the prefix path
        prefix_path = get_prefix_path(prefix_dict, get_callable_full_name(f), suffix_dict, root=root)
        return prefix_path

    @functools.wraps(f)
    def f_load(*args, **kwargs):
        # Apply defaults
        sig = inspect.signature(f)
        bound_args = sig.bind(*args, **kwargs)
        bound_args.apply_defaults()
        
        # Extract prefix args into one dictionary and all other args into another
        prefix_dict = {arg: bound_args.arguments[arg] for arg in bound_args.arguments if arg in prefix_args}
        suffix_dict = {arg: bound_args.arguments[arg] for arg in bound_args.arguments if arg not in prefix_args}
        
        # Generate the prefix path
        prefix_path = get_prefix_path(prefix_dict, get_callable_full_name(f), suffix_dict, root=root)
        
        # Find all subdirs in the prefix path
        subdirs = sorted([d for d in os.listdir(prefix_path) if os.path.isdir(os.path.join(prefix_path, d))])
        if not subdirs:
            raise FileNotFoundError("No subdirectories found in the prefix path", prefix_path)
        latest_subdir = subdirs[-1]
        latest_subdir_path = os.path.join(prefix_path, latest_subdir)
        return load(root=latest_subdir_path)
    
    @functools.wraps(f)
    def f_recompute(*args, **kwargs):
        # Apply defaults
        sig = inspect.signature(f)
        bound_args = sig.bind(*args, **kwargs)
        bound_args.apply_defaults()
        
        # Extract prefix args into one dictionary and all other args into another
        prefix_dict = {arg: bound_args.arguments[arg] for arg in bound_args.arguments if arg in prefix_args}
        suffix_dict = {arg: bound_args.arguments[arg] for arg in bound_args.arguments if arg not in prefix_args}
            
        # No result has been cached. So execute the function.
        timestamp = datetime.now().isoformat()
        result = f(*bound_args.args, **bound_args.kwargs)
        
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
            
        save_fn(result, prefix_dict, get_callable_full_name(f), suffix_dict, timestamp, None, root=root)
        
        return result
    
    @functools.wraps(f)
    def f_wrapper(*args, **kwargs):
        try:
            result = f_load(*args, **kwargs)
            return result
        except FileNotFoundError:
            pass
        
        return f_recompute(*args, **kwargs)
        
    f_wrapper.get_prefix_path = f_get_prefix_path
    f_wrapper.load = f_load
    f_wrapper.recompute = f_recompute
    return f_wrapper