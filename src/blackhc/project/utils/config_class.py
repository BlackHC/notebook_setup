"""
config_class.py: Enhanced Configuration Classes for Python

This module provides a `configclass` decorator that extends Python's dataclasses
with additional functionality for creating flexible and powerful configuration classes.

Key features:
- Easy creation of configuration classes with default values
- Dict-like access and iteration over fields
- Merging of configuration classes using the | operator
- Automatic initialization from keyword arguments
- Support for default values, default factories, and required fields
- Invocation of functions with config class fields

The `configclass` decorator wraps around the standard dataclass decorator,
adding custom methods for initialization, iteration, and dict-like operations.

Usage:
```
    from blackhc.project.utils.config_class import configclass
    import dataclasses

    @configclass
    class MyConfig:
        required_field: int
        string_field: str = "default"
        list_field: list = dataclasses.field(default_factory=list)
        class_default: float = 3.14

    # Basic usage
    config = MyConfig(required_field=1)
    print(config)  # MyConfig(required_field=1, string_field='default', list_field=[], class_default=3.14)

    # Dict-like access
    print(config['required_field'])  # 1
    print(config.get('string_field'))  # 'default'

    # Iteration
    for value in config:
        print(value)  # Prints: 1, 'default', [], 3.14

    # Dict conversion
    config_dict = dict(config)
    print(config_dict)  # {'required_field': 1, 'string_field': 'default', 'list_field': [], 'class_default': 3.14}

    # Merging configs
    other_config = MyConfig(required_field=2, string_field="new")
    merged_config = config | other_config
    print(merged_config)  # MyConfig(required_field=2, string_field='new', list_field=[], class_default=3.14)

    # Invoking functions with config
    def print_config(required_field, string_field, list_field, class_default):
        print(f"Fields: {required_field}, {string_field}, {list_field}, {class_default}")

    config.invoke(print_config)  # Prints: Fields: 1, default, [], 3.14

    # Changing class-level defaults
    MyConfig.class_default = 2.718
    new_config = MyConfig(required_field=3)
    print(new_config.class_default)  # 2.718
```

For more details, see the docstring of the `configclass` decorator.
"""
import dataclasses
import functools

__all__ = ["configclass"]

import inspect


def __init_config_class(self, /, **kwargs):
    """
    Initialize the config class.

    Either use the keyword arguments to set the fields, or use the default values.

    Args:
        self: The config class.
        **kwargs: The keyword arguments to set the fields.

    Returns:
        None
    """
    fields = dataclasses.fields(self)
    # Check kwargs only contains fields
    for key in kwargs:
        if key not in [field.name for field in fields]:
            raise ValueError(f"Unknown field {key} for {self.__class__}")

    for field in fields:
        if field.name not in kwargs:
            # Get the class attribute (if it exists).
            class_default = getattr(self.__class__, field.name, None)
            if class_default is not None:
                # Check if it is a Field (factory) or a value.
                if isinstance(class_default, dataclasses.Field):
                    # Check if the field has a default factory.
                    if class_default.default_factory is not dataclasses.MISSING:
                        # Call the default factory.
                        kwargs[field.name] = class_default.default_factory()
                    # Check if the field has a default value.
                    elif class_default.default is not dataclasses.MISSING:
                        # Use the default value.
                        kwargs[field.name] = class_default.default
                    else:
                        raise ValueError(
                            f"Field {field.name} does not have a default value for {self.__class__}"
                        )
                else:
                    # Use the class attribute.
                    kwargs[field.name] = class_default
            else:
                raise ValueError(
                    f"Field {field.name} for {self.__class__} has no default value"
                )

    self.__dict__.update(kwargs)


# Custom __iter__ to allow for unpacking.
def __iter_config_class(self):
    """Iterate over the fields (their values) of the config class.

    The field values are returned in the order they are defined in the class.

    Args:
        self:

    Returns:
        An iterator over the fields of the config class.
    """
    for field in dataclasses.fields(self):
        yield getattr(self, field.name)


# Custom mapping methods to allow for dict-like access:
# __getitem__, __len__, __contains__, keys, items, values, get
def __getitem_config_class(self, key):
    """Get the value of a field in the config class."""
    return getattr(self, key)


def __len_config_class(self):
    """Get the number of fields in the config class."""
    return len(dataclasses.fields(self))


def __contains_config_class(self, key):
    """Check if the config class contains a field."""
    return hasattr(self, key)


def __keys_config_class(self):
    """Get the keys of the config class."""
    return [field.name for field in dataclasses.fields(self)]


def __items_config_class(self):
    """Get the items of the config class."""
    return [
        (field.name, getattr(self, field.name)) for field in dataclasses.fields(self)
    ]


def __values_config_class(self):
    """Get the values of the config class."""
    return [getattr(self, field.name) for field in dataclasses.fields(self)]


# Add | operator to allow for merging config classes (strict).
def __or_config_class(self, other):
    """Merge two config classes."""
    # Verify that self can contain all fields from `other`.
    other_dict = {**other}
    # Check that all fields in other are in self.
    missing_fields = set(other_dict.keys()) - set(self.keys())
    if missing_fields:
        raise ValueError(
            f"Cannot merge {other} into {self} because {type(other)} contains fields that are not in {type(self)}: {missing_fields}"
        )

    merged_fields = {**self, **other}
    return self.__class__(**merged_fields)


def __invoke_config_class(self, func):
    """Invoke a callable with config class's fields."""
    # Assert that func is callable.
    if not callable(func):
        raise ValueError(f"{func} is not callable")
    # Filter the fields of the config class using the callable's signature.
    signature = inspect.signature(func)
    # Get the parameters of the callable.
    parameters = signature.parameters
    # Get the fields of the config class.
    kwargs = {**self}
    # Filter the fields of the config class using the parameters of the callable.
    # (If there are variadic keyword arguments, then do not filter.)
    if all(
        parameter.kind != inspect.Parameter.VAR_KEYWORD
        for parameter in parameters.values()
    ):
        kwargs = {
            key: kwargs[key]
            for key in parameters
            if key in kwargs and parameters[key].kind != inspect.Parameter.VAR_KEYWORD
        }
    # Invoke the callable with the filtered fields.
    return func(**kwargs)


# Create a configclass decorator that wraps dataclass
@functools.wraps(dataclasses.dataclass)
def configclass(cls, /, **kwargs):
    """Create a config class from a dataclass.

    Args:
        cls:
        **kwargs:

    Returns:

    """
    # Assert that there is no init keyword argument.
    if "init" in kwargs:
        raise ValueError("Cannot specify init keyword argument for configclass")
    # Assert no eq keyword argument.
    if "eq" in kwargs:
        raise ValueError("Cannot specify eq keyword argument for configclass")
    # Assert that there is no __init__ method.
    custom_methods = {
        "__init__": __init_config_class,
        "__iter__": __iter_config_class,
        "__getitem__": __getitem_config_class,
        "__len__": __len_config_class,
        "__contains__": __contains_config_class,
        "keys": __keys_config_class,
        "items": __items_config_class,
        "values": __values_config_class,
        "invoke": __invoke_config_class,
        "__or__": __or_config_class,
    }
    # Check if the class has any of the custom methods.
    for method_name, method in custom_methods.items():
        if method_name in cls.__dict__:
            raise ValueError(
                f"Cannot specify {method_name} method for configclass ({cls} already has this method)!"
            )
    # Set the custom methods.
    for method_name, method in custom_methods.items():
        setattr(cls, method_name, method)
    # Create the dataclass with the class and the arguments.
    return dataclasses.dataclass(init=False, eq=True, **kwargs)(cls)


configclass.__doc__ = """Returns the same class as was passed in, with dunder methods
added based on the fields defined in the class. This is a wrapper around dataclass.

It uses the class variables to set the default values for the fields---so changing
class variables will change the (future) default values for the fields.

Further, on instantiation of a configclass instance, the fields can be set using
keyword arguments. If a field is not set, the default value is used.

Two configclasses can be merged using the | operator. The fields of the right-hand
side configclass are used to overwrite the fields of the left-hand side configclass.

configclasses can be iterated over, and they can be accessed and unpacked like a dict.

It supports fields with default values, fields with default factories, and fields
with no default value.

It adds the following methods to the class:
    __init__(self, /, **kwargs)
    __iter__(self)
    __getitem__(self, key)
    __len__(self)
    __contains__(self, key)
    keys(self)
    items(self)
    values(self)
    __or__(self, other)

It also always adds __eq__, and __ne__ methods to the class via dataclass.

Examines PEP 526 __annotations__ to determine fields.

If repr is true, a __repr__() method is added. If order is true, rich
comparison dunder methods are added. If unsafe_hash is true, a
__hash__() method function is added. If frozen is true, fields may
not be assigned to after instance creation. If match_args is true,
the __match_args__ tuple is added. If kw_only is true, then by
default all fields are keyword-only. If slots is true, an
__slots__ attribute is added.
"""


# Test the configclass decorator.
if __name__ == "__main__":
    # Create a config class.
    @configclass
    class Config:
        a: int
        b: int
        c: int = 3

    # Create an instance of the config class.
    config = Config(a=1, b=2)
    # Print the config class.
    print(config)
    # Print the config class as a dict.
    print(dict(config))
    # Print the config class as a list.
    print(list(config))
    # Print the config class as a tuple.
    print(tuple(config))
    # Print the config class as a set.
    print(set(config))
    # Print the config class as a list of keys.
    print(list(config.keys()))
    # Print the config class as a list of values.
    print(list(config.values()))
    # Print the config class as a list of items.
    print(list(config.items()))

    # Change the default value of the config class.
    Config.c = 4
    # Create an instance of the config class.
    config = Config(a=1, b=2)
    # Print the config class.
    print(config)
