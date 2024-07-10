"""
Component class to make it easy to create component-based classes.

This module provides a flexible and robust way to implement component-based
architecture in Python. It includes utilities for interface definition,
component implementation, and protocol casting.

Key features:
1. Interface and Component protocols for defining component contracts
2. Explicit casting functions for type-safe component interactions
3. ComponentView for adapting interfaces
4. ProtocolWrapper for wrapping existing objects with component functionality

See test_component.py for usage examples.
"""
import typing
from dataclasses import dataclass

T = typing.TypeVar("T")
C = typing.TypeVar("C", bound="Component")


def explicit_try_cast(cls: typing.Type[T], instance) -> T | None:
    """Try to cast an instance to a protocol.

    If the instance implements the protocol, return the instance. Otherwise, return None.
    """
    if isinstance(instance, cls):
        return instance
    elif isinstance(instance, Component):
        return instance.query_protocol(cls)
    else:
        return None


def explicit_cast(cls: typing.Type[T], instance) -> T:
    """Cast an instance to a protocol.

    If the instance implements the protocol, return the instance. Otherwise, raise a TypeError.
    """
    view = explicit_try_cast(cls, instance)
    if view is None:
        raise TypeError(f"Cannot cast {type(instance)} to {cls} for:\n{instance}")
    return view


@typing.runtime_checkable
class Interface(typing.Protocol):
    """
    Interface for a component.

    This is a protocol that can be used to check if a class implements an interface and to cast to an interface.
    """

    @classmethod
    def try_cast(cls: typing.Type[T], instance) -> T | None:
        """Try to cast an instance to the class.

        If the instance implements the class, return the instance. Otherwise, return None.
        """
        return explicit_try_cast(cls, instance)

    @classmethod
    def cast(cls: typing.Type[T], instance) -> T:
        """Cast an instance to the class.

        If the instance implements the class, return the instance. Otherwise, raise a TypeError.
        """
        return explicit_cast(cls, instance)


@typing.runtime_checkable
class Component(Interface, typing.Protocol):
    """
    Default component implementation.
    """

    def query_protocol(self, cls: typing.Type[T]) -> T | None:
        """Query the protocol for a component.

        If the component implements the protocol, return the component. Otherwise, return None.
        """
        if isinstance(self, cls):
            return self
        return None


@dataclass
class ComponentView(Component, typing.Generic[C]):
    """
    Adapter for an interface.

    When you want to implement a different interface for a component, you can use this class.

    Example
    -------

    >>> class InterfaceA(Interface):
    ...     def a(self):
    ...         raise NotImplementedError
    >>> class InterfaceB(Interface):
    ...     def a(self):
    ...         raise NotImplementedError
    >>> class ImplementationAB(InterfaceA, Component):
    ...     def a(self):
    ...         return 0
    ...     def b(self):
    ...         return 1
    ...     def query_protocol(self, cls: typing.Type[T]) -> T:
    ...         if issubclass(InterfaceB, cls):
    ...             return ImplementationAB.ViewB(self)
    ...     class ViewB(InterfaceB, ComponentView['ImplementationAB']):
    ...         def a(self):
    ...             return self._component.b()
    """

    _component: C

    def query_protocol(self, cls: typing.Type[T]) -> T | None:
        """Query the protocol for a component.

        If the component implements the protocol, return the component. Otherwise, return None.
        """
        if isinstance(self, cls):
            return self

        return self._component.query_protocol(cls)


@dataclass
class ProtocolWrapper(Component, typing.Generic[T]):
    """Adapter for a protocol.

    When you want to implement a different protocol for a component, you can use this class.
    """

    protocol_type: typing.Type[T]
    instance: T

    def query_protocol(self, cls: typing.Type[T]) -> T | None:
        if issubclass(cls, self.protocol_type):
            return self.instance
        return super().query_protocol(cls)
