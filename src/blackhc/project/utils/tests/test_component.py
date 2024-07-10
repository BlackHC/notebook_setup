"""
Component tests.
"""
import typing
from dataclasses import dataclass
from blackhc.project.utils.component import Interface, Component, ComponentView


T = typing.TypeVar("T")


class InterfaceA(Interface):
    def a(self):
        raise NotImplementedError


class InterfaceB(Interface):
    def b(self):
        raise NotImplementedError


@typing.runtime_checkable
class ProtocolC(typing.Protocol):
    def c(self):
        raise NotImplementedError


class ImplementationAB(InterfaceA, InterfaceB):
    def a(self):
        return 0

    def b(self):
        return 1

    def c(self):
        return 2


def test_interface_ab():
    implementation_ab = ImplementationAB()
    assert implementation_ab.a() == 0
    assert implementation_ab.b() == 1

    assert isinstance(implementation_ab, ImplementationAB)
    assert isinstance(implementation_ab, InterfaceA)
    assert isinstance(implementation_ab, InterfaceB)
    assert isinstance(implementation_ab, Interface)

    assert InterfaceA.cast(implementation_ab) is implementation_ab
    assert InterfaceB.cast(implementation_ab) is implementation_ab
    assert isinstance(implementation_ab, ProtocolC)

    assert ImplementationAB.cast(implementation_ab) is implementation_ab


class ImplementationA(InterfaceA):
    def a(self):
        return 0


def test_interface_a():
    implementation_a = ImplementationA()

    assert not isinstance(implementation_a, InterfaceB)


@dataclass
class ImplementationB(InterfaceB, Component):
    field: int

    def b(self):
        return self.field

    def query_protocol(self, cls: typing.Type[T]) -> T:
        if issubclass(InterfaceA, cls):
            return ImplementationB_ViewA(self)
        return super().query_protocol(cls)


class ImplementationB_ViewA(InterfaceA, ComponentView[ImplementationB]):
    def a(self):
        return self._component.field


def test_component():
    implementation_b = ImplementationB(1)
    interface_a = InterfaceA.cast(implementation_b)
    assert interface_a.a() == 1
    interface_b = InterfaceB.cast(interface_a)
    assert interface_b.b() == 1
    assert isinstance(interface_b, InterfaceB)
