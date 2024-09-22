from blackhc.project.utils.collections.id_set import IdSet
from blackhc.project.utils.collections.testing.test_mutable_set import MutableSetTests


class TestIdSet(MutableSetTests):
    mutable_set = IdSet

    @staticmethod
    def get_element(i):
        return [i]
