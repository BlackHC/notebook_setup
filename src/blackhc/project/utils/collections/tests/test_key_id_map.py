from blackhc.project.utils.collections.key_id_dict import KeyIdDict
from blackhc.project.utils.collections.testing.test_mutable_mapping import MutableMappingTests


class TestKeyIdDict(MutableMappingTests):
    mutable_mapping = KeyIdDict

    @staticmethod
    def get_key(i):
        return [i]
