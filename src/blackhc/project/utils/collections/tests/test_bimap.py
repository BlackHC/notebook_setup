from blackhc.project.utils.collections.bimap import DictBimap, PersistentBimap, MappingBimap

import pytest


@pytest.mark.parametrize("bimap_factory", [DictBimap])
def test_bimap(bimap_factory):
    bimap: MappingBimap = bimap_factory()

    assert not bimap.has_value(1)
    assert not bimap.has_value(2)
    assert not 1 in bimap
    assert not 2 in bimap

    assert bimap.length() == 0

    bimap.put_key_value(1, 2)

    assert bimap.length() == 1

    assert 1 in bimap
    assert bimap.has_value(2)

    assert 2 not in bimap
    assert not bimap.has_value(1)

    bimap.update(3, 4)

    assert bimap.length() == 2

    assert 3 in bimap
    assert bimap.has_value(4)

    assert bimap.get_value(1) == 2
    assert bimap.get_value(3) == 4

    assert bimap.get_key(2) == 1
    assert bimap.get_key(4) == 3

    # We can delete values and keys that are not contained in the bimap without exception.
    bimap.del_key(5)
    bimap.del_value(6)

    assert bimap.length() == 2

    bimap.del_key(1)

    assert bimap.length() == 1

    assert 1 not in bimap
    assert not bimap.has_value(2)

    bimap.del_value(4)

    assert bimap.length() == 0

    assert 3 not in bimap
    assert not bimap.has_value(4)

    bimap.update(1, None)
    bimap.update(None, 4)

    assert bimap.length() == 0

    bimap.update(5, 6)
    bimap.update(7, 8)

    assert bimap.length() == 2

    bimap.update(5, None)

    assert bimap.length() == 1

    bimap.update(None, 8)

    assert bimap.length() == 0

    with pytest.raises(ValueError):
        bimap.put_key_value(1, None)

    with pytest.raises(ValueError):
        bimap.put_key_value(None, 2)

    with pytest.raises(ValueError):
        bimap.put_key_value(None, None)

    with pytest.raises(ValueError):
        bimap.update(None, None)
