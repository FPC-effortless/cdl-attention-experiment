from casm.reuse_merge import ReuseSignature, compatible, propose_reuse_groups


def test_compatible_requires_entity_version_and_dependencies():
    a = ReuseSignature(1, 2, (3, 4))
    assert compatible(a, ReuseSignature(1, 2, (3, 4)))
    assert not compatible(a, ReuseSignature(1, 3, (3, 4)))
    assert not compatible(a, ReuseSignature(1, 2, (4, 3)))
    assert not compatible(a, ReuseSignature(2, 2, (3, 4)))


def test_groups_are_deterministic_and_preserve_input_indices():
    sigs = [
        ReuseSignature(1, 2, (3,)),
        ReuseSignature(1, 2, (3,)),
        ReuseSignature(1, 2, (4,)),
    ]
    assert propose_reuse_groups(sigs) == ((0, 1), (2,))
