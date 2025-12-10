import jax

from photon_weave.photon_weave import Config, Session


def test_session_sets_and_restores_seed_and_flags():
    cfg = Config()
    before_seed = cfg.random_seed
    before_contractions = cfg.contractions
    before_dynamic = cfg.dynamic_dimensions
    before_use_jit = cfg.use_jit

    with Session(
        seed=0, contractions=False, dynamic_dimensions=True, use_jit=True
    ) as c:
        assert c.random_seed == 0
        assert c.contractions is False
        assert c.dynamic_dimensions is True
        assert c.use_jit is True
        key1 = c.random_key
        key2 = c.random_key
        assert not jax.numpy.array_equal(key1, key2)

    # Session should restore prior values
    assert cfg.random_seed == before_seed
    assert cfg.contractions == before_contractions
    assert cfg.dynamic_dimensions == before_dynamic
    assert cfg.use_jit == before_use_jit


def test_session_can_override_subset_and_restore():
    cfg = Config()
    cfg.set_contraction(True)
    cfg.set_dynamic_dimensions(False)
    cfg.set_use_jit(False)

    with Session(contractions=False) as c:
        assert c.contractions is False
        # unspecified flags remain unchanged
        assert c.dynamic_dimensions is False
        assert c.use_jit is False

    assert cfg.contractions is True
    assert cfg.dynamic_dimensions is False
    assert cfg.use_jit is False


def test_nested_sessions_restore_state():
    cfg = Config()
    cfg.set_contraction(True)
    cfg.set_dynamic_dimensions(False)
    cfg.set_use_jit(False)
    cfg.set_seed(5)

    with Session(contractions=False, seed=1) as s1:
        assert s1.contractions is False
        assert s1.random_seed == 1
        with Session(dynamic_dimensions=True, use_jit=True, seed=2) as s2:
            assert s2.dynamic_dimensions is True
            assert s2.use_jit is True
            assert s2.random_seed == 2
        # after inner session, outer session settings remain
        assert s1.contractions is False
        assert s1.dynamic_dimensions is False
        assert s1.use_jit is False
        assert s1.random_seed == 1

    # after both sessions, original values restored
    assert cfg.contractions is True
    assert cfg.dynamic_dimensions is False
    assert cfg.use_jit is False
    assert cfg.random_seed == 5
