from pytest import fixture

from rl_replicas.utils import set_seed_for_libraries


@fixture
def seed() -> int:
    seed: int = 0

    return seed


@fixture(autouse=True)
def set_seed(seed: int) -> int:
    set_seed_for_libraries(seed)

    return seed
