import importlib.util
from pathlib import Path


def runner():
    file = Path(__file__).resolve().parents[1] / "scripts/parallel_assin_tail.py"
    spec = importlib.util.spec_from_file_location("parallel_tail", file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_readiness_requires_fourth_complete_and_fifth_holding_lock(tmp_path):
    module = runner()

    output = tmp_path
    fourth = output / "candidates/binary-hard-ranking/run/training.json"
    fifth_lock = output / "locks/assin-three-way.lock"
    fourth.parent.mkdir(parents=True)
    assert not module.ready(output)
    fourth.write_text("{}")
    assert not module.ready(output)
    fifth_lock.parent.mkdir()
    import fcntl

    with fifth_lock.open("a") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        assert module.ready(output)
    assert not module.ready(output)
    sixth = output / "candidates/binary-seed42/run/training.json"
    sixth.parent.mkdir(parents=True)
    sixth.write_text("{}")
    with fifth_lock.open("a") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        assert not module.ready(output)


def test_tail_config_uses_frozen_grid_and_incumbent_validation_f1(tmp_path):
    module = runner()
    protocol = dict(
        initial={"directory": "/frozen/initial"},
        candidates=[
            dict(name="other"),
            dict(
                name="binary-seed42", hard=True, training={"seed": 42, "task": "equivalence_binary"}
            ),
        ],
    )
    config = module.tail_config(protocol, {"incumbent_validation": {"f1": 0.78}})
    assert config["initial"] == "/frozen/initial"
    assert config["training"] == {"seed": 42, "task": "equivalence_binary", "minimum_f1": 0.78}
    assert protocol["candidates"][-1]["training"] == {"seed": 42, "task": "equivalence_binary"}
