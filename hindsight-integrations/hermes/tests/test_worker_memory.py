"""worker_memory: how much of the shared bank a kanban worker (HERMES_KANBAN_TASK) may use."""

import pytest


def _drive(instance):
    """One worker turn through every hook that can reach Hindsight."""
    instance.queue_prefetch("what did we decide?")
    instance.prefetch("what did we decide?")
    instance.sync_turn("work kanban task t_1", "done")
    results = {
        tool: instance.handle_tool_call(tool, {"content": "a fact", "query": "a fact"})
        for tool in ("hindsight_retain", "hindsight_recall", "hindsight_reflect")
    }
    instance.shutdown()
    return results


@pytest.fixture
def kanban(monkeypatch):
    monkeypatch.setenv("HERMES_KANBAN_TASK", "t_1")


def test_full_is_the_default_and_unchanged(provider, kanban):
    worker, fake = provider({})
    results = _drive(worker)
    assert worker._worker_memory == "full"
    assert not results["hindsight_retain"].startswith("ERROR")
    assert len(fake.retains) == 2  # turn + tool
    assert fake.recalls and fake.reflects


def test_read_only_recalls_but_never_retains(provider, kanban):
    worker, fake = provider({"worker_memory": "read_only"})
    results = _drive(worker)
    assert (
        results["hindsight_retain"]
        == "ERROR: hindsight_retain is disabled for kanban workers (worker_memory=read_only)"
    )
    assert not results["hindsight_recall"].startswith("ERROR")
    assert fake.retains == []
    assert len(fake.recalls) == 2  # auto-recall + tool
    assert len(fake.reflects) == 1


def test_worker_memory_is_ignored_outside_kanban_workers(provider, monkeypatch):
    monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)
    main, fake = provider({"worker_memory": "off"})
    results = _drive(main)
    assert not results["hindsight_retain"].startswith("ERROR")
    assert len(fake.retains) == 2
    assert fake.recalls and fake.reflects


def test_off_makes_no_hindsight_call(provider, kanban):
    worker, fake = provider({"worker_memory": "off"})
    assert worker.get_tool_schemas() == []
    assert worker.system_prompt_block() == ""
    results = _drive(worker)
    assert all(result.startswith("ERROR") for result in results.values())
    assert (fake.retains, fake.recalls, fake.reflects) == ([], [], [])


def test_an_unknown_value_falls_back_to_full(provider, kanban):
    worker, _ = provider({"worker_memory": "readonly"})
    assert worker._worker_memory == "full"
    worker.shutdown()
