"""Tests for ResourceManager slot system."""
import threading
import pytest
from deepsearch.core.resource_manager import ResourceManager, Slot


@pytest.fixture
def rm():
    return ResourceManager()


class TestResourceManager:
    def test_initial_state_empty(self, rm):
        for slot in Slot:
            assert not rm.is_loaded(slot)

    def test_mark_loaded_unloaded(self, rm):
        rm.mark_loaded(Slot.B, "phi3-mini")
        assert rm.is_loaded(Slot.B)
        assert rm.loaded_model(Slot.B) == "phi3-mini"
        rm.mark_unloaded(Slot.B)
        assert not rm.is_loaded(Slot.B)

    def test_slot_a_acquire(self, rm):
        unload_called = []
        rm.acquire_slot_a("llm-main", lambda: unload_called.append(True))
        assert rm.slot_a_owner == "llm-main"
        assert rm.is_loaded(Slot.A)

    def test_slot_a_eviction(self, rm):
        evicted = []
        rm.acquire_slot_a("model-a", lambda: evicted.append("a"))
        rm.acquire_slot_a("model-b", lambda: evicted.append("b"))
        assert "a" in evicted           # model-a was evicted
        assert rm.slot_a_owner == "model-b"

    def test_slot_a_no_eviction_if_same(self, rm):
        evicted = []
        rm.acquire_slot_a("same-model", lambda: evicted.append("x"))
        rm.acquire_slot_a("same-model", lambda: evicted.append("x"))
        assert evicted == []            # no eviction for same model

    def test_slot_a_release(self, rm):
        unloaded = []
        rm.acquire_slot_a("llm", lambda: unloaded.append(True))
        rm.release_slot_a("llm")
        assert rm.slot_a_owner is None
        assert unloaded == [True]

    def test_slot_a_raises_for_b_c(self, rm):
        with pytest.raises(ValueError):
            rm.mark_loaded(Slot.A, "anything")

    def test_status_dict(self, rm):
        rm.acquire_slot_a("llm", lambda: None)
        rm.mark_loaded(Slot.B, "small")
        status = rm.status()
        assert status["A"] == "llm"
        assert status["B"] == "small"
        assert status["C"] is None

    def test_thread_safety(self, rm):
        """Multiple threads acquiring Slot A should not corrupt state."""
        results = []

        def load_model(name):
            rm.acquire_slot_a(name, lambda: None)
            results.append(rm.slot_a_owner)

        threads = [threading.Thread(target=load_model, args=(f"m{i}",)) for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Final owner should be exactly one model
        assert rm.slot_a_owner is not None
        assert rm.slot_a_owner.startswith("m")
