import pytest

from harness4h3.rsi.budget import BudgetLedger


def test_budget_is_idempotent_and_failed_work_is_charged(tmp_path):
    ledger = BudgetLedger(tmp_path / "budget.json", {"api_calls": 3, "gpu_seconds": 10})
    assert ledger.reserve("job-1", {"api_calls": 1, "gpu_seconds": 4})
    assert ledger.reserve("job-1", {"api_calls": 1, "gpu_seconds": 4})
    ledger.finalize("job-1", {"api_calls": 1, "gpu_seconds": 5})
    ledger.finalize("job-1", {"api_calls": 1, "gpu_seconds": 5})
    assert ledger.snapshot()["spent_or_reserved"] == {"api_calls": 1.0, "gpu_seconds": 5.0}
    assert ledger.reserve("job-2", {"api_calls": 2, "gpu_seconds": 5})
    assert not ledger.reserve("job-3", {"api_calls": 1, "gpu_seconds": 1})


def test_budget_rejects_conflicting_reservations_and_settlements(tmp_path):
    ledger = BudgetLedger(tmp_path / "budget.json", {"calls": 2})
    assert ledger.reserve("job", {"calls": 1})
    with pytest.raises(ValueError, match="reservation_conflict"):
        ledger.reserve("job", {"calls": 2})
    ledger.finalize("job", {"calls": 1})
    with pytest.raises(ValueError, match="settlement_conflict"):
        ledger.finalize("job", {"calls": 0})
