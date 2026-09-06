from bist_hunter.intraday import MonitorInput, decide


def test_intraday_state_machine():
    assert decide(MonitorInput(90, 90, 0.03)) == "HOLD"
    assert decide(MonitorInput(90, 50, 0.01)) == "EXIT"
    assert decide(MonitorInput(90, 80, -0.04)) == "EXIT"
    assert decide(MonitorInput(90, 70, 0.01)) == "CANCEL"
