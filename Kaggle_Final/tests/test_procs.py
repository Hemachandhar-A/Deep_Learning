"""procs.py (Phase 2.1, item R2): exit codes come back paired with their labels in job order, and a failure raises.
Any command line can be a job, so no GPU and no training is involved."""
import sys

import pytest

from filament import procs


def _py(code):
    return [sys.executable, "-c", code]


def test_results_are_paired_with_labels_in_job_order_even_when_processes_finish_out_of_order():
    jobs = [("slow ok", _py("import time; time.sleep(1.2)")),
            ("fast fail", _py("import sys; sys.exit(3)")),
            ("fast ok", _py("pass")),
            ("fails late", _py("import time, sys; time.sleep(0.4); sys.exit(7)"))]
    logged = []
    res = procs.run_processes(jobs, n_gpu=2, log=logged.append, poll_seconds=0.05)
    assert res == [("slow ok", 0), ("fast fail", 3), ("fast ok", 0), ("fails late", 7)]
    gpu_lines = [l for l in logged if l.startswith("GPU ")]
    assert len(gpu_lines) == 4 and all(l.startswith(("GPU 0:", "GPU 1:")) for l in gpu_lines)
    assert any("[fast fail]" in l for l in gpu_lines)  # the mapping is printed with the label


def test_at_most_one_process_per_gpu_and_each_gets_its_own_visible_device(tmp_path):
    out = tmp_path / "devices.txt"
    code = ("import os, time\n"
            "t = time.time()\n"
            "open(%r, 'a').write(os.environ['CUDA_VISIBLE_DEVICES'] + ' %%.2f %%.2f' %% (t, t + 0.5) + chr(10))\n"
            "time.sleep(0.5)\n") % str(out)
    res = procs.run_processes([("j%d" % i, _py(code)) for i in range(4)], n_gpu=2, log=None, poll_seconds=0.05)
    assert [c for _, c in res] == [0, 0, 0, 0]
    rows = [l.split() for l in out.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 4 and sorted({r[0] for r in rows}) == ["0", "1"]
    for dev in ("0", "1"):  # on one device the runs never overlapped
        spans = sorted((float(r[1]), float(r[2])) for r in rows if r[0] == dev)
        assert len(spans) == 2 and spans[0][1] <= spans[1][0] + 0.25


def test_a_command_that_cannot_start_is_reported_not_raised():
    res = procs.run_processes([("missing", ["/no/such/program-xyz"]), ("ok", _py("pass"))], n_gpu=1, log=None,
                              poll_seconds=0.05)
    assert res == [("missing", 127), ("ok", 0)]


def test_raise_on_failure_names_every_failed_label_and_code():
    procs.raise_on_failure([("e1 fold 0", 0), ("e1 fold 1", 0)], "e1")  # nothing to raise
    with pytest.raises(procs.ProcessFailures) as e:
        procs.raise_on_failure([("e1 fold 0", 0), ("e1 fold 1", 1), ("e1 fold 2", 137)], "e1")
    msg = str(e.value)
    assert "2 of 3 processes failed" in msg and "e1 fold 1 (exit code 1)" in msg and "e1 fold 2 (exit code 137)" in msg
    assert "e1 fold 0" not in msg
    assert procs.failed([("a", 0), ("b", 2)]) == [("b", 2)]
    procs.raise_on_failure([], "nothing")  # an empty session is not a failure
    with pytest.raises(procs.ProcessFailures, match="3 of 3"):  # a session of which every process crashed is
        procs.raise_on_failure([("a", 1), ("b", 1), ("c", 1)], "all")


def test_children_get_unbuffered_output_and_their_own_device(tmp_path):
    """Phase 2.2: the drill of 2026-10-16 showed the children's text in the Kaggle log only when they exited."""
    out = tmp_path / "env.txt"
    code = ("import os; open(%r, 'w').write(os.environ.get('PYTHONUNBUFFERED', 'unset') + ' ' + "
            "os.environ['CUDA_VISIBLE_DEVICES'])") % str(out)
    res = procs.run_processes([("child", _py(code))], n_gpu=1, log=None)
    assert res == [("child", 0)] and out.read_text() == "1 0"


def test_raise_if_any_names_failed_processes_and_later_problems():
    procs.raise_if_any([("a", 0)], [], "ok")  # nothing wrong: nothing raised
    with pytest.raises(procs.ProcessFailures) as e:
        procs.raise_if_any([("a", 0), ("b", 3)], [], "s")
    assert "1 of 2 processes failed: b (exit code 3)" in str(e.value) and "problem" not in str(e.value)
    with pytest.raises(procs.ProcessFailures) as e:
        procs.raise_if_any([("a", 0)], ["scoring of R failed", "inference after training failed"], "s")
    msg = str(e.value)
    assert "2 problem(s)" in msg and "scoring of R failed; inference after training failed" in msg and "processes failed" not in msg
    with pytest.raises(procs.ProcessFailures) as e:
        procs.raise_if_any([("a", 1)], ["manifest of R failed"], "s")
    assert "processes failed" in str(e.value) and "manifest of R failed" in str(e.value) and " | " in str(e.value)
