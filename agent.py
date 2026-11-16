"""The experiment loop.

    propose -> static check -> run in a subprocess -> physics check -> keep or revert -> log

Usage:
    python agent.py --task heat_exchanger --iters 30                 # LLM proposer
    python agent.py --task pipe_flow --iters 4 --proposer scripted   # offline test of the loop

Outputs, per task, under runs/<task>/:
    best.py                 the current best model (starts as baselines/generic.py)
    NNN/model.py, NNN/record.json   every attempt, kept or not
    results.tsv             one line per attempt
    REPORT.md               summary table, including the held-out extrapolation error
"""
import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import guardrails
import tasks
from proposers import llm_proposer, scripted_proposer

ROOT = Path(__file__).resolve().parent
TIMEOUT_S = 300
MIN_REL_GAIN = 0.02   # a change must cut relative RMSE by at least 2% of its value


def run_harness(task_name, path):
    """Score a model in a separate process. Returns (metrics, error)."""
    try:
        out = subprocess.run([sys.executable, str(ROOT / "harness.py"), "--task", task_name, str(path)],
                             capture_output=True, text=True, timeout=TIMEOUT_S, cwd=ROOT)
    except subprocess.TimeoutExpired:
        return None, f"timed out after {TIMEOUT_S}s"
    if out.returncode != 0:
        tail = out.stderr.strip().splitlines()
        return None, tail[-1] if tail else "non-zero exit"
    return json.loads(out.stdout.strip().splitlines()[-1]), None


def build_brief(task):
    inputs = "\n".join(f"- `{k}`: {v}" for k, v in task.INPUTS.items())
    constraints = "\n".join(f"   - **{c['name']}**: {c['desc']}" for c in task.CONSTRAINTS)
    return (ROOT / "program.md").read_text().format(
        title=task.TITLE, brief=task.BRIEF, inputs=inputs, target=task.TARGET,
        constraints=constraints, noise_pct=round(task.NOISE * 100),
        min_gain_pct=round(MIN_REL_GAIN * 100), timeout=TIMEOUT_S)


class Log:
    def __init__(self, run_root):
        self.root, self.history = run_root, []

    def add(self, record, run_dir=None):
        self.history.append(record)
        if run_dir:
            (run_dir / "record.json").write_text(json.dumps(record, indent=2))
        tsv = self.root / "results.tsv"
        new = not tsv.exists()
        with open(tsv, "a") as f:
            if new:
                f.write("id\tstatus\tcv_rel_rmse\textrap_rel_rmse\thypothesis\treason\n")
            f.write(f"{record['id']}\t{record['status']}\t{record.get('cv_rel_rmse', '')}\t"
                    f"{record.get('extrap_rel_rmse', '')}\t{record['hypothesis']}\t{record.get('reason', '')}\n")
        print(f"[{record['id']}] {record['status']:<17} cv={record.get('cv_rel_rmse', '-')}  "
              f"{record['hypothesis']}" + (f"\n      -> {record['reason']}" if record.get("reason") else ""))

    def report(self, task, best):
        counts = {}
        for h in self.history[1:]:
            counts[h["status"]] = counts.get(h["status"], 0) + 1
        base = self.history[0]
        lines = [f"# Run report: {task.TITLE}", "",
                 f"- Baseline: CV relative RMSE {base.get('cv_rel_rmse')}, physics check "
                 f"{'passed' if base['status'] == 'baseline_pass' else 'failed'}",
                 f"- Best passing model: {best if best is not None else 'none found'}",
                 f"- Measurement noise floor: about {task.NOISE}",
                 f"- Experiments: {len(self.history) - 1} "
                 f"({', '.join(f'{v} {k}' for k, v in sorted(counts.items()))})",
                 "", "Extrapolation error is measured on held-out inputs outside the training "
                 "range and is never shown to the agent.", "",
                 "| id | status | CV rel. RMSE | extrap. rel. RMSE | hypothesis | reason |",
                 "|---|---|---|---|---|---|"]
        for h in self.history:
            lines.append(f"| {h['id']} | {h['status']} | {h.get('cv_rel_rmse', '')} | "
                         f"{h.get('extrap_rel_rmse', '')} | {h['hypothesis']} | {h.get('reason', '')} |")
        (self.root / "REPORT.md").write_text("\n".join(lines) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=sorted(tasks.TASKS))
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--proposer", choices=["llm", "scripted"], default="llm")
    ap.add_argument("--fresh", action="store_true", help="discard earlier runs for this task")
    args = ap.parse_args()
    task = tasks.get(args.task)
    propose = llm_proposer if args.proposer == "llm" else scripted_proposer

    run_root = ROOT / "runs" / task.NAME
    if args.fresh and run_root.exists():
        shutil.rmtree(run_root)
    run_root.mkdir(parents=True, exist_ok=True)
    best_path = run_root / "best.py"
    if not best_path.exists():
        shutil.copy(ROOT / "baselines" / "generic.py", best_path)
    brief = build_brief(task)
    (run_root / "brief.md").write_text(brief)

    log = Log(run_root)
    metrics, err = run_harness(task.NAME, best_path)
    if err:
        sys.exit(f"starting model does not run: {err}")
    failed = guardrails.physics_check(metrics)
    log.add({"id": "000", "hypothesis": "starting model", **metrics,
             "status": "baseline_fail" if failed else "baseline_pass",
             "reason": "; ".join(failed)})
    best = None if failed else metrics["cv_rel_rmse"]

    for step in range(args.iters):
        rid = f"{len(log.history):03d}"
        run_dir = run_root / rid
        run_dir.mkdir(exist_ok=True)
        record = {"id": rid, "hypothesis": "(no proposal)"}

        try:
            hypothesis, code = propose(brief, best_path.read_text(), log.history, step)
        except Exception as e:
            record.update(status="proposal_failed", reason=str(e)[:200])
            log.add(record, run_dir)
            continue
        record["hypothesis"] = hypothesis
        model_path = run_dir / "model.py"
        model_path.write_text(code)

        problems = guardrails.static_check(code)
        if problems:
            record.update(status="rejected_static", reason="; ".join(problems))
            log.add(record, run_dir)
            continue

        metrics, err = run_harness(task.NAME, model_path)
        if err:
            record.update(status="crashed", reason=err[:200])
            log.add(record, run_dir)
            continue
        record.update(metrics)

        problems = guardrails.physics_check(metrics)
        if problems:
            record.update(status="rejected_physics", reason="; ".join(problems))
        elif best is None or metrics["cv_rel_rmse"] < best * (1 - MIN_REL_GAIN):
            record.update(status="kept")
            best = metrics["cv_rel_rmse"]
            shutil.copy(model_path, best_path)
        else:
            record.update(status="discarded", reason=f"no gain over {best}")
        log.add(record, run_dir)

    log.report(task, best)
    print(f"done. best passing cv_rel_rmse={best}. See {run_root.relative_to(ROOT)}/REPORT.md")


if __name__ == "__main__":
    main()
