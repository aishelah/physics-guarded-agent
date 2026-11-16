"""Where proposals come from.

A proposer takes the brief, the current best code and the experiment history,
and returns (hypothesis, new_code).

  llm_proposer      asks a language model (any OpenAI-compatible endpoint)
  scripted_proposer a fixed list of hand-written edits, used ONLY to test that
                    the loop and the guardrails behave correctly. Results from
                    it are not agent results.
"""
import json
import os
import re
import urllib.request


# ---------------------------------------------------------------- LLM
def _history_table(history, last=15):
    """What the agent is shown. Extrapolation error is deliberately left out."""
    if not history:
        return "(no experiments yet)"
    rows = ["id | status | cv_rel_rmse | hypothesis | notes"]
    for h in history[-last:]:
        rows.append(f"{h['id']} | {h['status']} | {h.get('cv_rel_rmse', '-')} | "
                    f"{h['hypothesis']} | {h.get('reason', '')}")
    return "\n".join(rows)


def parse_reply(text):
    """Pull the hypothesis line and the code block out of a model reply."""
    hyp = re.search(r"HYPOTHESIS:\s*(.+)", text)
    code = re.search(r"```(?:python)?\s*\n(.*?)```", text, re.DOTALL)
    if not hyp or not code:
        raise ValueError("reply did not contain a HYPOTHESIS line and a python code block")
    return hyp.group(1).strip(), code.group(1).strip() + "\n"


def llm_proposer(brief, code, history, step):
    """Needs LLM_API_KEY, and optionally LLM_BASE_URL and LLM_MODEL, in the environment."""
    base = os.environ.get("LLM_BASE_URL", "https://api.openai.com/v1").rstrip("/")
    body = {
        "model": os.environ.get("LLM_MODEL", "gpt-4o-mini"),
        "temperature": 0.7,
        "messages": [
            {"role": "system", "content": brief},
            {"role": "user", "content":
                f"Experiment history:\n{_history_table(history)}\n\n"
                f"Current best model:\n```python\n{code}```\n\n"
                "Propose the next experiment in the required reply format."},
        ],
    }
    req = urllib.request.Request(
        f"{base}/chat/completions", data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json",
                 "Authorization": f"Bearer {os.environ['LLM_API_KEY']}"})
    with urllib.request.urlopen(req, timeout=120) as resp:
        reply = json.load(resp)["choices"][0]["message"]["content"]
    return parse_reply(reply)


# ---------------------------------------------------------------- scripted (loop test only)
def _swap(code, old, new):
    if old not in code:
        raise ValueError(f"scripted edit could not find: {old!r}")
    return code.replace(old, new)


SCRIPT = [
    ("[test] Imports the ground-truth simulator; the static check should reject it.",
     lambda c: "from tasks import pipe_flow\n" + c),
    ("[test] Subtracts a large constant so predictions go negative; the bounds check should reject it.",
     lambda c: _swap(c, "return np.exp(np.mean(preds, axis=0))",
                     "return np.exp(np.mean(preds, axis=0)) - 1e4")),
    ("[test] Wider hidden layers (32 -> 64) for more capacity.",
     lambda c: _swap(c, "HIDDEN = (32, 32)", "HIDDEN = (64, 64)")),
    ("[test] Raises an error inside fit(); the loop should log a crash and carry on.",
     lambda c: _swap(c, "def fit(X, y):", "def fit(X, y):\n    raise RuntimeError('deliberate')")),
]


def scripted_proposer(brief, code, history, step):
    hyp, edit = SCRIPT[step % len(SCRIPT)]
    return hyp, edit(code)
