"""Checks a proposed model must pass before its score counts.

  static_check   reads the proposed code without running it
  physics_check  reads the constraint measurements made by harness.py

The static check filters careless or lazy proposals (reading data files,
importing the ground-truth simulators, shelling out). It is NOT a security
sandbox: run the agent only where executing model-written code is acceptable.
"""
import ast

ALLOWED_IMPORTS = {"numpy", "pandas", "scipy", "sklearn", "math", "warnings"}
BANNED_NAMES = {"open", "eval", "exec", "compile", "__import__", "input", "globals",
                "breakpoint", "getattr", "setattr", "vars"}
BANNED_ATTRS = {"read_csv", "read_table", "read_pickle", "read_json", "to_csv", "to_pickle",
                "system", "popen", "load", "loadtxt", "genfromtxt", "save", "savetxt", "fromfile"}


def static_check(code: str) -> list[str]:
    """Return a list of problems with the proposed source. Empty list = pass."""
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return [f"syntax error: {e.msg} (line {e.lineno})"]

    problems = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            mods = [a.name.split(".")[0] for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            mods = [(node.module or "").split(".")[0]] if node.level == 0 else ["<relative import>"]
        else:
            mods = []
        for m in mods:
            if m not in ALLOWED_IMPORTS:
                problems.append(f"import of '{m}' is not allowed")
        if isinstance(node, ast.Name) and node.id in BANNED_NAMES:
            problems.append(f"use of '{node.id}' is not allowed")
        if isinstance(node, ast.Attribute) and node.attr in BANNED_ATTRS:
            problems.append(f"call to '.{node.attr}' is not allowed")

    defined = {n.name for n in tree.body if isinstance(n, ast.FunctionDef)}
    for required in ("fit", "predict"):
        if required not in defined:
            problems.append(f"missing required function '{required}'")
    return sorted(set(problems))


def physics_check(metrics: dict) -> list[str]:
    """Return the physical constraints the fitted model violates. Empty list = pass."""
    return [f"{name}: {c['value']} > limit {c['limit']}"
            for name, c in metrics["constraints"].items() if not c["pass"]]
