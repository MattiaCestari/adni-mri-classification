import subprocess
from contextlib import contextmanager

def _git(*args, input=None, cwd=None, text=True):
    out = subprocess.run(
        ["git", *args],
        capture_output=True,
        check=True,
        input=input,
        cwd=cwd,
    ).stdout

    if text:
        return out.strip()
    return out

def get_repo_state():
    return {
        "commit": _git("rev-parse", "HEAD"),
        "patch": _git("diff", "HEAD", text=False),
        "branch": _git("rev-parse", "--abbrev-ref", "HEAD")
    }

@contextmanager
def repo_state(target_commit, target_patch=""):
    original_state = get_repo_state()

    try:
        # Move to experiment state
        _git("reset", "--hard")

        _git("checkout", target_commit)

        if target_patch:
            root = _git("rev-parse", "--show-toplevel")
            _git("apply", "-", input=target_patch, cwd=root)

        yield

    finally:
        # Restore original state
        _git("reset", "--hard")
        _git("checkout", original_state["branch"])

        if target_patch:
            root = _git("rev-parse", "--show-toplevel")
            _git("apply", "-", input=original_state["patch"], cwd=root)

       