import subprocess
from contextlib import contextmanager


def _git(*args):
    return subprocess.run(
        ["git", *args],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()


def get_repo_state():
    return {
        "commit": _git("rev-parse", "HEAD"),
        "patch": _git("diff", "HEAD"),
    }


def restore_repo_state(state):
    # Go back to the saved commit
    _git("checkout", state["commit"])

    # Reapply uncommitted changes if there were any
    if state["patch"]:
        subprocess.run(
            ["git", "apply", "-"],
            input=state["patch"],
            text=True,
            check=True,
        )


@contextmanager
def temporary_repo_state(target_commit, target_patch=""):
    original_state = get_repo_state()

    try:
        # Move to experiment state
        _git("checkout", target_commit)

        if target_patch:
            subprocess.run(
                ["git", "apply", "-"],
                input=target_patch,
                text=True,
                check=True,
            )

        yield

    finally:
        # Restore original state no matter what
        restore_repo_state(original_state)