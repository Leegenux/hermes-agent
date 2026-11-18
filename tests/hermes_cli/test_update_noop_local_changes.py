"""No-op planning must leave local work and older stashes exactly where they are.

Regression break: stashing before the exact-target/no-move decision. Real Git
fixtures exercise staged + unstaged edits, untracked patch files and two older
fixture stashes. Only TTY answers and the external completion worker are doubled.
"""
import subprocess
import sys

import pytest

from hermes_cli import update_cmd, update_cmd_stash
from tests.hermes_cli.test_update_target_identity import (
    _publish_stable,
    update_tree,
)


def git(root, *args):
    return subprocess.run(
        ["git", *args], cwd=root, check=True, capture_output=True,
        text=True, encoding="utf-8",
    ).stdout.strip()


class TTYProxy:
    def __init__(self, stream, tty):
        self.stream = stream
        self.tty = tty

    def isatty(self):
        return self.tty

    def __getattr__(self, name):
        return getattr(self.stream, name)


@pytest.fixture(autouse=True)
def isolate_in_memory_stash_notice(monkeypatch):
    # Each real CLI starts a new process. These direct helper tests deliberately
    # leave fixture stashes unsettled, so do not carry their notice into the next
    # test. This changes memory only, not any Git ref or native pending file.
    monkeypatch.setattr(update_cmd_stash, "_pending_autostash", None)


@pytest.fixture
def dirty_repo(tmp_path, monkeypatch):
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", str(tmp_path / "git-config"))
    monkeypatch.setenv("GIT_CONFIG_NOSYSTEM", "1")
    monkeypatch.setenv("GIT_ALLOW_PROTOCOL", "file")
    root = tmp_path / "checkout"
    root.mkdir()
    git(root, "init", "-q", "-b", "main")
    git(root, "config", "user.name", "Fixture")
    git(root, "config", "user.email", "fixture@example.invalid")
    git(root, "config", "commit.gpgsign", "false")
    git(root, "config", "core.autocrlf", "false")
    tracked = root / "tracked.txt"
    generated = root / "patch_generated.py"
    tracked.write_bytes(b"base\n")
    git(root, "add", "tracked.txt")
    git(root, "commit", "-qm", "base")
    base = git(root, "rev-parse", "HEAD")
    tracked.write_bytes(b"release\n")
    generated.write_bytes(b"upstream_value = 1\n")
    git(root, "add", ".")
    git(root, "commit", "-qm", "release")
    future = git(root, "rev-parse", "HEAD")
    # Destructive fixture preparation is confined to tmp_path, never the install.
    git(root, "reset", "--hard", base)
    git(root, "update-ref", "refs/remotes/origin/main", base)
    for number in (1, 2):
        tracked.write_text(f"older fixture work {number}\n", encoding="utf-8")
        git(root, "stash", "push", "-qm", f"older-fixture-{number}")
    assert len((root / ".git/logs/refs/stash").read_bytes().splitlines()) == 2
    tracked.write_bytes(b"staged local change\n")
    git(root, "add", "tracked.txt")
    tracked.write_bytes(b"staged plus unstaged local change\n")
    generated.write_bytes(b"local_patch = True\r\n")
    monkeypatch.setattr(update_cmd._m(), "PROJECT_ROOT", root)
    return root, base, future


def snapshot(root):
    """Raw bytes, not reconstructed state or a Git command that refreshes index."""
    return {
        name: (root / name).read_bytes()
        for name in ("tracked.txt", "patch_generated.py", ".git/index",
                     ".git/refs/stash", ".git/logs/refs/stash", ".git/HEAD")
    }


def prepare(root, *, target_ref=None, is_fork=False, assume_yes=True):
    return update_cmd._prepare_checkout_for_update(
        ["git"], "main", git(root, "rev-parse", "--abbrev-ref", "HEAD"),
        is_fork=is_fork, assume_yes=assume_yes, gateway_mode=False,
        gw_input_fn=None, switch_branch=False, target_ref=target_ref,
        _windows_gateway_resume=None,
    )


@pytest.mark.parametrize("channel", ["release", "main"])
@pytest.mark.parametrize("tty,assume_yes", [(True, False), (False, False), (True, True), (False, True)])
def test_same_commit_noop_keeps_work_index_and_older_stashes(
    dirty_repo, monkeypatch, channel, tty, assume_yes,
):
    root, base, _future = dirty_repo
    before = snapshot(root)
    monkeypatch.setattr(sys, "stdin", TTYProxy(sys.stdin, tty))
    monkeypatch.setattr(sys, "stdout", TTYProxy(sys.stdout, tty))
    monkeypatch.setattr("builtins.input", lambda *_: pytest.fail("no-op asked for input"))
    plan = prepare(root, target_ref=base if channel == "release" else None, assume_yes=assume_yes)
    assert plan.auto_stash_ref is None
    assert plan.commit_count == 0
    assert plan.prompt_for_restore is False
    assert snapshot(root) == before

    # Keep the real no-op finishing caller. Its external worker is the boundary:
    # invoking dependency installs / gateway resumes is not this regression test.
    completions = []
    monkeypatch.setattr(update_cmd, "_complete_source_update", completions.append)
    monkeypatch.setattr(update_cmd._m(), "_restore_stashed_changes",
                        lambda *_a, **_k: pytest.fail("no-op attempted a stash restore"))
    request = {"force_venv": True, "keep_stash": True}
    update_cmd._finish_already_up_to_date(
        ["git"], "main", "main", plan, gw_input_fn=None,
        completion_request=request, _windows_gateway_resume=None,
    )
    assert completions == [{"force_venv": True, "keep_stash": True,
                            "expected_sha": base, "completion_message": "✓ Already up to date!"}]
    assert snapshot(root) == before


def test_real_target_change_still_stashes_colliding_local_work(dirty_repo):
    root, base, future = dirty_repo
    old_log = (root / ".git/logs/refs/stash").read_bytes()
    plan = prepare(root, target_ref=future)
    assert plan.auto_stash_ref is not None
    assert plan.commit_count == -1
    assert git(root, "rev-parse", "HEAD") == base
    assert (root / "tracked.txt").read_bytes() == b"base\n"
    assert not (root / "patch_generated.py").exists()
    assert (root / ".git/logs/refs/stash").read_bytes().startswith(old_log)
    # Both the staged/unstaged tracked edit and colliding untracked patch file
    # really exist inside the new fixture stash, rather than only a mocked call.
    assert git(root, "show", f"{plan.auto_stash_ref}:tracked.txt") == "staged plus unstaged local change"
    assert git(root, "show", f"{plan.auto_stash_ref}^3:patch_generated.py") == "local_patch = True"


def test_same_commit_branch_switch_still_parks_local_work(dirty_repo):
    root, base, _future = dirty_repo
    git(root, "checkout", "-q", "--detach", base)
    old_log = (root / ".git/logs/refs/stash").read_bytes()
    plan = prepare(root)
    assert plan.auto_stash_ref is not None
    assert git(root, "rev-parse", "--abbrev-ref", "HEAD") == "main"
    assert git(root, "rev-parse", "HEAD") == base
    assert (root / ".git/logs/refs/stash").read_bytes().startswith(old_log)


def test_fork_same_origin_can_still_sync_upstream(dirty_repo, monkeypatch):
    root, base, future = dirty_repo
    old_log = (root / ".git/logs/refs/stash").read_bytes()

    def sync(_git_cmd, cwd, **_kwargs):
        assert cwd == root
        assert (root / "tracked.txt").read_bytes() == b"base\n"
        assert not (root / "patch_generated.py").exists()
        git(root, "merge", "--ff-only", future)
        return True

    monkeypatch.setattr(update_cmd._m(), "_sync_with_upstream_if_needed", sync)
    plan = prepare(root, is_fork=True)
    assert plan.auto_stash_ref is not None
    assert plan.commit_count > 0
    assert plan.pre_sync_sha == base
    assert git(root, "rev-parse", "HEAD") == future
    assert (root / ".git/logs/refs/stash").read_bytes().startswith(old_log)


def test_unresolved_target_does_not_claim_noop_or_skip_protection(dirty_repo):
    root, _base, _future = dirty_repo
    plan = prepare(root, target_ref="refs/tags/does-not-exist")
    assert plan.auto_stash_ref is not None
    assert plan.commit_count == -1


def test_unresolved_head_keeps_the_existing_protection_path(dirty_repo, monkeypatch):
    root, base, _future = dirty_repo
    monkeypatch.setattr(update_cmd, "_capture_head_sha", lambda *_: None)
    plan = prepare(root, target_ref=base)
    assert plan.auto_stash_ref is not None
    assert plan.commit_count == -1


def test_ancestor_is_not_mistaken_for_exact_head(dirty_repo):
    root, _base, _future = dirty_repo
    ahead = git(root, "commit-tree", "HEAD^{tree}", "-p", "HEAD", "-m", "local-ahead")
    git(root, "update-ref", "HEAD", ahead)
    plan = prepare(root)
    assert plan.auto_stash_ref is not None
    assert plan.commit_count == 0
    assert git(root, "rev-parse", "HEAD") == ahead


@pytest.mark.parametrize("keep_stash", [False, True])
def test_stable_noop_command_still_hands_off_completion_without_stash(
    update_tree, monkeypatch, keep_stash,
):
    """Real command caller + local-file remote; lifecycle/worker seams stay isolated."""
    t = update_tree
    # The fixture clones before publishing this commit to its local-file origin.
    git(t.clone, "fetch", "-q", "origin")
    git(t.clone, "checkout", "-q", "--detach", t.wanted)
    _publish_stable(monkeypatch, t)
    # Model the official repository identity without contacting GitHub. Git's
    # real origin remains the local-file fixture for every fetch/tag operation.
    def official_origin(_git_cmd, cwd):
        assert cwd == t.clone
        return "https://github.com/NousResearch/hermes-agent.git"

    monkeypatch.setattr(update_cmd._m(), "_get_origin_url", official_origin)
    local = t.clone / ".gitignore"
    for number in (1, 2):
        local.write_bytes(f"# older fixture {number}\n".encode())
        git(t.clone, "stash", "push", "-qm", f"older-fixture-{number}")
    staged = local.read_bytes() + b"# staged local change\n"
    local.write_bytes(staged)
    git(t.clone, "add", ".gitignore")
    working = staged + b"# unstaged local change\n"
    local.write_bytes(working)
    untracked = t.clone / "local_patch.py"
    untracked.write_bytes(b"local_patch = True\r\n")
    old_ref = (t.clone / ".git/refs/stash").read_bytes()
    old_log = (t.clone / ".git/logs/refs/stash").read_bytes()
    index_before = git(t.clone, "show", ":.gitignore")
    t.args.yes = False
    t.args.keep_stash = keep_stash
    monkeypatch.setattr(sys, "stdin", TTYProxy(sys.stdin, True))
    monkeypatch.setattr(sys, "stdout", TTYProxy(sys.stdout, True))
    monkeypatch.setattr("builtins.input", lambda *_: pytest.fail("same stable target prompted"))
    monkeypatch.setattr(update_cmd._m(), "_restore_stashed_changes",
                        lambda *_a, **_k: pytest.fail("same stable target attempted restore"))

    update_cmd._cmd_update_impl(t.args, gateway_mode=False)

    assert len(t.requests) == 1
    assert t.requests[0]["expected_sha"] == t.wanted
    assert t.requests[0]["completion_message"] == "✓ Already up to date!"
    assert local.read_bytes() == working
    assert untracked.read_bytes() == b"local_patch = True\r\n"
    assert git(t.clone, "show", ":.gitignore") == index_before
    assert (t.clone / ".git/refs/stash").read_bytes() == old_ref
    assert (t.clone / ".git/logs/refs/stash").read_bytes() == old_log
