# Native `hermes update` with required local patches

## Installed behavior

Use the native command from an independent terminal:

```powershell
hermes update --channel stable
```

This integration does not add a second updater, alias, persistent PATH edit or restart command. Native channel selection, backup, Git checks, PM dependency preparation and restart policy remain owned by Hermes.

Before the update pauses gateways or changes source, the integration checks its installation binding, inventories dirty/untracked paths against the active manifest and verifies the currently required patch set. Unknown edits or missing patches refuse the update. It writes a pending record outside the checkout with manifest, replay-driver, guard and binding hashes, plus updater PID and creation identity.

After the source move and native stash settlement, but before the new completion process starts, the pre-swap adapter calls the external replay driver. It verifies the entire registered set again and clears the pending record only after success. The adapter and launcher changes themselves are registered patches.

Emergency resume keeps the existing checkout-integrity gate, then checks required patches read-only. Missing/conflicted patches leave paused gateways stopped, keep the pending record and report an incomplete update. A required replay failure produces a real `partial` receipt and exit 1; Ctrl-C produces exit 130. No live update or gateway restart is part of installing this integration.

## Crossing a release that needs ported patches

Stage the ported set as `patches/replay-kit/manifest-<target commit>.json` plus its payloads before updating, and pass `--keep-stash` so the old patches stay parked instead of being re-applied onto the new code. Preparation and completion both use the guard's current-HEAD selection: the matching staged set when present, otherwise `manifest.json`, which fails closed if it is incompatible. The manual entry uses this same selection. Promoting a verified staged set to `manifest.json` is separate maintenance: obtain approval, confirm there is no pending update, and back up the original main manifest before replacing it; staged-file retirement is not required merely to run manual recovery.

## Interrupted-update protection

The existing installation-bound native launchers embed a stdlib-only check before importing checkout code. Normal launch only stats the pending record. While its updater PID and creation identity are live, another launch refuses. After an updater dies, a read-only rehearsal must prove every required patch is already present before startup is admitted. It never launches a new, unpatched tree just because Git moved HEAD.

The pending record is `patches/replay-kit/native-update-pending.json`. Do not delete it to bypass a failure. Run the existing manual recovery entry from a terminal:

```bash
bash <Hermes home>/bin/reapply-local-patches.sh --check
```

`--check` rehearses read-only; drop it only when you intend to apply the selected set. Then run the native command again. The launcher verifies the complete patch set before retiring a dead updater's record. An update killed halfway through a replay may also leave the replay driver's lock; verify its owner is dead before repairing it. Do not blindly remove live locks.

## Manual recovery entry

The installed entry at `<Hermes home>/bin/reapply-local-patches.sh` is an installation asset. It is not a second updater and does not restart Hermes. Its default form **applies** the selected patch set, so rehearse with `--check` unless an apply is what you want.

- **Manifest:** with no source argument, the entry calls the installed guard's `manifest_for`: `patches/replay-kit/manifest-<HEAD>.json` when the checked-out commit has a staged set, otherwise `manifest.json`. A staged set is used directly; removing it or copying it over `manifest.json` is not needed to make the entry follow the current commit. Promoting the staged set to `manifest.json` is a separate, user-approved step and cannot happen while a native update is pending.
- **Explicit sources:** `--manifest PATH` and the legacy positional patch (`bash <entry> PATH`, equivalent to `--patch PATH`) keep their old behavior and skip automatic HEAD discovery. `--check` and `--all-conflicts` are forwarded unchanged. A later `--workspace`/`--report` overrides the entry's default.
- **Interpreter:** candidates are tried in this order: `<source checkout>/.venv/Scripts/python.exe`, `<source checkout>/.venv/bin/python`, `HERMES_PATCH_PYTHON`, then `python`/`python3` found on `PATH`. Every candidate must pass a non-interactive Python 3.11 probe (capped at 20 seconds when the shell's `timeout` utility is available; without that utility no hard probe deadline is claimed). Windows Store / `Microsoft/WindowsApps` execution aliases are refused before execution, including when they come from `HERMES_PATCH_PYTHON` or `PATH`. A failed candidate falls through to the next one with a diagnostic; if none is usable the entry exits nonzero without running the driver.
- **Quoting:** `HERMES_PATCH_PYTHON` is one executable path or command name, never a command line with flags. It is never `eval`'d or word-split; paths with spaces and MSYS `/c/...` paths are quoted normally.

Back up the installed entry, guard, driver and active manifest before manual edits, and keep the logs of failed attempts. Never delete the pending record to make a failure disappear.

## Explicit boundaries

- Binding is opt-in and specific to one source checkout. Other profiles' files are not edited.
- Required local source patches refuse ZIP fallback **before download/swap** because the driver needs usable Git. Repair Git; do not silently drop patches.
- Standard generated launchers and generated runtime commands are protected. Direct interpreter calls such as `python -m hermes_cli.main` bypass the persisted launcher and are not a recovery shortcut.
- Power loss is not a cross-file atomic transaction. Keep backups and the pending record; recover and verify instead of claiming uninterrupted service.
- A malformed pending record, changed manifest/driver/guard/binding or missing external guard fails closed. Restore the captured valid inputs before retrying; do not waive hash checks.
- The active source manifest does not back up user configuration, skills, external plugins or credentials. A local Git branch is not an off-machine backup.
- The preserved stash is not dropped. Deliberately parked or unreviewed user changes can still make the native update partial even when registered patches are restored.

## Verification scope

The dedicated tests exercise native pre-pause preparation, native completion ordering, real partial receipts, emergency resume refusal, interruption, an actual killed temporary process, generated launcher refusal before application import, and restoration of the adapter's removed source file. They use isolated repositories/homes and never update or restart the live installation. Broad Windows regression failures must be compared by exact test ID to the same unchanged-source baseline; do not call the whole suite green.
