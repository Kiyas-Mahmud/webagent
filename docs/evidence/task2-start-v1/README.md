# Task 2 engineering start evidence

These are setup and regression checks, **not model performance or browser
completion results**. Follow [the current checklist](../../TASK2_AGENT_INTEGRATION.md).

- `preflight.json`: first asset check before the isolated agent installation.
- `preflight-after-install.json`: current asset/environment check; readiness
  remains false. All 5,069 Task 1 final evidence files remain unchanged.
- `upstream-source.json`: exact Browser Use source archive and source hashes.
- `dependency-lock.json`, `constraints.txt`: 104 resolved package versions and
  available artifact hashes used by the isolated installation.
- `dependency-resolution.log`, `installation.log`: preserved resolution/install
  output; neither original study environment was upgraded.
- `native-agent-import.json`: native Agent/protocol imports and hook signatures;
  byte identity of installed service/interface files against pinned source.
- `engineering-receipt.json`: 23 passing CPU test cases, source/evidence hashes,
  successful dependency consistency check, zero model calls/browser episodes.

The initial native import encountered the read-only home configuration directory.
Setting `BROWSER_USE_CONFIG_DIR` to the workspace-local `.task2-assets` directory
resolved it without changing upstream source. Telemetry was disabled for the
import probe. Browser processes were not launched by these checks.

Memory artifact identity passes with its original PC-01 query checkpoint and
correctly fails with the InternVL checkpoint despite equal 768 dimensions.
Live query parity is pending. No memory content or embeddings were changed.
