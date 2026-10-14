# Hardcode audit (angles / FOV / face size)
Hits outside allowlist: 4

## Findings
- `hashutil.py:8: possible face_size literal: _CHUNK = 1024 * 1024`
- `runner.py:172: possible face_size literal: return float(torch.cuda.max_memory_allocated() / (1024**2))`
- `ingest/safe_zip.py:101: possible face_size literal: chunk = src.read(min(1024 * 1024, remaining))`
- `ingest/csv_profile.py:62: possible FOV/right-angle literal: if abs(la_max) <= 90 and abs(lo_max) <= 180:`
