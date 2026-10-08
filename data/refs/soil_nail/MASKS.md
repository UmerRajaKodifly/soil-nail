# Reference masks

| Variant | Directory | Definition (default for experiments) |
|---------|-----------|--------------------------------------|
| **B (default)** | `masks/` | White corrugated **sleeve + bar end + hose roots only**. Exclude dark hole interior, rim soil, whiteboard, hands, boots, scaffold, date stamp. |
| **A (comparison)** | `masks_A/` | Hole opening + everything emerging from it. |

Annotate:

```bash
source .venv/bin/activate
python tools/annotate_mask.py --variant B
# open http://127.0.0.1:7860
```

1. Click the **photo** (crosshair) on sleeve / bar / hose roots.
2. Tick **Click = EXCLUDE**, then click hole / board / soil to carve those out.
3. **Try alternate mask** if needed → **Save mask** → **Next**.

Sanity: masks covering **> 25%** of the image are flagged in `summary.csv` (`mask_area_flag=1`).
