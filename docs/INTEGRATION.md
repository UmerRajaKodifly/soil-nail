# Integration contracts

## Site-walk images

| Field | Rule |
|-------|------|
| Projection | Full equirectangular panorama |
| Aspect | Exactly `W == 2 * H` (measured: all 591 are 3840×1920) |
| Rejection | `soilnail.projection.errors.NotFullEquirectError` |
| EXIF transpose | Applied via `ImageOps.exif_transpose` (none present on this dataset) |
| Colour | uint8 RGB; no sRGB↔linear conversion in projection |

## Cubemap faces

Produced only by `soilnail.projection.cubemap.equirect_to_cubemap`. Face ids: `F R B L U D` (`docs/PROJECTION.md`).

## Detections

Face-coordinate mask + `SphericalGeom` + optional equirect polygon. Matching labels↔dets: reproject to label face or compare on sphere (`configs/projection.yaml` / eval config).

## LOO golden

Unchanged — perspective refs only. Must still pass after SAM embedding-cache changes.
