# TESSELLATE asteroid rotation catalogue

Public data products and website for asteroid rotation periods, phase-folded lightcurves and
convex shape models derived from TESS sectors 27-55.

**This repository hosts data products and the website only.** The pipeline, LCDB reconciliation
and shape-inversion code live in the working repositories and are not mirrored here.

## Layout

| path | contents |
|---|---|
| `web/` | the published site (GitHub Pages) |
| `data/lightcurves/` | one phase-folded, binned lightcurve per asteroid (JSON, ~2 KB each) |
| `data/shapes/` | convex shape models (JSON mesh + solution metadata) |
| `data/figures/` | population figures shown on the splash page |
| `data/tables/` | catalogue and reference tables |

## Data products

**Folded lightcurves.** Phase-folded stacked photometry, binned, with the standard error on each
bin and the point count. Where a shape model exists, the convex model's own fitted lightcurve is
included on the same phase grid for overlay.

**Shape models.** Convex inversion (DAMIT `convexinv`), 289 facets. Single-apparition data does
not constrain the pole -- solutions from different starting poles scatter by 100 degrees or more
-- so the shape is meaningful while its orientation is not. `lambda_circ_std_deg` records the
spread on every solution and should be read before trusting an orientation.

**Tables.** `lcdb_updated.csv` is the LCDB summary extended with a `period_updated` column
carrying TESS precision where warranted; the published `Period` is never overwritten and
`period_source` gives the provenance of each row. `vetted_lcdb_errors.csv` lists 172 literature
periods confirmed wrong by inspection.
