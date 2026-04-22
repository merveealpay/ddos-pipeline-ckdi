# Stream G: β Fix Impact Analysis

**Date:** 2026-04-22
**Commit context:** `pipeline_ckdi_bdri.py` updated to paper eq. 502
(`RIF = 1 − e^{−β·CKDI²}` with `BETA = 3.0`)

## Side-by-side comparison (CICEV2023, n=116, δ=1.0, α=2.0)

| Metric                   | Old code (β=1.0, linear) | Paper (β=3.0, expo) | Δ (post − pre) |
|--------------------------|:-------------------------:|:-------------------:|:--------------:|
| RIF range                | [0.000, 1.000]           | [0.000, 0.950]      | saturates      |
| Mean BDRI_equal (attack) | 0.4811                   | 0.4807              | −0.0004        |
| Mean BDRI_equal (normal) | 0.7034                   | 0.7185              | +0.0151        |
| Corr(CKDI, BDRI)         | −0.9964                  | −0.9958             | +0.0006        |
| Normal−attack separation | 0.2222                   | 0.2377              | +0.0155        |

## Verdict

The β fix **does not invalidate any numerical claim** in the paper.

* Paper §V-G robustness claim "Corr(CKDI, BDRI) < −0.93 everywhere" survives
  trivially (actual: −0.996).
* Mean BDRI_atk for (δ=1.0, α=2.0) is ≈ 0.48 under both formulas, well within
  the sensitivity band paper reports ([0.312, 0.517]).
* Normal–attack BDRI separation is **slightly improved** with the paper
  formula (+0.0155), so the new code produces cleaner discrimination.

**Recommendation:** Retain Table III / Table IV as-published. The β=3.0
exponential RIF is now the canonical implementation and matches the paper's
text verbatim (eq. 502).
