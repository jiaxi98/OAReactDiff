# OA-ReactDiff Failure Audit

This pilot separates four questions before modifying the model:

1. **Coverage:** does any of `K` samples reach the reference/intended TS basin?
2. **Selection:** does the confidence model select an available good candidate?
3. **Physical fidelity:** are low-RMSD candidates stationary and index-1?
4. **Reference ambiguity:** do high-RMSD candidates optimize to other valid saddles?

Do not equate RMSD, one negative curvature mode, or local optimization with intended R/P connectivity. The final
certificate remains a frequency calculation plus bidirectional IRC at the reference electronic-structure level.

## Immediate Local Audit

The bundled `demo/example-3` already contains 128 generated TSs, 128 DFT Hessians, 32 optimized TSs, six clusters,
and selected IRC endpoints. The lightweight audit needs only NumPy:

```bash
python -m oa_reactdiff.analyze.failure_audit \
  --demo-dir demo/example-3 \
  --output-dir /tmp/oa_failure_audit
```

It mass-weights each Hessian, removes translational/rotational modes, and combines the resulting spectrum with the
bundled force RMS values from `demo/example-3/summary.csv` (independent PySCF gradients, not Hessian-derived).
`strict_imaginary_modes` counts every negative mode and reproduces the paper/demo's
index-one rule; `significant_imaginary_modes` applies the configurable 50 cm-1 threshold as a robustness diagnostic.
The default 0.05 eV/A force threshold distinguishes a single-negative-mode geometry from a stationary TS-like
candidate. Missing force or Hessian evidence remains `NA`, never a failed check. The command writes `candidates.csv`
plus `summary.json`. The pure-NumPy permutation RMSD is suitable for triage; reproduce any reported value with
`oa_reactdiff.analyze.rmsd` and Pymatgen.

## GPU Candidate Generation

First verify the checkpoint, test split, and source-index mapping without importing PyTorch:

```bash
python experiments/oa_failure_audit/generate_candidates.py --dry-run
```

Run a small GPU pilot before attempting the paper-scale `40 x 1,073` evaluation:

```bash
python experiments/oa_failure_audit/generate_candidates.py \
  --device cuda \
  --max-reactions 8 \
  --samples-per-reaction 8 \
  --resamplings 2 \
  --jump-length 2 \
  --output-dir experiments/oa_failure_audit/outputs/pilot \
  --resume
```

This writes one atomic manifest and timing record per reaction. Re-run the identical command with `--resume` after an
interruption. The `2/2` setting is a timing and pipeline pilot; the repository's paper evaluation uses
`--resamplings 10 --jump-length 10`. Keep those results in separate directories. Expand only after checking runtime,
memory, and output validity:

```bash
python experiments/oa_failure_audit/generate_candidates.py \
  --device cuda \
  --max-reactions 1073 \
  --samples-per-reaction 40 \
  --output-dir experiments/oa_failure_audit/outputs/full \
  --resume
```

`candidate_manifest.csv` deliberately reserves columns for confidence, force RMS, strict imaginary-mode count,
optimization, and IRC outcome. The repository contains a diffusion checkpoint but no confidence-model checkpoint,
so selection failure cannot yet be reproduced faithfully.

After filling any available diagnostic columns, aggregate the manifest with:

```bash
python -m experiments.oa_failure_audit.summarize_manifest \
  --manifest experiments/oa_failure_audit/outputs/pilot/candidate_manifest.csv \
  --output-dir experiments/oa_failure_audit/outputs/pilot/summary
```

The output uses three states: `true`, `false`, and `NA`. Coverage failure means no candidate reaches RMSD below 0.2;
ranking failure requires complete confidence scores and an available good candidate; geometry-good/physics-bad
requires low RMSD plus force or strict-mode evidence. Populate `irc_connects_target` only after validating and
following a saddle in both directions. For the bundled demo only, optimized-cluster labels provide weaker fallback
evidence when IRC labels are absent; `alternate_saddle_evidence` records which source was used. Missing confidence,
force, Hessian, or IRC/cluster evidence remains `NA`.

## HORM E/F/HVP Screening

Use the official paired LEFTNet checkpoints: `left_orig.ckpt` was trained on E/F, while `left.ckpt` adds Hessian
supervision. `screen_horm.py` verifies the official SHA-256 values and HORM commit before deserializing a checkpoint.
Run the models separately in the HORM environment:

```bash
python experiments/oa_failure_audit/screen_horm.py \
  --manifest outputs/pilot/candidate_manifest.csv \
  --horm-repo /path/to/HORM \
  --checkpoint /path/to/left_orig.ckpt \
  --label horm_left_ef \
  --output outputs/pilot/horm_left_ef.csv \
  --device cuda --resume
```

Repeat with `left.ckpt` and label `horm_left_efh`. The screen computes conservative E/F and the two algebraically
lowest vibrational eigenpairs using exact autograd HVPs. Two modes are sufficient to classify `0`, `1`, or `>=2`
significant negative modes without constructing a full Hessian. The CSV retains frequencies, force RMS, eigensolver
residuals, HVP calls, runtime, and errors; it does not collapse them into a hand-tuned score.

Merge the two results and make a reaction-capped, deterministic DFT worklist:

```bash
python experiments/oa_failure_audit/merge_screening.py \
  --manifest outputs/pilot/candidate_manifest.csv \
  --screen horm_left_ef=outputs/pilot/horm_left_ef.csv \
  --screen horm_left_efh=outputs/pilot/horm_left_efh.csv \
  --output outputs/pilot/screened.csv
python experiments/oa_failure_audit/select_dft_subset.py \
  --manifest outputs/pilot/screened.csv \
  --output outputs/pilot/dft_subset.csv \
  --budget 96 --max-per-reaction 2
```

The strata prioritize E/F-versus-E/F/H curvature disagreements, coverage failures, low-RMSD/physics-bad cases,
high-RMSD index-one candidates, and controls across RMSD and force bins. An `8 x 8` pilot can provide at most 16 cases
under a two-per-reaction cap; generate at least 25--50 reactions before targeting a 50--200 case DFT subset.

## DFT Hessian and IRC Gate

HORM used neutral singlets, GPU4PySCF 1.3.0, wB97X/6-31G(d), SCF energy tolerance `1e-10` Hartree, and orbital-gradient
tolerance `1e-5`. The runner records these settings and checkpoints every completed candidate:

```bash
python experiments/oa_failure_audit/run_dft_hessian.py \
  --subset outputs/pilot/dft_subset.csv \
  --output-dir outputs/pilot/dft_full \
  --backend gpu4pyscf --resume
```

Run two cases first with `--max-candidates 2` in a separate output directory. After merging `dft_results.csv` with
the subset under prefix `dft`, use `prepare_irc_worklist.py`. A stationary DFT index-one point is IRC-ready; an
index-one but nonstationary raw sample must undergo TS optimization and another Hessian first. Non-index-one anomalies
are review/optimization cases, not valid starting points for an IRC.

## Colab Environments

The OA checkpoint requires its historical PyTorch 1.12 stack, whereas HORM uses PyTorch 2.2.1 and DFT uses
GPU4PySCF. Do not install them into one environment. Open the notebooks directly from GitHub and run them in order,
using a fresh GPU runtime for each:

- [01 candidate generation](https://colab.research.google.com/github/jiaxi98/OAReactDiff/blob/agent/oa-failure-audit/experiments/oa_failure_audit/01_colab_generate.ipynb): resumable `8 x 8` generation and measured extrapolation.
- [02 HORM screening](https://colab.research.google.com/github/jiaxi98/OAReactDiff/blob/agent/oa-failure-audit/experiments/oa_failure_audit/02_colab_horm_screen.ipynb): verified paired HORM screens and DFT stratification.
- [03 DFT Hessians](https://colab.research.google.com/github/jiaxi98/OAReactDiff/blob/agent/oa-failure-audit/experiments/oa_failure_audit/03_colab_dft_hessian.ipynb): two-case GPU4PySCF benchmark, then the approved subset.

Each notebook shallow-clones the audit branch. The first downloads the Git LFS checkpoint from GitHub and verifies
its SHA-256; no upload bundle is required. Outputs remain under `MyDrive/OAReactDiff/audit_outputs/` across runtime
disconnects. After the branch merges, change `REPOSITORY_URL` and the three links from `jiaxi98/OAReactDiff` to
`chenruduan/OAReactDiff`, and change `REPOSITORY_REF` from `agent/oa-failure-audit` to `main`.

## Runtime Choice

The local Apple Silicon machine is appropriate for the NumPy artifact audit, which takes well under a second. It is
not the primary OA generation platform: MPS is deliberately not exposed for the pinned PyTorch/PyG stack. On a Colab
V100/L4/A100, expect roughly 2--5 minutes for the lightweight `8 x 8` pilot, plus 15--40 minutes for first-time
environment setup. Treat those as planning estimates and replace them with `timing_summary.json` from the allocated
GPU. Ordinary CPU PySCF on the local machine can take roughly 5--60 minutes per 5--21 atom Hessian; benchmark two
cases before scheduling 50--200. GPU4PySCF can change that estimate substantially.

## Interpretation Boundary

MLIP force or curvature is a screening signal, not a TS certificate. DFT index-one curvature without stationarity is
also insufficient. Intended R/P connectivity is established only after a stationary index-one structure connects to
the target basins under a bidirectional IRC (or a clearly documented equivalent endpoint workflow).
