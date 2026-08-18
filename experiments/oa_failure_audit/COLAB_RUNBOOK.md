# Colab Runbook: HORM and DFT

Run stages 02 and 03 in separate, fresh Colab GPU runtimes. Colab `/content` is temporary; each stage consumes a
locally saved archive and produces another archive that must be downloaded before disconnecting. Google Drive is
not used.

## Stage 02: HORM Screening

Open [02_colab_horm_screen.ipynb](https://colab.research.google.com/github/jiaxi98/OAReactDiff/blob/agent/oa-failure-audit/experiments/oa_failure_audit/02_colab_horm_screen.ipynb), choose a T4 or better GPU, and upload
`oa_audit_01_generation_8x8_r2_j2.tar.gz` to `/content` using the left Files panel.

Run cells in order:

1. Validate and extract the stage-01 archive.
2. Clone the fork, create the isolated `oa-horm` environment, and verify CUDA and the environment `libstdc++`.
3. Download both verified HORM checkpoints. Both SHA-256 checks must report `OK`.
4. Run the one-candidate E/F and E/F/H smoke tests. Do not continue unless both CSV rows have no error and the
   notebook prints `Both checkpoints passed`.
5. Run both 64-candidate screens. Re-run this cell after a transient interruption; `--resume` skips completed rows.
6. Merge the screens and construct the deterministic, reaction-capped DFT subset.
7. Package the results, refresh the Files panel, and download
   `oa_audit_02_horm_screen_generation_8x8_r2_j2.tar.gz`.

The two primary outputs are `horm_left_ef.csv` and `horm_left_efh.csv`. Each must contain 64 successful rows. The
stage-02 archive also contains the generated XYZ structures, merged manifest, and `dft_subset.csv`.

## Stage 03: DFT Hessians

Start another fresh GPU runtime and open
[03_colab_dft_hessian.ipynb](https://colab.research.google.com/github/jiaxi98/OAReactDiff/blob/agent/oa-failure-audit/experiments/oa_failure_audit/03_colab_dft_hessian.ipynb). Upload the stage-02 archive to `/content` through the Files panel.

Run cells in order:

1. Validate and extract the archive, clone the fork, and create the isolated `oa-dft` GPU4PySCF environment.
2. Run the two-case benchmark. The next cell verifies that both cases completed without errors and reports the
   measured time per case plus projections for the actual pilot subset and 96 cases.
3. Inspect `dft_benchmark_two/dft_results.csv` and the printed timing. Continue only if the benchmark passes and the
   projected runtime fits the remaining Colab session.
4. Change `RUN_FULL_SUBSET = False` to `RUN_FULL_SUBSET = True`, then run that cell. It computes every selected case
   and verifies that the result count matches `dft_subset.csv`. If execution is interrupted while the runtime remains
   alive, re-run the same cell; `--resume` preserves successful cases.
5. Run the gate cell to merge DFT evidence and create `irc_worklist.csv`. This file schedules eligible IRC cases; the
   notebook does not itself perform IRC calculations.
6. Run the packaging cell and download
   `oa_audit_03_dft_hessian_generation_8x8_r2_j2.tar.gz` from the Files panel.

The `8 x 8` pilot contains at most 16 DFT cases because selection is capped at two candidates per reaction. Do not
interpret the printed 96-case estimate as the pilot workload.

## Recovery and Local Handoff

- `CXXABI_1.3.15 not found` indicates an old runtime or notebook. Delete the runtime and reopen the current branch.
- A micromamba prefix under `/root/.local/share/mamba` also indicates stale notebook code; the current environments
  live under `/content/micromamba/envs/`.
- Colab's `Could not load the JavaScript files` dialog is a browser authentication or third-party-cookie problem,
  not a model error. Refresh, reauthenticate, and allow cookies for Colab before continuing.
- A deleted or disconnected runtime cannot be resumed because `/content` is ephemeral. Restart the stage from its
  locally downloaded input archive.

For local analysis, unpack downloaded archives beneath the ignored `experiments/oa_failure_audit/outputs/`
directory. Preserve the archive itself until its printed SHA-256 has been checked.
