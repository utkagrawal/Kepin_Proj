# Dynamic Koopman KePIN

Goal: beat the original KePIN on C-MAPSS by letting the Koopman operator change over time, with a fair comparison (same pipeline, same 3 seeds as KePIN).

## What is new here

**`eigen_koopman.py` — eigen-parameterised Koopman layer.**

The operator is built from its eigenvalues:

```
K = P · blockdiag(ρ_k R(θ_k)) · Pᵀ
```

- `P` is orthogonal (a Cayley transform).
- `ρ_k = sigmoid(s_k)` is the decay of mode k.
- `θ_k` is the frequency of mode k.

This gives three benefits:

- **No eigendecomposition.** The eigenvalues are `ρ_k e^{±iθ_k}` by construction.
- **Cheap propagation.** Rollouts run in mode coordinates at O(d) per step.
- **Interpretable parameters.** Decay and frequency of every mode are explicit.

The old conditioned layer called `tf.linalg.eigvals` on one 128×128 matrix per sample, on CPU. That took about 450 s per epoch.

The layer has four modes:

| Mode | Operator |
|---|---|
| `static` | Fixed operator. A drop-in replacement for the original layer. |
| `exo` (variant A) | The operator changes **at every time step** with the operating condition `u_t` (a 6-way one-hot): `s_t = s + tanh(f(u_t))`, `θ_t = θ + 0.5·tanh(g(u_t))`. All steps share the basis P, so products of operators stay closed-form (decays multiply, angles add). |
| `health` (variant B) | Decay is shifted by a learned health index `h ∈ (0,1)` per window, read from the latent sequence: `s_k + tanh(a_k·(2h−1))`. `a_k` is each mode's sensitivity to health. `h` is saved per test engine for the XAI analysis. |
| `both` | A + B. Operating conditions drive ρ and θ per step, and health shifts the decay on top. |

All dynamic modes are initialised to behave exactly like `static`.

The layer is checked against an explicit dense K: one-step and 3-step predictions, eigenvalues and loss all agree to float precision.

**`dk_model.py`** is the original KePIN model with the Koopman layer swappable:

- `--koopman svd`: the original layer
- `--koopman eig`: the new layer

**`dk_data.py`** uses the original KePIN loader (same windows, scaling and EMA smoothing) and adds:

- the engine id of every training window
- the operating-condition one-hot at every time step

**`train.py`** runs one (dataset, arm, protocol, run) combination. It uses:

- the original `CMAPSS_CONFIGS` hyper-parameters
- the original `EnhancedKePINTrainer`
- the original seeds (`42 + 100·run_id`, 3 runs as in the paper)

## Evaluation protocols

The original KePIN code passes the **test set** as validation data. The best epoch, and the choice between SWA and best weights, are both picked on test RMSE. The paper's numbers are therefore optimistic. We report both protocols:

| Protocol | Validation | Use |
|---|---|---|
| `clean` | 20% of training engines held out (fixed split); the test set is used once | Main results |
| `paper` | Test set (same as original KePIN) | Only for comparison with the paper table |

## Running

Only 2 jobs per user are allowed (`mtech` QOS), so each job takes a whole node (2 GPUs) and runs 2 workers per GPU. The workers pull tasks from a shared queue.

```bash
python prepare_dirs.py tasks/phase1.txt     # pre-create output folders (avoids NFS races)
sbatch -J dk_queue1 slurm/queue.slurm tasks/phase1.txt
sbatch -J dk_queue2 slurm/queue.slurm tasks/phase1.txt
python aggregate.py                  # tables: mean ± std over 3 runs + 3-run ensemble
```

- **Resuming is safe.** Finished tasks (those with a result JSON) are skipped.
- **Retrying killed tasks:** if a job was killed mid-task, resubmit with `--reclaim` added to the worker command line.
- **GPU setup:** the job scripts must put the pip CUDA libraries on `LD_LIBRARY_PATH`. Without them, TensorFlow silently runs on CPU (about 5× slower). The scripts abort if TF sees no GPU.

| Path | Contents |
|---|---|
| `runs/<protocol>/<dataset>/<arm>/runX_*` | Per-run result, predictions, history, weights |
| `runs/logs/` | Per-task training logs |
| `slurm/logs/` | Job logs |

## Experiment log

**18 Oct 2026, phase 1.** Submitted `tasks/phase1.txt` (60 runs, all datasets) as jobs 4890 and 4891 on P100.

**18 Oct 2026, narrowed to FD002 only (user decision, to get results faster).**

- **Still running from phase 1:** the 8 runs already in progress, all FD002 with the clean protocol: `svd_static` r0–2, `eig_static` r0–2, `eig_exo` r0–1.
- **Parked:** every other phase-1 task. Its claim directory contains a `HOLD` file.
- **Next:** `tasks/fd002.txt` (22 runs):
  1. clean: `eig_health` and `eig_both` r0–2, then `eig_exo` r2
  2. paper: all 5 arms × 3 runs
- **Handover:** `slurm/resubmit_when_free.sh` runs detached on the login node (log in `slurm/logs/watcher.log`). As each phase-1 job finishes, it un-parks the FD002 tasks and submits a `dk_fd002_*` queue job in its place.

**To run another dataset later:** add its lines to a task file, delete the matching `HOLD` claims in `runs/.claims/`, and submit.

**19 Oct 2026, Koopman ablation on FD002 (does Koopman help at all?).**

`--no_koopman_loss` keeps the same model but drops the Koopman one-step, multi-step and spectral losses. Arm name: `svd_static_nokooploss`. Six runs (clean and paper protocols, 3 seeds each) are in `tasks/fd002.txt`, running as job 4921.

Why we run our own ablation: the original `kepin_ablation.py` cannot answer this question.

- Its "w/o Spectral" and "w/o Multi-step" variants set a weight to 0 but keep auto-weighting on. Auto-weighting ignores those weights, so both variants train full KePIN.
- Its "Baseline FCN" is a different, smaller network, not KePIN without Koopman.
- Its saved results contain no C-MAPSS rows.

Related finding: in KePIN, K is one fixed matrix. The 12 spectral features fed to the RUL head are therefore identical for every engine (`tf.tile` in `kepin_model.call`). Koopman can only affect predictions through its training losses, and that is exactly what this ablation measures.

**19 Oct 2026, FD002 report and Koopman time-to-failure head.**

- **Report:** `reports/fd002/REPORT.md`, also bundled as `reports/fd002_report.zip`. It covers all FD002 arms and the ablation, with paired t-tests. Nothing beats the fixed operator, Koopman off is not significantly different (p ≈ 0.66), and the spectral features are constant across engines.
- **`--head ttf`:** the RUL is the soft number of Koopman steps until the projected health (a linear readout of the state, rolled forward by the static eig operator, horizon 125) crosses 0. No classic RUL head is used, so the Koopman operator *is* the predictor. The health trajectory for each test engine is saved for the XAI plots.
- **Jobs:** 6 runs (clean and paper protocols × 3 seeds) as jobs 4930/4931.
- **Caveat:** an `extra_worker.sh` step is killed when its host job ends (paper TTF seed 2 was cut off at epoch ~97 and rerun as a separate job).
- **Infrastructure fixes:**
  - `os.makedirs` can race on NFS, so `train.py` now tolerates `FileExistsError`.
  - `slurm/extra_worker.sh` starts an extra worker inside a running allocation (`srun --jobid=<job> --overlap`), for when released tasks have no live worker.

**19 Oct 2026, TTF results (FD002).**

| Protocol | Average of 3 runs | Ensemble | Compared with fixed-rule `eig_static` |
|---|---|---|---|
| clean | 16.44 ± 0.46 | 15.86 | worse by 0.65 (p = 0.07; fixed rule: 15.79 / 15.34) |
| paper (2 of 3 runs) | 15.40 | 15.02 | seed 2 rerun as job 4937 |

The XAI check found that the projected health never reaches the threshold for any engine (`reports/fd002/ttf_health_trajectories.png`). Trajectories revert toward a common level. The prediction is therefore driven by the *current* health level (corr(H_0, RUL) = −0.91), not by a forecast crossing.

Cause: a stable operator (|λ| < 1) can only make modes fade, so it cannot represent damage that keeps accumulating. The planned fix is a wear-accumulator mode (λ = 1 plus a learned non-negative wear rate), so that projected health actually climbs to the threshold.

**19 Oct 2026, wear-accumulator head (`--head wear`).**

- **Model:** projected health = Koopman transient + `v·k`, where `v = softplus(w·z_T + b) ≥ 0` is a per-engine wear rate (an integrator mode with eigenvalue 1). The crossing is measured in flights (`H_k / v`, τ = 2), so survival only drops near the actual crossing. RUL ≈ the step where projected health reaches 0.
- **Smoke test:** 91% of engines cross the threshold, against 0% with plain TTF.
- **Jobs:** 6 runs (clean and paper × 3 seeds) as job 4940, 3 workers per GPU.

**19 Oct 2026, wear head results (FD002, complete).**

| Protocol | Wear: average ± std | Wear: ensemble | Original KePIN (`svd_static`) | Paired test vs original |
|---|---|---|---|---|
| clean | 16.18 ± 0.13 | 15.56 | 15.98 / 15.40 | +0.20, p = 0.59 |
| paper | 15.38 ± 0.37 | 14.87 | 15.40 / 15.00 (paper: 15.08) | −0.02, p = 0.68 |

- **Accuracy:** statistically tied with original KePIN.
- **Explanations work:** 92–100% of engines cross the threshold. The engines that don't cross are healthy (median true RUL 125, minimum 98). The soft RUL matches the hard crossing step within about 2–4 flights (median).
- **The wear rate is meaningful:** corr(wear rate, true RUL) is −0.90 to −0.93 in every run.
- **Figure:** `reports/fd002/wear_explanations.png`.
- **Infrastructure:** 3 workers per GPU ran out of GPU memory on P100. `train.py` now predicts in chunks of 1024; use at most 2 workers per GPU.

**19 Oct 2026, explainability of the wear model (`analyze_wear.py`, FD002 clean runs 0–2).** Outputs are in `reports/fd002_wear_xai/`.

| Check | Result |
|---|---|
| Life trajectories (52 held-out engines) | Wear rate rises over life in 100% of engines (median Spearman +0.92). Predicted RUL falls in 100% of engines (median −0.92). |
| Shortcut test (out-of-fold R²) | Wear rate: R² from RUL 0.85–0.87, from condition mix −0.12 to −0.13. Current health: R² from RUL 0.69–0.89, from condition mix about −0.13. This is the opposite of Signal A (RUL 0.02–0.04, conditions 0.85–0.88). |
| Attribution (Integrated Gradients) | Top drivers: s11, s1, s15, s4, s2, s17. The 3 settings together account for 7.6%. s1 (fan inlet temperature) depends on the operating condition, so its role is likely to put the other sensors in context for each condition. |

**19 Oct 2026, generalisation runs (`tasks/other_datasets.txt`, jobs 5018/5019; reruns as job 5039).**

Ensemble RMSE, paper protocol, all 3 seeds:

| Dataset | Our KePIN (`svd_static`) | Wear model | Paper's KePIN | Paired p (averages) |
|---|---|---|---|---|
| FD001 | 12.92 | 13.43 | 12.92 | 0.62 |
| FD002 | 15.00 | 14.87 | 15.08 | 0.68 |
| FD003 | 11.56 | 11.43 | 11.42 | 0.99 |
| FD004 | 17.11 | 16.98 | 17.04 | 0.48 |

- **Reproduction:** our KePIN matches the paper within 0.15 on all four datasets.
- **Wear model:** no significant difference anywhere. FD001 includes one bad seed (+1.41).
- **Clean protocol so far:** FD002 15.40 vs 15.56 and FD004 17.84 vs 17.90 (ensembles). The 4 missing FD001/FD003 runs ran out of GPU memory in the final prediction step and are being rerun. Fix: prediction chunks reduced from 1024 to 256.
- **Explanations hold on every dataset:** threshold crossing 85–100%, corr(wear rate, RUL) between −0.88 and −0.95.

**19 Oct 2026, complete Koopman ablation of KePIN (paper protocol only, per user).**

- `--koopman none` removes the Koopman layer, the 12 spectral features into the head and the Koopman losses. Same encoder and head; 1.14M parameters against 1.16M. Arm name: `none_static`.
- 12 runs (FD001–FD004 × 3 seeds) in `tasks/no_koopman.txt`, jobs 5044/5045. Compared with `svd_static` (paper protocol, same seeds), ensemble RMSE.
- From now on the user wants **only the paper protocol and ensemble RMSE**. The 4 unfinished clean-protocol reruns (FD001 svd r1/r2, FD001 wear r2, FD003 wear r0) were cancelled and are parked with HOLD.
- An NFS folder race still hit 3 runs even with retries. Fix: `prepare_dirs.py` pre-creates the output folders from the login node before submitting.

**20 Oct 2026, results of the complete Koopman ablation (paper protocol, ensemble of 3 runs).**

| Dataset | KePIN with Koopman (`svd_static`) | KePIN without Koopman (`none_static`) | Difference | Paired p (per-run RMSE) | Wear model | Paper |
|---|---|---|---|---|---|---|
| FD001 | 12.92 | 13.60 | +0.68 | 0.21 | 13.43 | 12.92 |
| FD002 | 15.00 | 14.73 | −0.27 | 0.39 | 14.87 | 15.08 |
| FD003 | 11.56 | 11.47 | −0.09 | 0.80 | 11.43 | 11.42 |
| FD004 | 17.11 | 17.23 | +0.12 | 0.22 | 16.98 | 17.04 |

None of the differences is significant. FD001 is the only dataset where all three seeds are worse without Koopman (+0.11, +0.42, +1.14), which is a weak hint that Koopman helps on the single-condition data.

**20 Oct 2026, future-trained Koopman in original KePIN.** The wear model is on hold, per the user.

- **`--future_loss W`** (`future_trainer.py`): each training step also samples 64 pairs of (window at flight t, the same engine's window at t + h), h = 5..60. It adds W · ‖K^h z_T(t) − stopgrad(z_T(t+h))‖². Everything else is the original KePIN recipe. Arm names: `svd_static_future1` and `svd_static_future10`.
- **Checks:** the pairs are verified to come from the same engine exactly h flights apart. The new loss is about 1.0 at the start, the same scale as KePIN's one-step and multi-step Koopman losses.
- **Jobs:** 12 runs (FD002 and FD001, paper protocol, 3 seeds, W = 1 and W = 10) in `tasks/future_koopman.txt`, jobs 5070/5071.

**20 Oct 2026, future-trained Koopman results: negative.** Paper protocol, ensemble of 3 runs.

| Dataset | KePIN | KePIN without Koopman | KePIN + future loss W=1 | KePIN + future loss W=10 |
|---|---|---|---|---|
| FD001 | **12.92** | 13.60 | 14.02 (+1.10, worse in all seeds, p = 0.05) | 14.37 (+1.45, p = 0.16) |
| FD002 | 15.00 | **14.73** | 15.79 (+0.79, p = 0.08) | 16.31 (+1.31, worse in all seeds, p = 0.03) |

Why it fails:

- **Eigenvalue collapse.** Plain KePIN keeps several slow modes (top |λ| ≈ 0.99, 0.99, 0.99). With the future loss, only one stays slow (≈ 0.99); the others collapse to about 0.40–0.66 at W = 1.
- **Information loss.** To be predictable 5–60 flights ahead by a stable, fading linear rule, the shared latent is squeezed into about one slow direction and loses the detail the RUL head needs.
- **Same limitation as the TTF head.** A stable operator cannot represent degradation that grows.

**20 Oct 2026, subspace future loss with a growth term (`--future_loss 1 --future_subspace 8`).** The loss touches only 16 dedicated latent dims: 8 state and 8 rate, the rate scaled by 0.02. It has its own diagonal rule with a growth term: x(t+h) = ρ^h x + r(1 − ρ^h)/(1 − ρ). ρ is a trainable model weight, initialised at 0.95. Initial loss ≈ 1.0. 6 runs (FD002 and FD001, paper protocol, 3 seeds), jobs 5081/5082.

**20 Oct 2026, subspace future loss results: still worse than KePIN.** Paper protocol, ensemble of 3 runs.

| Dataset | KePIN | Without Koopman | Future loss (full latent) | Future loss (subspace + growth) |
|---|---|---|---|---|
| FD001 | **12.92** | 13.60 | 14.02 | 13.67 (+0.75; all seeds worse, p = 0.09) |
| FD002 | 15.00 | **14.73** | 15.79 | 15.57 (+0.56; all seeds worse, p = 0.08) |

- **The main operator no longer collapses:** top |λ| stays at 0.99, 0.99, 0.98 (FD002 run 0).
- **Accuracy is still lost.** Any extra future objective on the shared encoder costs RUL accuracy.
- **Conclusion:** auxiliary Koopman training signals do not improve KePIN on C-MAPSS. The accuracy route is exhausted.

**20 Oct 2026, condition-aware Koopman KePIN (`--cond_norm`, `--koop_control`).**

- **`cond_norm`:** a learned per-condition scale and offset on the raw (unsmoothed) inputs, initialised to per-condition standardisation (std floor 0.1). KePIN's EMA smoothing moves inside the model as a fixed matrix, so it runs after this step.
- **`koop_control`:** the latent splits as z = s + C·u. KePIN's Koopman losses and RUL head use s, so z_{t+1} = K z_t + C u_{t+1} − K C u_t.
- **Checks:** the in-model EMA matches KePIN preprocessing (max error 4e-7). With neutral settings the predictions equal KePIN's (6e-6).
- **Caveat:** the noise augmentation is now applied before EMA, so it is slightly weaker than in KePIN.
- **Jobs:** 12 runs (FD004 and FD002 × {cn, cn+ctrl} × 3 seeds, paper protocol) in `tasks/condition_aware.txt`, jobs 5107/5108.

**21 Oct 2026, condition-aware Koopman KePIN results: first real improvement.** Paper protocol, ensemble of 3 runs.

| Dataset | KePIN | Paper | + cond_norm | + cond_norm + koop_control |
|---|---|---|---|---|
| FD002 | 15.00 | 15.08 | 14.13 (−0.88, p = 0.052) | **13.87** (−1.13; every seed better, p = 0.014) |
| FD004 | 17.11 | 17.04 | **15.32** (−1.79; every seed better, p = 0.003) | 15.41 (−1.70, p = 0.004) |

- **NASA score (ensemble), FD002:** 869 → 859 (cn) → 825 (cn+ctrl).
- **NASA score (ensemble), FD004:** 1265 → 1051 (cn) → 1043 (cn+ctrl).
- **Where the gain comes from:** mostly the condition-aware input layer. Koopman-with-control adds a further −0.26 on FD002 and +0.09 on FD004. Neither difference between cn and cn+ctrl is significant.

**21 Oct 2026, report update and follow-ups.**

- **Report:** `FULL_REPORT/REPORT.md` (and the zip) now covers Steps 13–18, an overview figure (`reports/make_overview_figure.py`) and the architecture changes.
- **Single-condition datasets:** `train.py` now slices the condition one-hot to `n_conditions` columns, so cond_norm and koop_control run on FD001/FD003 (1 condition). Results on FD002 are unchanged.
- **Jobs:** 12 runs (FD003 and FD001 × {cn, cn+ctrl} × 3 seeds, paper protocol) in `tasks/condition_aware_single.txt`, as 1-GPU jobs 5145/5146. Only 1 GPU was free on the cluster.
- **`analyze_condition_state.py`:** probes the pooled hidden state (KePIN z, cn z, ctrl z, ctrl wear state s) for condition-mix and RUL information, using ridge and random forest, out-of-fold by engine.

**21 Oct 2026, bug fix in `slurm/queue.slurm`.** The script used `nvidia-smi -L` to count GPUs. On these nodes that lists **every** GPU, including GPUs allocated to other users. So 1-GPU jobs (5109 on gpu-A100-01 on 20 Oct, and 5145 for about 2 minutes on 21 Oct) also started workers on a GPU they did not own. The script now uses only the GPUs SLURM allocates (`CUDA_VISIBLE_DEVICES`/`SLURM_JOB_GPUS`) and prints them as "Allocated GPUs". Job 5145 was cancelled and resubmitted as 5147/5148. The FD002/FD004 results from job 5109 are valid (they ran correctly), but that job did use a second GPU it was not allocated.

**21 Oct 2026, probe results (`analyze_condition_state.py`, FD002, paper runs 0–2, mean of 3).** Out-of-fold R²:

| Hidden state read by the head | Condition mix (linear) | Condition mix (forest) | RUL (linear) | RUL (forest) |
|---|---|---|---|---|
| KePIN z | 0.263 | 0.055 | 0.887 | 0.883 |
| cond_norm z | 0.004 | −0.007 | 0.893 | 0.889 |
| cond_norm+ctrl z (before −C·u) | 0.007 | −0.005 | 0.890 | 0.886 |
| cond_norm+ctrl **s = z − C·u** | **0.968** | 0.022 | 0.891 | 0.886 |

- **The condition-aware input layer removes the flight-condition information** from KePIN's latent (linear R² 0.26 → 0.00) while keeping all the RUL information.
- **The "wear state" s is not condition-free:** subtracting the learned C·u writes the condition mix back as an exact linear signal (R² 0.97). koop_control does not do what it was designed to do. The gain belongs to cond_norm.

**21 Oct 2026, teaching PDF.** `../KePIN_Project_Explained.pdf` (21 pages) is an easy-language guide covering 12 topics plus a glossary, with diagrams, real code excerpts and figures.

- `reports/teaching/make_figures.py` generates the diagrams and new figures.
- `reports/teaching/build_pdf.py` assembles the PDF. It needs reportlab, which was installed into the scratchpad, not the kepin env: `PYTHONPATH=<dir with reportlab> python build_pdf.py`.

**21 Oct 2026, condition-aware KePIN on all four datasets.** Paper protocol, ensemble of 3 runs (jobs 5147/5148).

| Dataset | KePIN | Paper | cn | cn+ctrl |
|---|---|---|---|---|
| FD001 | 12.92 | 12.92 | 13.41 (+0.50, p = 0.48) | **12.85** (−0.06, p = 0.63) |
| FD002 | 15.00 | 15.08 | 14.13 (−0.88) | **13.87** (−1.13) |
| FD003 | 11.56 | 11.42 | 11.17 (−0.39, p = 0.55) | **11.07** (−0.49, p = 0.45) |
| FD004 | 17.11 | 17.04 | **15.32** (−1.79) | 15.41 (−1.70) |

- **Multi-condition data:** big, significant gains.
- **Single-condition data:** no significant change, as expected.
- **cn+ctrl** is at or below the paper on all four datasets. The probe result still stands: its "wear state" is not condition-free.

**21 Oct 2026, feature replacement experiment saved as a script.** `analyze_spectral_freeze.py` (results in `reports/spectral_freeze/summary.json`). The experiment was first run inline (19 Oct); the script reproduces those numbers exactly.

- **Constant:** each model's average of the 12 spectral features over the 253 FD002 test engines. It is derived, not random.
- **Correction:** dynamic B's features are effectively constant across engines (spread ≈ 4e-7). Only dynamic A's vary (0.001–0.005), and freezing them changes predictions by ≤ 0.09 flights.
- **Updates:** the teaching PDF (section 2, with a Q&A box) and `FULL_REPORT/REPORT.md` (Step 9) are updated.

**21 Oct 2026, feature replacement on the Signal A (RC-KePIN) models.** Script: `RCKePIN_Signal_A/analyze_spectral_freeze_signal_a.py`; results in `RCKePIN_Signal_A/results/spectral_freeze_signal_a/summary.json`. All loaded models reproduce their reported RMSE (±0.003).

- **The features vary a lot between engines:** spread 0.02–0.13, against ≤ 0.005 for our Dynamic A. Δs spread is 0.14–0.39.
- **Freezing them moves individual predictions:** mean |change| 0.35–1.15 flights, max 2.8–9.4 flights. The predictor does use them.
- **But accuracy does not suffer.** Ensemble RMSE with real → frozen features:

| Dataset | Real features | Frozen features | Change |
|---|---|---|---|
| FD001 | 13.40 | 13.30 | −0.10 |
| FD002 | 14.94 | 14.89 | −0.06 |
| FD003 | 11.21 | 11.20 | −0.01 |
| FD004 | 16.81 | 16.82 | +0.01 |

Per run, freezing makes no difference or improves slightly in 9 of 12 runs. The per-engine Koopman information is used but adds no accuracy, consistent with Signal A encoding operating-condition switching rather than wear.

**24 Oct 2026, why the condition-aware input works (`analyze_condnorm.py`, results in `reports/condnorm_analysis/`).**

| Measure (mean over sensors) | FD002 global → per-condition | FD004 global → per-condition |
|---|---|---|
| Share of a sensor's variance due to the flying condition (η²) | 0.992 → 0.003 | 0.993 → 0.001 |
| \|corr(sensor, RUL)\| | 0.017 → 0.345 (best sensor 0.09 → 0.77) | 0.023 → 0.295 (best 0.11 → 0.77) |
| \|corr\| after EMA: KePIN order → new order | 0.059 → 0.459 | 0.079 → 0.336 |

- **Theory:** for random condition switching, EMA leaves α/(2−α) ≈ 3.3% of the condition variance. That residual still dwarfs the wear signal, so normalising before smoothing matters.
- **Parameters:** the learned γ and β barely move from their per-condition-standardisation initialisation. Median relative scale change is 0.4–0.7% (90th percentile ≤ 2.7%), and correlation with the initial values is ≥ 0.987. The gain therefore comes from per-condition normalisation applied before EMA; learning adds little. This should be tested with a frozen-γ/β ablation.

**4 Nov 2026, regime-adaptive Koopman (thesis focus: the operator adapts to the engine's health stage).** The wear model is on hold, per the user.

Goal: healthy engine → the Koopman operator forgets slowly; near failure → it forgets fast. Signal A tried this but its regime signal measured the operating condition (R² 0.97 from the condition mix), and the old `eig_health` health index collapsed to a constant (spread ≈ 4e-7). The new model constrains the regime variable on three fronts:

| Piece | What it does | Where |
|---|---|---|
| Condition-free input | `cond_norm` + in-model EMA in front of the eig operator, so the operating condition cannot drive h | `dk_model._condition_aware_latent` |
| Ordered map h → decay | `s_k(h) = s_k − gap_k·(1 − h)`, `gap_k = 2·sigmoid(c_k) ∈ (0, 2)`, init 0.036. h = 1 healthy operator, h = 0 failure operator; lower health can only mean faster decay. A learned gap ≈ 0 would mean the data does not support the idea | `eigen_koopman.health_shift`, `--health_map ordered` |
| Regime reaches the prediction | the head also reads the current state carried 10 flights ahead by K(h) | `eigen_koopman.propagate`, `--head_state 10` |
| h tied to degradation | pairs (window at t, same engine at t+dt, dt = 10..60); `mono`: relu(h(t+dt) − h(t) + 0.5·dt/life), uses only time order; `sup`: h ≈ RUL/125 (reference) | `regime_trainer.py`, `--regime_loss` |

- **Checks:** reload of saved weights is exact; with a learned gap the state 10 flights ahead is ~100× smaller at h = 0.5 and ~10⁴× smaller at h = 0 than at h = 1. CPU smoke test (2000 windows, 3 epochs): mono already gives corr(h, RUL) = 0.82 on the test engines (unconstrained: −0.03).
- **Cost:** pairs (2 × 32 windows) go through the encoder in one extra call per step; smoke run 23 s/epoch on CPU subset. Expected ~30–45 min per FD002 run on a P100, like `svd_static_cn`.
- **Runs:** `tasks/regime_fd002.txt`, 9 runs (FD002, paper protocol, 3 seeds × {unconstrained, mono, sup}), jobs 5948/5949 (1 GPU each; cluster busy). Baseline: `svd_static_cn` (ensemble 14.13). `train.py` saves `health_corr_rul`, `health_std`, `decay_gap_*` and the per-step regime loss.

**4 Nov 2026, regime-adaptive Koopman results (FD002, paper protocol, 3 seeds; jobs 5948/5949, ~2.7 h wall-clock).**

| Arm | Ensemble RMSE | NASA score | Runs (mean ± std) | vs cond_norm KePIN | corr(h, RUL) | spread of h |
|---|---|---|---|---|---|---|
| cond_norm KePIN (`svd_static_cn`, baseline) | **14.13** | 859 | 14.53 ± 0.31 | | | |
| regime, h unconstrained | 14.26 | 859 | 14.56 ± 0.19 | +0.03, p = 0.82 | — (h stuck at 1.0) | 0.000 |
| regime, h monotone over life (`mono`) | 14.16 | 860 | 14.51 ± 0.08 | −0.02, p = 0.94 | 0.87–0.90 | 0.28–0.30 |
| regime, h supervised (`sup`) | **14.13** | 858 | 14.50 ± 0.25 | −0.03, p = 0.92 | 0.94 | 0.36–0.37 |

- **Accuracy:** all three arms tie cond_norm KePIN.
- **Health index:** without a constraint, h collapses again (saturates at 1). The monotone loss, which never sees RUL labels, gives a real, condition-free health index (corr 0.87–0.90 on test engines, spread over the full range).
- **Regime-dependent decay is barely used:** the learned gap only grew from 0.036 to 0.04–0.17 (decay-logit units). For the slowest modes ρ changes by ≤ 0.004 between a healthy and a failing engine; over 10 flights a failing engine's state is only 0–10% smaller. The model, given a working health index and the freedom to forget faster near failure, chooses almost no difference.
- **Cost:** 46–49 s/epoch against 26–35 for the baseline (+40–50%, more than the estimated 15%).
- **Open caveat:** the gap starts near 0, which favours "no effect". A run starting from a large gap would show whether the data actively rejects regime-dependent decay or is indifferent to it.

**4 Nov 2026, what drives the health index h (`analyze_regime_h.py`, results in `reports/regime_h/`).** FD002 paper runs 0–2 of the mono and sup arms.

| Check | mono (no RUL labels) | sup |
|---|---|---|
| R² of h from the condition mix (linear / forest) | −0.00 / −0.04 | −0.01 / −0.04 |
| R² of h from RUL (linear / forest) | 0.74–0.79 / 0.74–0.80 | 0.88 / 0.90 |
| h falls over life (per-engine Spearman, all 260 training engines) | −0.97, 100% of engines | −0.94, 100% |
| Top Integrated-Gradients drivers (share of \|attribution\|) | s11 17–21%, s15 12–15%, s4 10–12%, then s14, s17, s3, s9, s2 | same order |
| Operating settings (3 columns) | 1.4–4.4% | 1.9–2.6% |

- IG baseline = each flight's condition mean (the input the condition-normalisation maps to zero); completeness error 0.002 against a mean h change of 0.25.
- Model-free check agrees: the per-condition-normalised sensors most correlated with RUL are s11 (−0.77), s4 (−0.74), s15 (−0.72), s17, s2, s3.
- Near failure every top sensor pushes h down. Attribution is spread over the whole window (last 5 flights ≈ 35%), so h reads the level and trend over 30 flights.
- In the capped part of the RUL label (true RUL ≥ 125, "healthy" by the label), h still ranges about 0.2–0.9 and engines start life at different h (0.6–0.9): it separates engines the label treats as identical, consistent with C-MAPSS engines starting with different initial wear.

**4 Nov 2026, Signal A with per-condition scaling (`RCKePIN_Signal_A/analyze_signal_a_condnorm.py`).** No model trained; 46,219 FD002 training windows; out-of-fold R² (folds = engines), linear / forest.

| Signal A variant | R² RUL | R² condition mix |
|---|---|---|
| original (global scaling, 4 channel-averaged moments) | 0.04 / 0.19 | 0.50 / 0.69 |
| per-condition scaling, same 4 moments | 0.55 / 0.63 | 0.04 / 0.02 |
| per-condition, 4 moments per sensor (84) | 0.72 / 0.74 | 0.68 / 0.76 |
| per-condition, per-sensor mean + slope (42) | 0.82 / 0.85 | −0.01 / 0.22 |

With per-condition scaling the averaged mean moment correlates −0.72 with RUL (−0.09 before); skewness and kurtosis stay ≈ 0. Per-sensor variance/kurtosis bring the condition back (near-constant sensors differ by condition).

### 2026-11-04 — Signal-A-driven regime Koopman (queued: jobs 5955, 5956)
Idea: drive the decay of K(h) with a readable, hand-made regime signal instead of a learned MLP.
`--health_source signal_a`: inside the model, each sensor is z-scored with its operating condition's
exact mean/std (fixed, from training data), then [mean, var, skew, kurt] over the 30-flight window,
averaged over the non-constant sensors -> A (4 numbers); h = sigmoid(w . standardise(A) + b), only
w (4) and b learned (reported as `sigA_w`, `sigA_b`). Check: in-model A gives corr(mean, RUL) -0.71,
R^2(RUL) 0.50, R^2(condition mix) 0.10 (offline analysis: -0.72 / 0.55 / 0.04).
`--gap_init c`: initial decay-gap logit (gap = 2 sigmoid(c)); default -4 -> 0.036, `0` -> 1.0, i.e.
a failing engine's decay logit starts 1.0 lower than a healthy one's. Does a large gap survive training?
Tasks: tasks/sigA_regime_fd002.txt — sigA+gap1.0 / sigA+gap0.036 / learned h+gap1.0, FD002 paper, 3 seeds, mono loss.
Partial results (7/9 runs, 2026-11-05; ensemble over finished runs):
- learned h + gap 1.0 (2 runs): ens 14.23. Gap SURVIVES (mean 1.08-1.15), h corr RUL 0.93. 10 flights ahead,
  a near-failure engine (RUL<=20) keeps 1-13% (median over modes) of what a healthy one (RUL>100) keeps;
  with the small gap init it was 56-84%. -> clearly regime-dependent shrinking at no RMSE cost.
- Signal A + gap 1.0 (3 runs): ens 13.98, gap survives (~1.0), BUT h is not a health index: corr RUL
  -0.12..-0.48, h drop 0.006, w_var ~ +0.5 (wrong direction), b ~ w_var. Mono loss not met with 5 learnable params.
- Signal A + small gap (2 runs): ens 14.24, same h failure.

### 2026-11-05 — Three ways to make the Koopman operator adaptive (queued: job 5960 + one more)
Diagnosis first (FD002 paper, mean of single runs): no Koopman 15.20, KePIN Koopman 15.40, +cond_norm 14.53,
+control term 14.20 +- 0.07. The RUL head mostly reads encoder features, so K hardly moves RMSE. Fix: the
head reads the state carried 1/5/10/25/50 flights ahead by the engine's own operator (`--forecast_head`).
Now allowed with the control term (`--koop_control`; z = s + C u, condition-free state). New `--adapt`:
- `gate`    health-gated decay (as before; gap init 1.0).
- `mixture` 3 regime operators (shared basis, own decay/rotation; decay offsets start 0/-0.5/-1), blended
            by soft regime probabilities; h = pi . [1, .5, 0] so the mono loss orders healthy -> near failure.
- `engine`  per-engine eigenvalues: 1-D DMD fit of each mode on the window, blended with the fleet eigenvalue
            (learned alpha, starts 0.5); Koopman losses keep the fleet operator.
Logged per run: `keep10_nearfail_over_healthy` (how much more a near-failure engine's state fades in 10
flights), regime shares / mean RUL per regime (mixture), alpha (engine). Tasks: tasks/adaptive3_fd002.txt.
Smoke tests pass for all four; a bug (regime offsets reset to 0 by assign() in build) was fixed before queuing.
Results (2026-11-05, FD002 paper, 3 seeds; single-run mean / ensemble; bootstrap vs svd_static_cn_ctrl):
| arm | single | ens | note |
| svd_static_cn_ctrl (static K + control) | 14.20 +- 0.07 | 13.87 | still best |
| gate + ctrl, hs10 | 14.45 +- 0.11 | 14.17 | h corr 0.90-0.94, gap 1.06-1.32, near-fail keeps 0.1-8% of healthy in 10 flights |
| gate + ctrl, forecast head | 14.58 +- 0.04 | 14.28 | forecast head does not help |
| mixture + ctrl, forecast head | 14.49 +- 0.11 | 14.17 | collapses to 2 regimes in all 3 runs: "healthy" (mean RUL 100-106, 55-64% of test engines) and "near failure" (mean RUL 24-33), the latter with ~0.5-0.8 lower decay logit; middle regime unused |
| engine (per-engine eigenvalues) + ctrl, fc | 14.39 +- 0.10 | 13.97 | alpha stayed 0.50 (not learned). Near-failure engines fade SLOWER: in 55-59/64 modes rho rises as RUL falls (median Spearman(rho, RUL) -0.50..-0.60); mean rho 0.44-0.50 (RUL>100) -> 0.54-0.66 (RUL<=20) |
No adaptive arm beats the static K + control (all CIs include 0, P(better) 0.04-0.34). The data-driven per-engine
fit contradicts the imposed "fade faster near failure": the engine's own latent dynamics become MORE persistent
near failure (eigenvalues move toward the unit circle) — consistent with critical slowing down. To check: is it
only the within-window trend? (detrend before the fit).
Drift check (analyze_slowing.py -> reports/slowing/summary.json, 2026-11-05): the "slower fading near failure"
is mostly the within-window drift. Mean |lam| RUL>100 vs RUL<=20 on test engines: raw 0.54-0.60 -> 0.83-0.89;
after removing mean+slope per window 0.36-0.50 -> 0.40-0.66, Spearman(|lam|, RUL) -0.29/-0.38/-0.06, and on
training windows over life it rises in 73%/64%/15% of engines (run 2 reverses). Model-free: detrended lag-1
autocorrelation of the per-condition-normalised sensors is ~ -0.07 at every RUL (raw: 0.30 -> 0.65 only because of
the trend); detrended variance flat. => No critical slowing down in FD002: around the degradation trend the
sensor noise is white. Not a claim we can make.
Knockout test (analyze_knockout.py, 2026-11-05): switch the adaptive parts off at test time, no retraining.
Gate models: forcing every engine to h=1 (healthy operator) or h=0 (failing operator) changes RMSE by 0.00;
zeroing the forecast states by <=0.02. Mixture: forcing one regime for all engines +-0.07. Per-engine K: fleet K for
all +0.02. => The RUL head ignores the Koopman branch: K(h)^k z_T is a function of z_seq, which the head already
reads through pooled encoder features + skip. An adaptive K can only change RMSE if the prediction is forced
through K (e.g. RUL = time until K's forecast reaches a failure level), not as extra head features.

### 2026-11-05 — Signal A in the wear model (job 5979 + 1 more)
Knockout showed the KePIN-style head ignores K. Literature (Koopa NeurIPS'23, KNF ICLR'23, Garmaev & Fink RESS'24,
EKIDM) uses K as THE predictor. Our wear model already is that design (RUL = flights until K's projected health
H_k = H_koopman(k) + v k reaches 0). New: wear head on the condition-aware path (`--head wear --cond_norm`,
eig static), and `--health_source signal_a` sets the wear rate from condition-normalised Signal A only:
v = softplus(10 (w . standardise(A) + b)), 4 moments, 5 learned numbers (x10 learning-rate multiplier; with the
shared rate they barely moved in the earlier Signal A run). Current health H_0 still comes from the latent state.
Tasks: tasks/wear_cn_sigA_fd002.txt (FD002 paper, 3 seeds, 2 arms). Logged: wear_rate_corr_rul, sigA_wear_w/b.
Results (2026-11-05, FD002 paper, 3 seeds):
| arm | single | ens | NASA score | note |
| eig_static_wear (old, no cn) | 15.38 +- 0.30 | 14.87 | 921 | |
| svd_static_cn_ctrl (previous best) | 14.20 +- 0.07 | 13.87 | 825 | |
| **eig_static_wear_cn** | 14.19 +- 0.28 | **13.68** | **748** | NEW BEST; vs old wear CI [-2.32, -0.10]; vs 13.87 CI [-0.70, +0.36]; wear corr RUL -0.91..-0.93; 100% cross |
| eig_static_wear_cn_sigA | 33.72 +- 6.00 | 33.33 | 5624 | BUG, see below |
BUG (Signal A, also affects the earlier regime sigA runs): Signal A used the EXACT per-condition std. Some sensors
have std 1e-4..1e-3 within a condition, and training adds input noise (~0.009, augment + per-batch), up to 100x
their spread: in training the variance feature moved by ~5,700 std units and corr(mean moment, RUL) fell from
-0.71 to -0.12; at test (no noise) the features were clean -> train/test mismatch. Fix: Signal A uses cond_norm's
floored std (0.1); under noise corr stays mean -0.79, var -0.70. Buggy runs moved to runs/paper/FD002/_bug_noise_*.
Re-run: tasks/wear_sigA_fix_fd002.txt (job 5987).

### 2026-11-05 — Static vs adaptive K, same baseline (jobs 5987, 5988)
User: cond_norm and the wear model are known; the contribution to test is static vs adaptive K on the SAME
baseline. Baseline = eig_static_wear_cn (ens 13.68), where K is the only path to the RUL. Change: the wear head's
health forecast now uses each engine's operator (`health_rollout(..., log_rho, theta)`) when `--mode health`;
static mode is unchanged. Arms (FD002 paper, 3 seeds; appended to tasks/wear_sigA_fix_fd002.txt):
eig_health_wear_cn_ord_gap0_mono (gate), eig_health_wear_cn_mix_mono (mixture), eig_health_wear_cn_eng (local K).
Results (2026-11-05 evening, FD002 paper, 3 seeds; bootstrap vs static):
| arm | single | ens | score | vs static |
| static K (eig_static_wear_cn) | 14.19 +- 0.28 | **13.68** | **748** | - |
| gate | 14.49 +- 0.07 | 14.17 | 847 | CI [-0.17, +1.12] |
| mixture | 14.34 +- 0.15 | 14.04 | 838 | CI [-0.20, +0.92] |
| local K | 14.53 +- 0.38 | 14.17 | 798 | CI [-0.01, +1.02] |
| Signal A wear rate (fixed bug) | 14.73 +- 0.05 | 14.38 | 783 | CI [+0.10, +1.35] (worse) |
Mixture: 2 regimes in 2 runs (healthy mean RUL 101-104 / near failure 26-32), 3 in one (105 / 65 / 29), decay
offsets ordered (faster near failure). Gate: one run's h collapsed (corr 0.02). Signal A wear: v uses only the mean
moment (w_mean +0.15..+0.17, others ~0), corr(v, RUL) -0.77 vs -0.92 learned.
Knockout (analyze_knockout_wear.py): K NOW MATTERS — freezing the fading modes costs +1.5 (gate), +3.0 (mixture),
+2.3 (local K). But adaptivity does not: forcing the HEALTHY operator on every engine changes RMSE by -0.01 / -0.00,
fleet operator for local K +0.15; only forcing the failing operator on everyone hurts (+0.5..+0.7).
=> On FD002 a single static K is enough once K is the predictor; three adaptive operators, two architectures,
same answer. Open: FD004 (two fault modes), where different dynamics per regime are plausible.

### 2026-11-05 — 2x2 study: linear/nonlinear x static/dynamic latent dynamics (jobs 6005, 6006)
Run on the condition-aware wear model (K is the only path to the RUL; in the KePIN head any dynamics are ignored).
Adaptive K is itself a structured nonlinear system (z(k+1) = K(h(z(k))) z(k)); the earlier gate held h fixed for the
whole forecast. `--dynamics`:
- `nl_static`  z~(k+1) = K z~(k) + tanh(MLP(z~(k))), output zero-initialised (starts as static K). Same for all engines.
- `nl_dynamic` z~(k+1) = K(h(z~(k))) z~(k): a state-level health net re-reads h from the CURRENT state at every one of
  the 125 forecast steps; ordered gap (lower h -> faster decay), mono regime loss on h of the window's last state.
Both roll out flight by flight; the one-step Koopman loss uses the same nonlinear step. ~4-5x slower per epoch (CPU smoke).
Logged: forecast_h_start/at50 (does h fall along the forecast?), forecast_h_corr_rul. Tasks: tasks/nl2x2_fd002.txt.
| | static | dynamic |
| linear | 13.68 | 14.04-14.17 |
| nonlinear | pending | pending |
Latent curvature check (analyze_latent_curvature.py, 2026-11-05): latent "health" = out-of-fold ridge projection of
z_T predicting remaining flights; per engine, slope over the last 50 flights / slope 100-150 flights before failure.
Condition-normalised sensor index: ~6x (100% of engines). KePIN-head encoder (svd_static_cn): 1.8-1.9x (75-79%).
Wear-model encoder (eig_static_wear_cn): 1.5-1.8x (72-77%). => encoders remove most of the acceleration, and the
same amount whether or not the RUL goes through a linear K: the straightening comes from training on the
(linear, capped) RUL label, not from K. Caveat: a linear readout fitted to a linear-in-time target picks the
straightest direction, so these ratios are lower bounds on the remaining curvature.
Also fixed: _condition_aware_call passed h= to the svd Koopman layer (broke svd+cond_norm since the Signal A change;
no runs used that path in between).

### 2026-11-05 — Frozen self-supervised encoder: adaptive K on a latent that keeps the nonlinearity (jobs 6009, 6010)
2x2 jobs 6005/6006 cancelled (user): they tested nonlinear dynamics on the RUL-straightened latent, off the question.
Question now: does adaptive K help when the latent still contains the accelerating degradation?
- `pretrain_ssl.py`: same encoder (cond_norm -> EMA -> CNN/BiLSTM/attention), trained only to reconstruct its own
  input window (denoising, noise 0.009 = training noise), 40 epochs, no RUL labels -> runs/ssl/FD002/run<k>_encoder.npz.
- `train.py --frozen_encoder`: loads that encoder (+ cond_norm) and freezes it (BatchNorm in inference mode);
  only K and the wear head train (~17k params static, ~21k nl_dynamic vs 1.7M end-to-end).
- slurm/ssl_then_queue.slurm: pre-trains the 3 encoders in the background, workers start at once; frozen tasks
  wait for their encoder file.
Arms (tasks/frozen_fd002.txt, FD002 paper, 3 seeds): control nl_dynamic end-to-end; frozen: static linear K,
linear gate (h per window), nl_dynamic (h re-read every forecast step). The comparison is static vs adaptive WITHIN
the frozen arms; frozen RMSE will be worse than end-to-end (13.68) since the encoder never sees RUL.
To do when encoders exist: curvature of the SSL latent (expect closer to the sensors' ~6x than RUL-trained ~1.8x).
SSL encoders done (2026-11-05 22:40): reconstruction MSE = 2.0% of target variance (all 3 seeds), ~20 min each.
Validity check (analyze_latent_curvature.py "self-supervised"): late/mid slope ratio 2.5-2.8x, faster late in 87-89%
of engines, linear R2 of remaining flights 0.61-0.62 — vs RUL-trained latents 1.5-1.9x / 72-79% / 0.73-0.76 and
sensors ~6x / 100%. The SSL latent keeps clearly more of the acceleration (lower bound) -> test is valid, runs continue.
Control run0 had crashed (GPU OOM next to 2 SSL processes); released after pre-training for a retry.
Same-method curvature (ridge readout of remaining flights, late/mid slope): encoder input (cond_norm + EMA) 5.0x
(100% of engines), SSL latent 2.8x (89%), RUL-trained latent 1.8x (74%). So the SSL latent keeps only about half.
Layer by layer (analyze_layer_curvature.py -> reports/latent_curvature/layers.json, run 0): the first ReLU does the
largest drop in BOTH encoders: input 5.0x -> projection 5.0x -> +BatchNorm 5.0x -> +ReLU 3.4x. Then the conv blocks:
SSL 3.4 -> 2.5-2.8x, RUL-trained 3.4 -> 2.1x, and RUL training removes a bit more in BiLSTM/attention (-> 1.8x).
LayerNorm/BatchNorm do NOT flatten (earlier guess wrong). Reading: with ReLU features a LINEAR readout (what the
wear model's linear health readout + linear K use) can be made straighter; without activations it cannot.
Added (2026-11-05 23:3x): `--encoder linear` (latent = Dense of cond-norm + EMA sensors, no activation), end-to-end,
static / gate / nl_dynamic x 3 seeds, appended to tasks/frozen_fd002.txt. Extra 2 workers started inside job 6009
(srun --overlap slurm/extra_workers.sh) after the SSL pre-training freed GPU memory.

Results (2026-11-06, FD002 paper; 19/21 runs, job 6009 ended at 03:09 and killed the 2 srun-overlap workers mid-run
-> lin gate run2 and lin nl_dynamic run1 resubmitted as job 6013). Bootstrap vs static K of the SAME encoder:
| encoder (curvature kept) | static K | gate K (h per window) | per-flight h K |
| RUL-trained deep (1.8x) | 13.68 | 14.17 (n.s.) | 13.97 (n.s.), h corr RUL 0.90-0.94 |
| frozen self-supervised (2.8x) | 18.61 | **16.73** CI [-2.83,-0.96] | 18.55 (n.s.), h corr RUL 0.77-0.84 |
| linear (5x) | 19.25 | **16.32** (2 runs) CI [-4.32,-1.55] | 19.25 (2 runs, n.s.), h corr RUL ~0.8 |
Knockout of the gate on frozen/linear: forcing one h for all engines costs +2.5..+7 RMSE (median h) -> the per-window
operator IS used (unlike the deep encoder: ~0). BUT h is not health: R2 of h from RUL ~0, from condition mix ~0,
falls over life in only 44-53% of engines; within-engine std 0.17 (deep gate: h R2 RUL 0.84, falls in 100%).
The per-flight-h arm, whose h does track RUL (~0.8), gives NO gain. So the gain is not "adapting to the health regime";
likely confound: the gate adds a nonlinear MLP of the window ([mean z, z_T]) to an otherwise linear model (linear
encoder / frozen encoder + linear readouts) = extra capacity. Needed control: static K + the same MLP elsewhere.
Also: per-flight h cannot fall along the forecast (0.96 -> 0.97) because wear lives in the v*k term OUTSIDE the latent;
the projected latent only fades, so h re-read from it returns to "healthy". Design flaw for testing that hypothesis.
Linear-encoder arms complete (job 6013 re-ran the 2 killed runs): see table update in the next entry.

### 2026-11-06 — Hypernetwork Koopman on ORIGINAL KePIN (jobs 6031, 6032)
User asked to try hypernetworks (Hyper-LR-PINN, Cho et al., NeurIPS 2023) on the original KePIN, not the wear model.
Mapping: our K = P blockdiag(rho_k R(theta_k)) P^T already is "shared orthogonal basis + diagonal"; the hypernetwork
generates the diagonal per engine. `--mode hyper` (eig operator, original EMA pipeline, KePIN head): context
[mean_t z, z_T] (+ `--hyper_age`: flights flown so far, known at test time, /100) -> MLP(64) -> zero-initialised
ds (2 tanh), dth (0.5 tanh); `--hyper_rank`: g_k = clip(ReLU(1 + dg_k), 0, 1) switches modes off (adaptive rank).
New channel: the eigenvalues now differ per engine, so the KePIN head's spectral features differ per engine (they
were constant before). dk_data now also returns age_train/age_test. Logged: hyper_rho_spread, corr(mean rho, RUL/age),
active modes. Tasks: tasks/hyper_kepin_fd002.txt (eig_hyper, eig_hyper_age, eig_hyper_age_rank x 3 seeds).
Results (2026-11-07, FD002 paper, 3 seeds; original KePIN pipeline):
| arm | single | ens | score |
| svd_static (KePIN) | 15.40 +- 0.24 | 15.00 | 869 |
| eig_static | 15.34 +- 0.19 | 14.91 | 951 |
| eig_hyper (window) | 15.13 +- 0.21 | 14.72 | 854 |
| eig_hyper_age | 14.81 +- 0.13 | 14.39 | 972 |
| eig_hyper_age_rank | 14.43 +- 0.25 | 14.04 | 818 | all 3 seeds (14.07-14.64) below all eig_static seeds (15.08-15.47); ens bootstrap vs eig_static CI [-1.94, +0.16]
Rank arm: 36-41 of 64 modes active, active-mode count not tied to RUL (corr -0.29..0); mean rho corr RUL -0.6..-0.7.
Knockout (analyze_knockout_hyper.py): window-only hyper: same K for all / spectral fixed -> 0.00 (head ignores K, as before).
With age: same K for all +2.9 / +2.4 (rank), spectral features fixed +2.9 / +2.2, age shuffled +5.9 / +7.7.
=> With age as context, the head DOES use the per-engine K (through the spectral features), but what K carries is
mainly the age. The head has no other access to age, so K may just be a conduit. Needed control: static K + age given
directly to the head; if it matches ~14.0-14.4 the gain is the age feature, not the adaptive Koopman.

### 2026-11-09 — Health regime into the hypernetwork (original KePIN; jobs dk_hh1/dk_hh2)
User: keep the conditioned (cond_norm) models aside; hypernetwork made K contribute in original KePIN (with age).
Rule found: K contributes only when its spectrum carries information the head cannot already see (window context: 0.00).
So the health regime must come from OUTSIDE the window: dk_data now builds 42 engine-history features per window
(past flights only; each sensor z-scored with its own condition's training mean/std; drift = mean of last 10 flights
- mean of first 10; rate = slope over last 30 flights), cached in runs/cache/hist_FD002_v1.npz. Linear R2 to capped
RUL 0.81 (age alone 0.49). Model (`--hist_health`): h = sigmoid(MLP(42)) trained label-free (mono regime loss);
hypernetwork context = h only (`hyper_use_window=False`, bottleneck) or [h, age]. Control `--head_ctx`: static K,
[h, age] concatenated to the KePIN head input (hook `_head_extra` in kepin_model.py).
Arms (tasks/hyper_health_fd002.txt, 3 seeds): eig_hyper_hh_mono, eig_hyper_hh_age_rank_mono, eig_static_hh_hctx_mono.

### Results — health regime into the hypernetwork (FD002, paper protocol, 3 seeds)

| Arm | single runs | ens RMSE | ens score |
|---|---|---|---|
| eig_static (reference) | 15.47 / 15.08 / 15.47 | 14.91 | 951 |
| eig_hyper_age_rank (previous best) | 14.64 / 14.07 / 14.58 | 14.04 | 818 |
| A: eig_hyper_hh_mono (K from h only) | 13.31 / 13.65 / 13.19 | 13.09 | 668 |
| B: eig_hyper_hh_age_rank_mono (K from h + age, rank gates) | 11.58 / 11.33 / 11.44 | **11.03** | 543 |
| C: eig_static_hh_hctx_mono (static K, h + age into head) | 11.62 / 11.10 / 11.23 | 11.04 | **465** |

Bootstrap (253 test engines, ensemble RMSE differences): B − C = −0.01 [−0.56, +0.52] (tie);
B − previous best = −3.02 [−4.22, −1.81]; A − static = −1.81 [−3.08, −0.51]; C − static = −3.87 [−5.12, −2.56].

- h is a real health index (label-free, mono loss): corr(h, RUL) 0.92–0.94 in every run.
- A: mean decay rate tracks h almost perfectly (corr −0.94..−0.98). In A, h reaches the prediction ONLY through K, so K genuinely carries health.
- B: K follows h, but the sign flips across seeds (−0.71, +0.90, +1.00); active modes vs h inconsistent (+0.53, +0.46, −0.15). No consistent "modes switch off near failure" story.
- Verdict: the gain comes from the history-health information, not from routing it through K. Static K + h in the head (C) ties B on RMSE and has a better score. K is a working pipe, not an extra benefit.

### Arm A readability + knockout (analyze_hyper_health.py → reports/hyper_health/eig_hyper_hh_mono_paper.{json,png})

Knockout, 253 test engines (runs 0/1/2): normal 13.31/13.65/13.19 → same K for all engines 27.3/30.2/27.9,
h shuffled 36.4/40.9/36.2, spectral features fixed 27.5/30.2/27.9. In arm A, K is the ONLY route for health
into the prediction, and removing per-engine K costs about +15 RMSE.

How the generated K changes over life (every 4th training engine, full lives):
- The operator is very dissipative: typical rho ≈ 0.13 per step, half-life < 1 step.
- Mean rho rises near failure: 0.13 (>150 flights left) → 0.24 (at failure); corr(mean rho, h) −0.94..−0.98.
- Slow modes (rho > 0.3): healthy 2 / 13 / 1 → near failure (<30 left) 17 / 32 / 60; corr with h −0.86..−0.98.
- So the latent space is squeezed HARDER while healthy and kept larger near failure — the opposite of the
  original "shrinks faster near failure" idea, consistent in all 3 seeds. With half-lives below one step, the
  spectrum acts as a health code read by the head's spectral features rather than as multi-step dynamics.

### Clean protocol (val = 20% of training engines; test touched once) — arms A/B/C, FD002, 3 seeds

| Arm | single runs | ens RMSE | ens score |
|---|---|---|---|
| eig_static (reference, clean) | 15.44 / 15.84 / 16.09 | 15.34 | 1018 |
| A: K from h only | 14.17 / 13.78 / 14.00 | 13.70 | 847 |
| B: K from h + age, rank gates | 12.39 / 12.48 / 12.12 | **11.77** | **610** |
| C: static K, h + age into head | 12.96 / 12.06 / 11.98 | 11.99 | 636 |

Bootstrap: B − C −0.22 [−0.76, +0.32] (tie); A − static −1.64 [−3.31, −0.04]; B − static −3.57 [−5.21, −1.96];
C − static −3.34 [−4.96, −1.67]. The paper-protocol picture holds without test-set selection (≈ +0.7 RMSE
everywhere). Arm A clean knockout: same K for all +15.2, h shuffled +25.9, spectral fixed +15.3; corr(mean rho, h)
−0.96..−0.99; slow modes (rho > 0.3) healthy 0–1 → near failure 49–64 (all 3 seeds): the latent space opens up near
failure, as in the paper runs. B: direction of K vs h still flips across seeds (+0.86, −0.64, +0.04).

**9 Nov 2026, teaching PDF updated.** `../KePIN_Project_Explained.pdf` now has Part 2 (sections 13–20: adaptive Koopman,
wear-model tests, encoder curvature, hypernetwork architecture, history-health arms A/B/C with both protocols, arm A
readability, where we stand) and an extended glossary; 37 pages. New figures: `reports/teaching/make_figures_part2.py`.
Build: `PYTHONPATH=<reportlab dir> python reports/teaching/build_pdf.py`. Previous version kept as
`reports/teaching/KePIN_Project_Explained_2026-10-21.pdf` (script backup `build_pdf_2026-10-21.py.bak`).

### 2026-11-09 — History-health hypernetwork on FD001, FD003, FD004 (jobs 6182, 6183)
Same arms A/B/C as FD002, both protocols, 3 seeds: 54 runs (tasks/hyper_health_all.txt, FD004 first).
Baseline = existing svd_static (original KePIN) runs under both protocols. History caches built
(runs/cache/hist_FD00x_v1.npz); aligned with windows; ridge R2 of capped RUL from the 42 features on test
engines: FD001 0.81, FD003 0.82, FD004 0.83. Smoke test FD001 (B, C) passes.
Results (2026-11-10; ensemble RMSE / NASA score; baseline = svd_static original KePIN; full output with bootstrap CIs
in reports/hyper_health/all_datasets.txt):
| dataset | protocol | KePIN | A (K from h) | B (K from h+age, gates) | C (static K, h+age to head) | B − C [95%] |
| FD001 | clean | 14.52 (1 seed) / 450 | 13.98 / 312 | 11.68 / 216 | 11.85 / 229 | −0.17 [−0.97, +0.64] |
| FD001 | paper | 12.92 / 268 | 12.28 / 229 | 10.59 / 192 | 10.55 / 176 | +0.05 [−0.49, +0.55] |
| FD002 | clean | 15.40 / 1062 | 13.70 / 847 | 11.77 / 610 | 11.99 / 636 | −0.22 [−0.77, +0.32] |
| FD002 | paper | 15.00 / 869 | 13.09 / 668 | 11.03 / 543 | 11.04 / 465 | −0.01 [−0.53, +0.52] |
| FD003 | clean | 13.12 / 373 | 13.14 / 294 | 11.06 / 221 | 11.31 / 234 | −0.25 [−0.90, +0.37] |
| FD003 | paper | 11.56 / 227 | 11.61 / 201 | 9.90 / 149 | 10.11 / 158 | −0.21 [−0.98, +0.57] |
| FD004 | clean | 17.84 / 1658 | 15.21 / 1131 | 12.92 / 754 | 12.64 / 759 | +0.29 [−0.29, +0.89] |
| FD004 | paper | 17.11 / 1265 | 14.51 / 956 | 11.84 / 593 | 11.88 / 588 | −0.04 [−0.53, +0.46] |
- B and C beat KePIN on every dataset/protocol (−1.5 to −5.3); CI excludes 0 except FD003 (−2.1/−1.7, CI up to +0.4).
- B ties C everywhere (all CIs include 0). A helps on the multi-condition sets (FD002 −1.7, FD004 −2.6) but not on
  FD001/FD003 (−0.5..+0.05).
- Arm A readability: corr(mean rho, h) −0.96..−0.98 in all 12 clean runs (rho rises near failure everywhere); paper
  runs are mixed (FD001 run1 +0.52, FD004 runs 0/1 +0.60/+0.78, FD003 run0/1 positive with h flipped).
- Caveat: h came out FLIPPED (corr with RUL ≈ −0.9) in 9 of 72 h-runs (mostly arm C, FD003/FD004): the mono loss did
  not hold the direction there. Harmless for RMSE (the head reads either sign) but h is then "damage", not "health".
- FD001 clean baseline had only run0; runs 1–2 queued (job 6258).

### 2026-11-10 — Explainability analysis of arm A on all datasets + report/PDF update
`analyze_hyper_explain.py <ds> clean` -> reports/hyper_explain/<ds>_clean.{json,npz}: knockout, life curves (h, mean rho,
slow modes), eigenvalues healthy vs failure, per-mode rho vs h, and a "health dial" (override h = 0..1 for 5 test engines).
All 12 fair-test arm A models: h falls in 100% of engines (corr RUL 0.87–0.95), corr(mean rho, h) −0.96..−0.99, mean rho
~0.15 → 0.27–0.43 near failure, slow modes ~0–1 → 11–64; knockout same K +7.6..+15.8, h shuffled +13..+26. Dial: h 0→1 moves
the prediction by ~35–95 flights; slow modes switch on below h ≈ 0.25 (FD002). PDF sections 20–23 added
(reports/teaching/make_figures_part3.py; tables computed from runs at build time); FULL_REPORT/REPORT.md Part 3 generated by
reports/make_report_part3.py (re-run to refresh). FD001 clean svd_static runs 1–2 were HOLD claims from 2026-10-19; released,
job 6259.
FD001 clean svd_static runs 1–2 done (15.31, 14.73; run0 14.52): ensemble 14.32 / score 399 (was 14.52 with 1 seed).
FD001 fair test: A −0.34 [−2.22, +1.47], B −2.65 [−4.37, −0.91], C −2.48 [−4.19, −0.86], B − C −0.17 [−0.99, +0.65].
PDF, FULL_REPORT/REPORT.md and FULL_REPORT.zip refreshed with all seeds.

### 2026-11-10 — Reviewer check: simple models on the history features alone (no deep network)
42 history features + age/100, train on all training windows (capped RUL 125), predict test, clip to [0,125]; one fit,
no model selection. Test RMSE: ridge 17.15 / 16.31 / 16.01 / 16.96; gradient-boosted trees (HistGB, 400 iters, lr 0.05)
**10.45 / 13.00 / 10.68 / 12.70** (FD001/2/3/4). vs best deep (fair test) 11.68 / 11.77 / 11.06 / 12.64.
=> On FD001/FD003 the features + trees beat our deep models; on FD004 tie; only FD002 is the deep model clearly better.
The accuracy gain is the feature engineering, not KePIN or Koopman. Any accuracy claim must include this baseline.

### 2026-11-10 — Cross-domain study (original KePIN's multi-domain benchmark; jobs 6272, 6273)
User: check cross-domain behaviour like original KePIN. Datasets via KePIN's own loader/configs (dk_domain.py, same
windows as the Signal A cross-domain runs): Cylinder_Wake (150 flows, lift 5 steps ahead), Building_Energy (120
buildings, energy 6 h ahead), Jena_Climate (temperature 24 h ahead), SPY_Stock (time to a 5% drawdown).
History context = same definition as C-MAPSS (drift of last 10 steps vs first 10 of the series, slope of last 30;
2 x n_features); series recovered from window overlap (120/30, 96/24, 1/1, 1/1 series = KePIN's splits). Single
series (Jena, SPY) use a trailing baseline (10 windows back), since their start is arbitrary (with "since start"
Jena's test R2 from history was −2.2; trailing: 0.37). Ridge test R2 from history / last window step: Cylinder
0.85/0.86, Building 0.35/0.40, Jena 0.37/0.42, SPY −5.0/−2.4.
Training as KePIN's multi-domain runs: target min-max to [0,1] (metrics in original units), forecasting losses only
(rul, koopman, spectral, multi_step), no mixup / input noise / label smoothing, arch = KePIN auto_configure, epochs/
lr/batch from the Signal A cross-domain script (SPY from run_optimized_multidomain.py). Fit on every 2nd (Cylinder) /
4th (Building) training window to bound cost; validation and test untouched. h has NO mono loss (no degradation
order outside RUL): a free 1-number regime bottleneck. Clean protocol only.
Arms x 3 seeds: svd_static (KePIN), eig_static, eig_hyper (window context), eig_hyper_hh (A), eig_hyper_hh_age_rank (B),
eig_static_hh_hctx (C). 72 runs, tasks/cross_domain.txt. Code: dk_model hist_dim (was fixed 42), train.py domain mode.
Results (2026-11-11, 70/72 runs; 2 Jena runs (eig_static r0, B r0) hit GPU OOM with 2 workers/GPU, re-running as
job 6336 with 1 worker/GPU). Ensemble RMSE (original units), clean protocol; full output reports/cross_domain_partial.txt:
| dataset | KePIN | static K | hyper (window) | A | B | C |
| Cylinder_Wake | 0.0051 | 0.0051 | 0.0051 | 0.0051 | 0.0051 | 0.0051 |  (R2 0.90 for all; differences < 1e-5)
| Building_Energy | **0.0921** | 0.0936 | 0.0959 | 0.0938 | 0.0930 | 0.0934 |
| Jena_Climate | **5.24** | 5.37 (2 seeds) | 5.25 | 5.26 | 5.30 (2 seeds) | 5.65 |
| SPY_Stock | 43.1 | 38.4 | 33.4 | 34.4 | 37.2 | 33.5 |  (R2 < 0 for EVERY model: worse than predicting the mean)
- No arm beats original KePIN on Cylinder / Building / Jena (block-bootstrap ranges include 0 or favour KePIN).
- The hypernetwork barely varies the operator: spread of rho across test windows 0.000–0.014 (A: ~0.000 on
  Building/Cylinder). h does not track the target (|corr| mostly < 0.3, sign flips across seeds).
- Why (consistent with the rule): the history context carries no new information here — ridge R2 from history vs
  from the last window step: Cylinder 0.85 vs 0.86, Building 0.35 vs 0.40, Jena 0.37 vs 0.42. Nothing outside the
  window to route through K, so K is not used.
- SPY: unpredictable target (time to 5% drawdown, 525 test days, one regime period); differences there are noise
  between equally bad models — block bootstrap with 2 blocks is not valid, ignore those ranges.
- Reference (Signal A cross-domain, paper protocol, same windows): Cylinder 0.0051, Building 0.0785, Jena 5.2466.
Jena final (all 3 seeds, 2026-11-15): KePIN 5.24, static K 5.34, hyper(window) 5.25, A 5.26, B 5.22, C 5.65 — B vs KePIN within noise; conclusions unchanged.
