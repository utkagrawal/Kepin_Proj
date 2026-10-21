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
