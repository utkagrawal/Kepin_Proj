# Does a Dynamic Koopman Operator Improve KePIN? Results on FD002

*Date: 19 October 2026 · Dataset: NASA C-MAPSS FD002 · 3 training runs per model, with the same seeds as the KePIN paper*

## Summary

- **No dynamic version improved KePIN on FD002.** Letting the Koopman rule follow the operating condition (A), a learned health value (B) or both (A+B) gave the same error as a fixed rule, within training noise.
- **Switching Koopman off entirely also made no measurable difference.** Same model, same training, only the Koopman losses removed: the error changed by +0.13 (fair test) and −0.12 (paper's method). Both are well inside the run-to-run noise.
- **The reason:** in KePIN, the Koopman operator does not feed any engine-specific information into the prediction. The 12 "spectral features" it passes to the predictor are the same numbers for every engine.
- **Two problems with the original KePIN code:**
  - it picks its final model using the **test set**, which makes its scores look about 0.4–0.6 better than a fair evaluation
  - its ablation script never actually tested "without Koopman"
- **We reproduced the paper.** Our original KePIN scores 15.00 (ensemble, the paper's method) against the paper's 15.08, so the setup is trustworthy.

---

## 1. What we tested

KePIN predicts how many flights (cycles) a jet engine has left before failure: its Remaining Useful Life (RUL). Inside it is a **Koopman operator**, a learned rule that says how the engine's hidden state moves from one time step to the next. In the original KePIN this rule is **fixed**: the same for every engine at every moment.

We built a faster version of this layer, then tried making the rule **dynamic**:

| Model | What the Koopman rule depends on |
|---|---|
| Original KePIN | Nothing (fixed). Original code. |
| Original KePIN, Koopman switched off | Same model, but the Koopman training losses are removed (ablation). |
| New fast layer, fixed rule | Nothing (fixed). Our faster layer; it sets the rule's eigenvalues directly. |
| A: operating condition | Changes at every time step with the flight condition (6 conditions in FD002). |
| B: learned health | Changes with a health value the model learns for each 30-cycle window. |
| A + B | Both. |

Every dynamic model starts out identical to the fixed one, so any difference comes from what it learns. The rest of the model, the training recipe, the hyper-parameters and the 3 seeds are exactly those of the KePIN paper.

### Two ways of evaluating

The original KePIN code uses the **test set** to choose which training epoch to keep (and whether to use the averaged "SWA" weights). That is like peeking at the exam answers to decide which practice attempt to submit. We report both:

- **Fair test (clean protocol):** 20% of the *training* engines are held out to choose the model. The test set is used only once, at the end. **These are the main results.**
- **Paper's method (paper protocol):** the test set chooses the model, exactly as in the original code. Use these only to compare with the paper's numbers.

### Average vs ensemble

- **Average of 3 runs:** train 3 times, compute each model's error, and average the 3 errors. This shows how good one typical training run is. We also give the spread (± std).
- **Ensemble of 3 runs:** average the 3 models' *predictions* for each engine, then compute a single error. Combining models cancels some of their mistakes, so it is usually lower. **The paper's 15.08 is an ensemble number.**

---

## 2. Results

![Test RMSE of every model on FD002](fd002_all_arms.png)

*Grey dots: single training runs. Blue: average of 3 runs (± std). Orange: ensemble of 3 runs. Lower is better.*

| Model | Fair test: average ± std | Fair test: ensemble | Paper's method: average ± std | Paper's method: ensemble |
|---|---|---|---|---|
| Original KePIN | 15.98 ± 0.45 | 15.40 | 15.40 ± 0.30 | **15.00** (paper: 15.08) |
| Original KePIN, Koopman switched off | 16.12 ± 0.29 | 15.49 | 15.28 ± 0.39 | 14.81 |
| New fast layer, fixed rule | **15.79 ± 0.33** | **15.34** | 15.34 ± 0.23 | 14.91 |
| A: operating condition | 16.16 ± 0.06 | 15.47 | 15.31 ± 0.07 | 14.98 |
| B: learned health | 15.96 ± 0.72 | 15.45 | 15.07 ± 0.49 | 14.65 |
| A + B | 16.19 ± 0.42 | 15.68 | 15.16 ± 0.42 | 14.80 |

### Are the differences real?

Each model was trained with the same 3 seeds, so we compare them seed by seed (a paired t-test). A p-value below 0.05 would mean a difference is unlikely to be chance.

| Comparison (fair test) | Difference per seed | Mean difference | p-value |
|---|---|---|---|
| A vs fixed rule | +0.77, +0.35, 0.00 | +0.37 (A worse) | 0.24 |
| B vs fixed rule | +0.10, +0.95, −0.56 | +0.17 | 0.74 |
| A+B vs fixed rule | +0.29, +0.70, +0.20 | +0.40 (A+B worse) | 0.12 |
| Koopman off vs original KePIN | +0.54, −0.35, +0.21 | +0.13 | 0.66 |
| Original layer vs new fast layer | +0.03, +0.41, +0.14 | +0.19 | 0.23 |

**None of the differences is statistically significant.** With 3 seeds we can rule out large effects, but not small ones (below about 0.5).

---

## 3. Findings

### Finding 1: Making the Koopman rule dynamic does not help on FD002

On the fair test, the fixed rule has the lowest average error (15.79) and the lowest ensemble error (15.34). A, B and A+B are equal or slightly worse, and never significantly.

Under the paper's method, B has the lowest numbers (15.07 average, 14.65 ensemble). B also has the largest spread between runs, and when the test set picks the model, the model with the most spread benefits most from that selection. On the fair test, B shows no advantage. **We do not count B's paper-method result as a real improvement.**

### Finding 2: Removing Koopman makes no measurable difference

Same model, same training, only the three Koopman losses removed (one-step prediction, multi-step prediction and spectral stability):

- Fair test: 16.12 without Koopman vs 15.98 with it (difference +0.13, p = 0.66)
- Paper's method: 15.28 without Koopman vs 15.40 with it (difference −0.12, p = 0.67)

The sign flips between the two evaluations and between seeds. On FD002, KePIN's accuracy comes from the rest of the network (convolution, LSTM and attention), not from the Koopman part.

### Finding 3: The Koopman operator gives the predictor no engine-specific information

The final predictor in KePIN receives:

1. a summary of the encoder output for that engine (different for each engine)
2. 12 spectral features from the Koopman operator (decay rates, frequencies and so on)

In the original KePIN the operator is **one fixed matrix**. Its spectral features are therefore computed once and **copied to every engine** (`tf.tile` in `kepin_model.py`, lines 302–310 of the original code). A brand-new engine and one about to fail receive exactly the same 12 Koopman numbers. Numbers that never change cannot help tell engines apart; they act like a constant offset.

We also tested this in our dynamic models, where the 12 numbers *do* differ per engine. We replaced them with a constant before prediction and measured the change:

| Model | Error with real features | Error with features replaced by a constant | Largest change in any prediction |
|---|---|---|---|
| B: learned health, run 0 | 15.538 | 15.538 | 0.000 flights |
| A: operating condition, run 0 | 16.207 | 16.208 | 0.09 flights |

Even when the information is available, the predictor learned not to use it.

This is the key explainability result: **the paper presents the Koopman spectral features as an interpretable part of the prediction, but they cannot explain any individual engine's prediction, and on FD002 the Koopman training losses do not measurably change accuracy either.**

### Finding 4: The "health" value in model B did not learn health

Model B learns a health value between 0 and 1 for each engine. If it captured wear, it would follow the true remaining life closely. Its correlation with true RUL on the test engines was **+0.10, −0.20 and −0.17** for the three runs: weak, and with inconsistent sign. Nothing in training forced it to mean "wear", so it learned something else.

### Finding 5: The paper's evaluation is optimistic by about 0.4–0.6

Every model scores about 0.4–0.6 better (on the average of 3 runs) when the test set picks the model than on the fair test. Part of the gap comes from the fair test training on 80% of the engines, but most of it is the test-set selection. The paper's 15.08 should be read with this in mind.

### Finding 6: The original ablation script could not answer "does Koopman help?"

The KePIN repository has an ablation script (`kepin_ablation.py`), but:

- **"w/o Spectral" and "w/o Multi-step" train the full model.** They set a loss weight to 0 but keep automatic loss weighting on, and automatic weighting ignores those weights.
- **"Baseline FCN" is a different, smaller network** (no LSTM or attention), not KePIN without Koopman. So it measures architecture, not Koopman.
- **The saved results contain no C-MAPSS rows.** Only Building Energy is present, and four of its five runs crashed.

Our ablation (Finding 2) is the first controlled test of this question on FD002.

---

## 4. Limits

- **Only FD002 was tested.** Koopman might matter more on the single-condition datasets, FD001 and FD003.
- **Only 3 seeds**, to match the paper. Effects smaller than about 0.5 RMSE cannot be detected.
- **All dynamic variants kept KePIN's prediction head**, which ignores the Koopman information (Finding 3). They test "a better rule in an unused place", not whether dynamic Koopman models can help in general.

---

## 5. Next step

Make the Koopman operator **drive the prediction** instead of sitting beside it:

1. Map the engine to its hidden state.
2. Use the Koopman rule to fast-forward that state flight by flight.
3. Predict failure at the step where a learned health reading crosses a threshold.

The predicted remaining life is then *the number of Koopman steps to failure*. Koopman must be used, and the prediction is explainable by construction: you can show the projected health trajectory and the step where it crosses the threshold.

---

## Files

| Path (relative to `dynamic_koopman/`) | Contents |
|---|---|
| `eigen_koopman.py` | The fast eigenvalue-based Koopman layer (fixed, A, B, A+B) |
| `train.py` | Trains one model; `--protocol clean/paper`, `--no_koopman_loss` |
| `runs/<protocol>/FD002/<model>/` | Per-run results, predictions, training history, weights |
| `runs/summary.csv` | All numbers in the table above (from `aggregate.py`) |
| `reports/make_fd002_figures.py` | Regenerates the figure |
| `README.md` | Setup, how to run, experiment log |
