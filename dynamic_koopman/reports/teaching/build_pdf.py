#!/usr/bin/env python3
"""Builds KePIN_Project_Explained.pdf: an easy-language guide to the whole investigation.

Run with reportlab on PYTHONPATH:  python build_pdf.py
Figures come from make_figures.py (fig/) and the earlier analyses in the repo.
"""
import os

import matplotlib
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (Image, KeepTogether, PageBreak, Paragraph, Preformatted,
                                SimpleDocTemplate, Spacer, Table, TableStyle)

HERE = os.path.dirname(os.path.abspath(__file__))
DK = os.path.abspath(os.path.join(HERE, "..", ".."))
REPO = os.path.abspath(os.path.join(DK, ".."))
FIG = os.path.join(HERE, "fig")
OUT = os.path.join(REPO, "KePIN_Project_Explained.pdf")

# ---------------------------------------------------------------- fonts & styles
FONT_DIR = os.path.join(os.path.dirname(matplotlib.__file__), "mpl-data", "fonts", "ttf")
pdfmetrics.registerFont(TTFont("DV", os.path.join(FONT_DIR, "DejaVuSans.ttf")))
pdfmetrics.registerFont(TTFont("DVB", os.path.join(FONT_DIR, "DejaVuSans-Bold.ttf")))
pdfmetrics.registerFont(TTFont("DVI", os.path.join(FONT_DIR, "DejaVuSans-Oblique.ttf")))
pdfmetrics.registerFont(TTFont("DVM", os.path.join(FONT_DIR, "DejaVuSansMono.ttf")))
pdfmetrics.registerFontFamily("DV", normal="DV", bold="DVB", italic="DVI", boldItalic="DVB")

INK = colors.HexColor("#1d1d1b")
MUTED = colors.HexColor("#6b6a66")
BLUE = colors.HexColor("#2a78d6")
ACCENT = colors.HexColor("#1f5fae")
WHY_BG = colors.HexColor("#fff4e5")
WHY_EDGE = colors.HexColor("#eda100")
KEY_BG = colors.HexColor("#e8f1fc")
KEY_EDGE = BLUE
CODE_BG = colors.HexColor("#f4f3ef")
GRIDC = colors.HexColor("#cfcdc6")

S = {
    "title": ParagraphStyle("title", fontName="DVB", fontSize=24, leading=29, textColor=INK, spaceAfter=6),
    "subtitle": ParagraphStyle("subtitle", fontName="DV", fontSize=12.5, leading=17, textColor=MUTED),
    "h1": ParagraphStyle("h1", fontName="DVB", fontSize=16.5, leading=21, textColor=ACCENT, spaceBefore=4, spaceAfter=8),
    "h2": ParagraphStyle("h2", fontName="DVB", fontSize=12, leading=16, textColor=INK, spaceBefore=9, spaceAfter=4,
                         keepWithNext=1),
    "body": ParagraphStyle("body", fontName="DV", fontSize=10, leading=14.5, textColor=INK, spaceAfter=5),
    "bullet": ParagraphStyle("bullet", fontName="DV", fontSize=10, leading=14, textColor=INK, leftIndent=14,
                             bulletIndent=3, spaceAfter=2.5),
    "caption": ParagraphStyle("caption", fontName="DVI", fontSize=8.5, leading=11.5, textColor=MUTED, spaceAfter=8),
    "box": ParagraphStyle("box", fontName="DV", fontSize=9.8, leading=14, textColor=INK),
    "code": ParagraphStyle("code", fontName="DVM", fontSize=7.6, leading=10, textColor=INK),
    "cell": ParagraphStyle("cell", fontName="DV", fontSize=8.8, leading=11.5, textColor=INK),
    "cellb": ParagraphStyle("cellb", fontName="DVB", fontSize=8.8, leading=11.5, textColor=INK),
    "toc": ParagraphStyle("toc", fontName="DV", fontSize=10.5, leading=15, textColor=INK, leftIndent=6),
}
W = A4[0] - 36 * mm          # usable width


# ---------------------------------------------------------------- building blocks
def P(text, style="body"):
    return Paragraph(text, S[style])


def bullets(items):
    return [Paragraph(t, S["bullet"], bulletText="•") for t in items]


def fig(name, caption=None, width=1.0, folder=FIG):
    path = name if os.path.isabs(name) else os.path.join(folder, name)
    img = Image(path)
    ratio = img.imageHeight / img.imageWidth
    img.drawWidth = W * width
    img.drawHeight = W * width * ratio
    max_h = 205 * mm
    if img.drawHeight > max_h:
        img.drawHeight, img.drawWidth = max_h, max_h / ratio
    parts = [img]
    if caption:
        parts.append(Paragraph(caption, S["caption"]))
    return KeepTogether(parts)


def callout(title, text, bg=WHY_BG, edge=WHY_EDGE):
    inner = [Paragraph(f"<b>{title}</b>", S["box"]), Spacer(1, 2)]
    if isinstance(text, str):
        inner.append(Paragraph(text, S["box"]))
    else:
        inner += [Paragraph(t, S["box"]) for t in text]
    t = Table([[inner]], colWidths=[W])
    t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), bg),
                           ("LINEBEFORE", (0, 0), (0, -1), 3, edge),
                           ("LEFTPADDING", (0, 0), (-1, -1), 9), ("RIGHTPADDING", (0, 0), (-1, -1), 9),
                           ("TOPPADDING", (0, 0), (-1, -1), 6), ("BOTTOMPADDING", (0, 0), (-1, -1), 7)]))
    return KeepTogether([t, Spacer(1, 7)])


def why(text):
    return callout("Why?", text)


def key(text, title="In one sentence"):
    return callout(title, text, KEY_BG, KEY_EDGE)


def code(snippet, source):
    pre = Preformatted(snippet.strip("\n"), S["code"])
    t = Table([[pre], [Paragraph(f"Source: <font name='DVM'>{source}</font>", S["caption"])]], colWidths=[W])
    t.setStyle(TableStyle([("BACKGROUND", (0, 0), (0, 0), CODE_BG),
                           ("BOX", (0, 0), (0, 0), 0.5, GRIDC),
                           ("LEFTPADDING", (0, 0), (-1, -1), 7), ("TOPPADDING", (0, 0), (0, 0), 6),
                           ("BOTTOMPADDING", (0, 0), (0, 0), 6), ("TOPPADDING", (0, 1), (0, 1), 2)]))
    return KeepTogether([t, Spacer(1, 4)])


def table(rows, widths=None, bold_cols=(), highlight_rows=()):
    data = []
    for r, row in enumerate(rows):
        data.append([Paragraph(str(c), S["cellb"] if r == 0 or (c_i in bold_cols) else S["cell"])
                     for c_i, c in enumerate(row)])
    if widths:
        widths = [W * w for w in widths]
    t = Table(data, colWidths=widths, repeatRows=1)
    style = [("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#eceae4")),
             ("LINEBELOW", (0, 0), (-1, 0), 0.8, INK),
             ("LINEBELOW", (0, 1), (-1, -1), 0.3, GRIDC),
             ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
             ("LEFTPADDING", (0, 0), (-1, -1), 5), ("RIGHTPADDING", (0, 0), (-1, -1), 5),
             ("TOPPADDING", (0, 0), (-1, -1), 3.5), ("BOTTOMPADDING", (0, 0), (-1, -1), 3.5)]
    for r in highlight_rows:
        style.append(("BACKGROUND", (0, r), (-1, r), colors.HexColor("#e3f4ec")))
    t.setStyle(TableStyle(style))
    return KeepTogether([t, Spacer(1, 8)])


def section(num, title):
    return [PageBreak(), P(f"{num}. {title}", "h1")]


def page_decor(canv, doc):
    canv.saveState()
    canv.setFont("DV", 8)
    canv.setFillColor(MUTED)
    canv.drawString(18 * mm, 10 * mm, "Making KePIN better: an easy guide to what we did and why")
    canv.drawRightString(A4[0] - 18 * mm, 10 * mm, f"page {doc.page}")
    canv.restoreState()


# ---------------------------------------------------------------- content
def build():
    sig = os.path.join(REPO, "RCKePIN_Signal_A", "results", "condition_net_interpretability_fd002")
    rep = os.path.join(DK, "reports")
    st = []

    # ---------- cover
    st += [Spacer(1, 12 * mm), P("Making KePIN Better", "title"),
           P("An easy guide to what we did, what we found, and <b>why</b>", "subtitle"), Spacer(1, 6 * mm),
           P("Remaining-useful-life prediction for jet engines (NASA C-MAPSS) · 18–21 October 2026", "subtitle"),
           Spacer(1, 8 * mm)]
    st.append(key([
        "We started out trying to improve KePIN by making its Koopman part \"dynamic\". By explaining what the "
        "model was really doing, we found that KePIN's Koopman part was not driving its accuracy, and that the real "
        "problem on the hardest datasets was <b>flying-condition noise</b> drowning out the <b>wear signal</b>.",
        "Fixing that inside the architecture (a <b>condition-aware KePIN</b>) beat the original KePIN and the "
        "published paper by <b>1.1 RMSE on FD002</b> and <b>1.8 RMSE on FD004</b>. Along the way we built a "
        "<b>wear model</b> that matches KePIN's accuracy and explains every single prediction."],
        "The whole story in one paragraph"))
    st.append(P("How to read this guide", "h2"))
    st += bullets([
        "Each section follows the same pattern: <b>what we did</b>, the <b>proof</b> (a figure, real code or a table), "
        "a yellow <b>Why?</b> box, and what it means.",
        "Blue boxes give the short version. You can read just the blue and yellow boxes for a quick overview.",
        "All numbers are RMSE (how many flights off the predictions are, on average). <b>Lower is better.</b>"])
    st.append(P("Contents", "h2"))
    for i, t in enumerate(["KePIN architecture and RUL prediction",
                           "Why Koopman wasn't helping: the feature replacement experiment",
                           "Koopman ablation study across FD001–FD004",
                           "Signal A explainability: operating conditions vs engine wear",
                           "Why dynamic Koopman failed",
                           "The time-to-failure (TTF) model and its failure",
                           "The wear model: wear accumulator and failure threshold",
                           "Wear model explainability and validation",
                           "Condition-aware KePIN: the final architecture",
                           "Condition-aware normalisation and EMA smoothing",
                           "Koopman with control",
                           "Final results, ablation and limitations",
                           "Glossary"], 1):
        st.append(P(f"{i}.&nbsp;&nbsp;{t}", "toc"))

    # ---------- 1
    st += section(1, "KePIN Architecture and RUL Prediction")
    st.append(key("KePIN reads the last 30 flights of an engine's sensors and predicts how many flights it has left; "
                  "it has a \"Koopman\" part that is supposed to model how the engine changes over time."))
    st.append(P("The problem: Remaining Useful Life (RUL)", "h2"))
    st.append(P("Every jet engine slowly wears out. If we can predict how many flights an engine has left before it "
                "fails, airlines can service it just in time: not too early (wasteful), not too late (dangerous). "
                "That number is the <b>Remaining Useful Life (RUL)</b>."))
    st.append(P("The data: NASA C-MAPSS", "h2"))
    st.append(P("C-MAPSS is a simulated fleet of engines. Each engine has 21 sensors (temperatures, pressures, speeds) "
                "and 3 operating settings, recorded once per flight until the engine fails. There are four sub-datasets:"))
    st.append(table([["Dataset", "Flying conditions", "Failure types", "Difficulty", "Paper's KePIN RMSE"],
                     ["FD001", "1", "1", "easiest", "12.92"],
                     ["FD002", "6", "1", "hard", "15.08"],
                     ["FD003", "1", "2", "medium", "11.42"],
                     ["FD004", "6", "2", "hardest", "17.04"]], [0.15, 0.2, 0.17, 0.18, 0.3]))
    st.append(P("A <b>flying condition</b> is a combination of altitude, speed and throttle. On FD002 and FD004 an "
                "engine switches between 6 of them, almost every flight. Remember this: it turns out to be the key "
                "to the whole story."))
    st.append(P("How KePIN makes a prediction", "h2"))
    st.append(fig("01_kepin_architecture.png", "Figure 1. The original KePIN. The encoder turns 30 flights of sensor "
                  "data into a hidden state z. The predictor reads a summary of z plus 12 numbers from the Koopman rule."))
    st.append(P("<b>What is the Koopman part?</b> A Koopman operator is a learned \"rule of motion\": a matrix K that "
                "moves the hidden state from one flight to the next (next state = K × current state). It splits the "
                "engine's behaviour into simple patterns (\"modes\") that each fade at their own speed. The idea is "
                "elegant: complex, non-linear wear becomes simple, linear motion in the hidden space."))
    st.append(P("Two things you need to know about how results are measured", "h2"))
    st += bullets([
        "<b>Average vs ensemble.</b> We train every model 3 times (3 \"seeds\"). The <b>average</b> is the mean error "
        "of the 3 models. The <b>ensemble</b> first averages the 3 models' predictions, then scores once (usually "
        "a bit better). <b>The paper reports ensembles</b>, so that is our headline number.",
        "<b>Fair test vs the paper's method.</b> Training produces many versions of the model; one must be chosen. "
        "The original code chooses it using the <b>test set itself</b>, which is like checking practice attempts against "
        "the real exam answers. We call this the <b>paper's method</b>, and we use it so our numbers compare directly "
        "with the paper. Choosing on held-out <i>training</i> engines instead (the <b>fair test</b>) gives errors about "
        "0.4–0.6 higher."])
    st.append(code("""
history, swa_weights = trainer.fit_enhanced(
    X_train_aug, Y_train_aug, X_test_tuple, Y_test,      # <- the TEST set is passed as "validation"
    ...
if val_rmse < best_val_rmse:                              # <- so "best" means best on the test set
    best_val_rmse = val_rmse
    best_weights = self.model.get_weights()""",
                   "RCKePIN_Signal_A/train_fd001_conditioned.py:169-170 and kepin_cmapss_optimized.py:486-489"))
    st.append(why("We kept the paper's method so our results are an apples-to-apples comparison with the published "
                  "numbers. Our reproduction of KePIN matches the paper on all four datasets (12.92 / 15.00 / 11.56 / 17.11 "
                  "vs 12.92 / 15.08 / 11.42 / 17.04), so the baseline we compare against is faithful."))

    # ---------- 2
    st += section(2, "Why Koopman Wasn't Helping: the Feature Replacement Experiment")
    st.append(key("The 12 numbers Koopman gives KePIN's predictor are exactly the same for every engine, so they "
                  "cannot help tell a healthy engine from a failing one."))
    st.append(P("What we found in the code", "h2"))
    st.append(P("In the original KePIN, the Koopman rule K is <b>one fixed matrix</b>, the same for every engine. Its 12 "
                "\"spectral features\" (decay rates, frequencies and so on) are computed once from K, then simply "
                "<b>copied</b> to every engine in the batch:"))
    st.append(code("""
spec_feats = extract_spectral_features(
    koopman_out["eigenvalues"], top_k=self.spectral_k,    # computed from K only (not from the engine)
)
batch_size = tf.shape(inputs)[0]
spec_feats_batch = tf.tile(                               # the SAME 12 numbers copied to every engine
    tf.expand_dims(tf.cast(spec_feats, tf.float32), 0), [batch_size, 1],
)
head_input = tf.concat([pooled, spec_feats_batch], axis=-1)""", "original_kepin/code/kepin_model.py:303-311"))
    st.append(P("The feature replacement experiment", "h2"))
    st.append(P("In our <i>dynamic</i> Koopman versions (section 5), the 12 numbers <i>do</i> differ per engine. So we "
                "tested whether the predictor uses them: we took the trained models, <b>replaced the 12 numbers with a "
                "constant</b>, and measured how much the predictions changed."))
    st.append(table([["Model (FD002)", "Error with real features", "Error with a constant", "Largest change in any prediction"],
                     ["Dynamic Koopman B (health), run 0", "15.538", "15.538", "0.000 flights"],
                     ["Dynamic Koopman A (condition), run 0", "16.207", "16.208", "0.09 flights"]],
                    [0.34, 0.2, 0.2, 0.26]))
    st.append(why(["Analogy: a doctor predicts each patient's health from their test results plus a \"Koopman note\". "
                   "If the note is the same printed sheet for every patient, it cannot help tell patients apart.",
                   "And even when the note <i>was</i> different per patient, the predictor had learned to ignore it, "
                   "because it already received the full hidden state (192–256 numbers) directly."]))
    st.append(P("<b>What it means:</b> in KePIN, Koopman never informs an individual prediction. It can only influence "
                "accuracy indirectly, through its training losses shaping the encoder. Section 3 tests whether even that helps."))

    # ---------- 3
    st += section(3, "Koopman Ablation Study Across FD001–FD004")
    st.append(key("We removed Koopman completely from KePIN and retrained it on all four datasets: accuracy did not "
                  "change significantly anywhere."))
    st.append(P("What we did", "h2"))
    st.append(P("An <b>ablation</b> means removing one part and retraining, to see whether that part matters. We built "
                "KePIN with <b>no Koopman layer, no spectral features and no Koopman losses</b>, keeping everything else "
                "identical (same encoder, predictor, training recipe and seeds). Command: "
                "<font name='DVM'>train.py --koopman none</font>."))
    st.append(fig("03_ablation.png", width=0.82, caption="Figure 2. Ensemble of 3 runs, paper's method. p-values compare the two versions "
                  "seed by seed; below 0.05 would mean a real difference."))
    st.append(why("Removing Koopman sometimes helps (FD002, FD003) and sometimes hurts (FD001, FD004), and never "
                  "significantly. That pattern is what you expect when a part does very little. KePIN's accuracy "
                  "comes from its encoder (convolution + LSTM + attention), not from Koopman."))
    st.append(P("Why the original paper never found this", "h2"))
    st.append(P("The KePIN repository has an ablation script, but its \"without spectral\" and \"without multi-step\" "
                "variants set a loss weight to 0 <b>while also switching on automatic loss weighting</b>, and the "
                "automatic weighting ignores those zeros:"))
    st.append(code("""
{"name": "B: w/o Spectral", "use_auto_weights": True,          # auto-weighting ON ...
 "fixed_weights": {..., "spectral": 0.0, ...}}                 # ... so this 0.0 is never used

if ab_config["use_auto_weights"] and ab_config["use_koopman"]:
    loss_fn = make_kepin_loss(loss_weights_layer=model.loss_weight_layer,
                              use_auto_weights=True)           # fixed_weights not even passed""",
                   "original_kepin/code/kepin_ablation.py:136-144 and 372-376"))
    st.append(P("So those variants trained the full model. Its \"no Koopman\" baseline used a different, smaller network, "
                "and its saved results contain no C-MAPSS rows at all. Our ablation is the first controlled test."))

    # ---------- 4
    st += section(4, "Signal A Explainability: Operating Conditions vs Engine Wear")
    st.append(key("The earlier dynamic model (\"Signal A\") was supposed to react to engine wear, but explaining it "
                  "showed that it reacted to the flying conditions instead: a shortcut."))
    st.append(P("What Signal A was", "h2"))
    st.append(P("Before this work, the project's dynamic model computed 4 statistics of each 30-flight window (mean, "
                "variance, skewness, kurtosis), called <b>Signal A</b>, and fed them to a small network that changed "
                "the Koopman rule. The claim was that it speeds up the model's \"decay\" as the engine wears out."))
    st.append(P("How we tested what it really follows", "h2"))
    st.append(P("For every window we computed the <b>mix of flying conditions</b> (fraction of its 30 flights in each "
                "of the 6 conditions) and the true RUL. Then we asked which of the two predicts Signal A and the "
                "network's output better (random forests, always tested on engines they never saw)."))
    st.append(fig(os.path.join(sig, "confound_r2.png"), "Figure 3. R² = how much of each quantity can be explained "
                  "(1 = all, 0 = nothing). Blue: by the flying-condition mix. Orange: by remaining life (wear). "
                  "Blue wins everywhere."))
    st.append(why(["KePIN scales every sensor with <b>one global average and spread</b> for all 6 conditions. A window "
                   "that switches conditions a lot therefore looks \"spread out\" and \"spiky\", even for a healthy "
                   "engine. Signal A's variance and kurtosis mostly measure <i>condition switching</i>, so the network "
                   "learned to follow that.",
                   "This is a <b>shortcut</b>: the model found an easy pattern that is not the one we wanted. "
                   "It was only discovered because we explained the model instead of just looking at its RMSE."]))
    st.append(fig(os.path.join(sig, "shapley_mean_sigma.png"), "Figure 4. Shapley values (an explainability method "
                  "that fairly splits the output between inputs): kurtosis, the \"spikiness\" statistic most sensitive "
                  "to condition switching, drives the network the most in all 3 runs.", 0.75))
    st.append(P("<b>What it means:</b> this finding is the seed of our final improvement (sections 9–10): fix the "
                "flying-condition problem <i>inside</i> the architecture."))

    # ---------- 5
    st += section(5, "Why Dynamic Koopman Failed")
    st.append(key("Making the Koopman rule change over time did not help, because it only re-packaged information "
                  "the predictor already had, and the predictor ignored it."))
    st.append(P("What we tried (FD002, 3 seeds)", "h2"))
    st.append(P("We first built a faster Koopman layer that sets the rule's eigenvalues directly. The original layer had "
                "to compute them for every engine on the CPU, which took about 450 s per training epoch. Then we tried "
                "three ways of making the rule dynamic:"))
    st.append(table([["Version", "What the Koopman rule followed", "Fair-test RMSE (avg of 3)", "vs fixed rule"],
                     ["Fixed rule (baseline)", "nothing", "15.79", "-"],
                     ["A", "the flying condition, at every flight", "16.16", "worse (p = 0.24)"],
                     ["B", "a learned \"health\" value", "15.96", "no better (p = 0.74)"],
                     ["A + B", "both", "16.19", "worse (p = 0.12)"]], [0.18, 0.36, 0.23, 0.23]))
    st.append(fig("05_dynamic_failure.png"))
    st.append(why(["<b>1. No new information.</b> The dynamic rule was computed from things the predictor already "
                   "had (the flying condition is visible in the sensors; the \"health\" came from the same hidden state).",
                   "<b>2. It followed the wrong thing.</b> A followed flying conditions, which say nothing about wear. "
                   "B's \"health\" was never told what health means: its correlation with true RUL was +0.10, −0.20 "
                   "and −0.17 in the 3 runs.",
                   "<b>3. A weak channel.</b> The 12 numbers involve sorting eigenvalues, which makes learning signals "
                   "awkward, while the 192–256 encoder numbers are a direct, easy path. Training takes the easy path."]))
    st.append(P("<b>The lesson:</b> \"dynamic\" is not the same as \"useful\". To improve predictions, a component must "
                "(1) bring information the model does not already have and (2) sit on a path the predictor actually uses."))

    # ---------- 6
    st += section(6, "The Time-to-Failure (TTF) Model and Its Failure")
    st.append(key("We made Koopman the predictor itself (\"fast-forward health until it hits the failure line\"), but "
                  "the projected health never reached the line, because a stable Koopman rule can only make things fade."))
    st.append(P("The idea", "h2"))
    st.append(P("If the predictor ignores Koopman, make Koopman <b>be</b> the predictor:"))
    st += bullets(["read the engine's current hidden state and turn it into a <b>health score</b>;",
                   "use the Koopman rule to <b>fast-forward</b> that state flight by flight, up to 125 flights;",
                   "<b>RUL = number of flights until the projected health crosses the failure line</b> (0)."])
    st.append(P("What happened", "h2"))
    st.append(fig(os.path.join(rep, "fd002", "ttf_health_trajectories.png"), "Figure 5. Projected health of 4 test "
                  "engines (FD002). None of the curves reaches the failure line at 0; they flatten out towards an "
                  "average level. Across all 253 test engines and 3 runs, 0% crossed.", 0.6))
    st.append(P("Accuracy was a little worse than KePIN (fair test 16.44 vs 15.98). The model fell back on reading the "
                "<i>current</i> health level (correlation −0.91 with RUL) instead of forecasting a crossing."))
    st.append(fig("06_fade_vs_grow.png", "Figure 6. Left: a stable Koopman rule (every mode shrinks by a factor below 1 "
                  "each flight) can only drift back to a resting level. Right: real wear builds up until it crosses "
                  "the failure line, which needs a mode that never fades.", 0.8))
    st.append(why("Koopman rules are built to be <b>stable</b> (all decay factors below 1), so predictions never "
                  "explode. But stability means every pattern <b>fades</b>. Wear is the opposite: it <b>accumulates</b>. "
                  "A rule that can only fade cannot draw a curve that climbs to failure. This physical mismatch "
                  "reappears later (section 12, \"future loss\")."))

    # ---------- 7
    st += section(7, "The Wear Model: Wear Accumulator and Failure Threshold")
    st.append(key("Adding one Koopman mode that never fades (the \"wear accumulator\") lets projected health climb to "
                  "the failure line, giving a model as accurate as KePIN that explains every prediction."))
    st.append(fig("07_wear_architecture.png", "Figure 7. The wear model. The encoder is KePIN's; the black-box predictor "
                  "is gone. RUL is read directly from the Koopman projection."))
    st.append(P("The fix in one line of maths", "h2"))
    st.append(P("<b>projected health after k flights = Koopman fading part(k) + v × k</b>, where <b>v ≥ 0</b> is a "
                "learned, per-engine <b>wear rate</b>. In Koopman terms, v × k comes from an extra mode with eigenvalue "
                "exactly 1 (an \"integrator\"), which never fades. RUL is the flight where the curve crosses 0, roughly "
                "<b>−current health ÷ wear rate</b>."))
    st.append(code("""
v = tf.nn.softplus(tf.tensordot(z_T, self.wear_w, 1) + self.wear_b)[:, None] + 1e-3   # wear rate >= 0
k = tf.range(TTF_HORIZON + 1, dtype=tf.float32)[None, :]                              # flights 0..125
health = health + v * k                                  # wear accumulator mode
logits = -(health[:, 1:] / v) / WEAR_TAU                 # crossing measured in flights""",
                   "dynamic_koopman/dk_model.py:193-198"))
    st.append(why("Measuring the crossing <b>in flights</b> (dividing by v) stops the model from cheating as the TTF "
                  "model did: the \"probability still alive\" only drops near the real crossing, not gradually from far away."))
    st.append(fig(os.path.join(rep, "fd002", "wear_explanations.png"), "Figure 8. Left: each prediction is literally "
                  "the point where projected health hits the failure line (dots). Right: the learned wear rate against "
                  "true remaining life for 253 test engines: faster wear means shorter life (correlation −0.91)."))
    st.append(table([["Dataset", "Our KePIN", "Wear model", "Paper's KePIN"],
                     ["FD001", "12.92", "13.43", "12.92"], ["FD002", "15.00", "14.87", "15.08"],
                     ["FD003", "11.56", "11.43", "11.42"], ["FD004", "17.11", "16.98", "17.04"]],
                    [0.25, 0.25, 0.25, 0.25]))
    st.append(P("<i>Ensemble of 3, paper's method. No difference is statistically significant: the wear model ties KePIN "
                "while explaining every prediction.</i>", "caption"))

    # ---------- 8
    st += section(8, "Wear Model Explainability and Validation")
    st.append(key("Three independent checks show the wear rate is real: it rises over every engine's life, it ignores "
                  "flying conditions, and it is driven by the sensors known to show wear."))
    st.append(P("Check 1: how the wear rate changes over time", "h2"))
    st.append(P("We ran the model at every flight of 52 engines it never trained on, from the first flight to failure."))
    st.append(fig(os.path.join(rep, "fd002_wear_xai", "life_trajectories.png"), "Figure 9. Top: predicted (blue) vs true "
                  "(dashed) remaining life. Bottom: the learned wear rate over the engine's life."))
    st += bullets(["The wear rate rose over life in <b>100%</b> of the 52 engines (median trend strength +0.92).",
                   "Predicted life fell in <b>100%</b> of them (median −0.92).",
                   "The wear rate has <b>two phases</b>: flat and low (about 0.05) while healthy, then a sharp climb (to "
                   "about 0.14) once degradation starts. The model <b>found the start of degradation by itself</b>."])
    st.append(P("Check 2: did it learn the flying-condition shortcut again?", "h2"))
    st.append(table([["What explains it? (R²)", "From true remaining life", "From the flying-condition mix"],
                     ["Signal A network (old, section 4)", "0.02–0.04", "0.85–0.88 (shortcut)"],
                     ["Wear rate (new)", "0.85–0.87", "−0.13 (no information)"]], [0.4, 0.3, 0.3], highlight_rows=(2,)))
    st.append(KeepTogether([P("Check 3: which sensors drive the wear rate?", "h2"),
                            fig(os.path.join(rep, "fd002_wear_xai", "wear_attribution.png"), "Figure 10. Integrated "
                                "Gradients: how much each input contributes to the wear rate (mean ± spread over 3 runs).")]))
    st.append(why(["The top drivers (<b>s11</b> compressor pressure, <b>s15</b> bypass ratio, <b>s4</b> turbine "
                   "temperature, s2, s17, s12) are the sensors known to show degradation in C-MAPSS, and the model "
                   "found them without being told. The 3 flying-condition settings together get only 7.6%.",
                   "s1 (fan inlet temperature) ranks high but depends mostly on the flying condition. Since the wear "
                   "rate shows no condition dependence (check 2), the model most likely uses s1 to judge the other "
                   "sensors <i>relative to the current condition</i>."]))
    st.append(P("Explanations held on every dataset: 85–100% of engines cross the failure line, and the wear rate "
                "correlates −0.88 to −0.95 with true remaining life on FD001–FD004."))

    # ---------- 9
    st += section(9, "Condition-Aware KePIN: the Final Architecture")
    st.append(key("We fixed, inside the architecture, the flying-condition problem found in section 4, and it beat "
                  "KePIN and the paper by 1.1–1.8 RMSE on FD002 and FD004, without hurting FD001 and FD003."))
    st.append(fig("09_condition_aware_architecture.png", "Figure 11. Original KePIN (left) vs condition-aware KePIN "
                  "(right). Only the red and orange parts changed; the encoder and predictor are KePIN's."))
    st.append(table([["Change", "Before (KePIN)", "After (new)", "Why"],
                     ["1. Condition-aware input layer (new)", "one global scaling for all 6 conditions",
                      "own learned scale & offset per condition", "removes the jumps caused by condition switching (section 10)"],
                     ["2. Smoothing moved (changed)", "EMA smoothing before the model", "EMA inside the model, after change 1",
                      "smoothing first blends different conditions together (section 10)"],
                     ["3. Koopman with control (new)", "Koopman rule on z", "z = s (wear state) + C × condition",
                      "intended to give Koopman a condition-free state (section 11)"]],
                    [0.22, 0.22, 0.24, 0.32]))
    st.append(P("Safety check: is the comparison fair?", "h2"))
    st.append(P("With the new parts set to neutral (scale 1, offset 0, C = 0), the model gives <b>exactly the same "
                "predictions as original KePIN</b> (difference 0.000006), and moving the smoothing inside the model "
                "changes the inputs by only 0.0000004. So any improvement comes <b>only</b> from the new parts."))
    st.append(fig("12_final_results.png", "Figure 12. Ensemble of 3 runs, paper's method. On FD002 and FD004 all 12 new runs beat KePIN "
                  "with the same seed (p = 0.003–0.05), and the NASA score (which punishes late predictions most) "
                  "improved: FD002 869 → 825, FD004 1265 → 1043. On FD001 and FD003 (one flying condition) "
                  "the changes are small and not significant, as expected."))

    # ---------- 10
    st += section(10, "Condition-Aware Normalisation and EMA Smoothing")
    st.append(key("Scaling each sensor separately for each flying condition, before smoothing, turns a jumpy signal "
                  "into a clean wear trend; that is where the accuracy gain comes from."))
    st.append(fig("10_condition_norm_real_data.png", "Figure 13. Real data: sensor s11 of engine 1 in FD002, coloured "
                  "by flying condition."))
    st.append(why(["<b>(a) The problem.</b> With one global scaling, the same sensor sits at a completely different "
                   "level in each flying condition. Every flight the value jumps between these levels, and the jumps "
                   "are far larger than the slow drift caused by wear.",
                   "<b>(b) The fix.</b> Scaling per condition asks \"how unusual is this reading <i>for this flying "
                   "condition</i>?\". The jumps vanish and the wear trend appears clearly, rising towards failure.",
                   "<b>(c) Order matters.</b> KePIN smooths each window first. When conditions switch every flight, "
                   "smoothing averages readings from different conditions into a meaningless mixture. Doing the "
                   "condition fix first and smoothing afterwards keeps the trend."]))
    st.append(KeepTogether([P("How it is built", "h2"), code("""
x, u_flat = inputs                                   # raw sensors + flying condition of every flight
u = tf.reshape(u_flat, [-1, T, n_conditions])        # one-hot condition per flight
x = x * (u @ gamma) + (u @ beta)                     # NEW: learned scale & offset PER CONDITION
x = ema_matrix @ x                                   # KePIN's EMA smoothing, now AFTER the condition step
z = self.encode(x)                                   # KePIN's encoder, unchanged""",
                   "simplified from dynamic_koopman/dk_model.py:155-162 (_condition_aware_call)")]))
    st += bullets(["<b>gamma, beta</b> (one value per condition and sensor) start as plain per-condition normalisation "
                   "and are then <b>learned</b> during training, so this is a trainable layer, not just preprocessing.",
                   "The EMA is a fixed matrix with KePIN's exact smoothing strength, checked to reproduce KePIN's "
                   "preprocessing to 0.0000004.",
                   "On FD001/FD003 (one flying condition) the layer reduces to a learned normalisation, so no big gain "
                   "is expected: results were 13.41 / 12.85 on FD001 (KePIN 12.92) and 11.17 / 11.07 on FD003 "
                   "(KePIN 11.56), none significantly different. It does not hurt where there is nothing to fix."])

    # ---------- 11
    st += section(11, "Koopman with Control")
    st.append(key("The idea was to split the hidden state into \"flight effect\" + \"wear state\", but explaining it "
                  "showed the \"wear state\" still contains the flying condition; the gain belongs to change 1."))
    st.append(fig("11a_koopman_control_idea.png", "Figure 14. Koopman with control, the standard physics form for a "
                  "system driven by an external input. The flight effect C × condition is subtracted, and the Koopman "
                  "rule and predictor work on the remainder s."))
    st.append(P("How we checked it", "h2"))
    st.append(P("We asked simple probes (a linear model and a random forest) how well each hidden state predicts "
                "(a) the window's flying conditions and (b) the remaining life, always on unseen engines (FD002, 3 runs)."))
    st.append(fig("11b_koopman_control_probe.png", "Figure 15. Left: flying-condition information (we want it low). "
                  "Right: remaining-life information (we want it high)."))
    st.append(why(["<b>The condition-aware input layer works exactly as intended:</b> it removes the flying-condition "
                   "information from KePIN's hidden state (0.26 → 0.00) and keeps all the remaining-life information (0.89).",
                   "<b>Koopman with control does not work as designed.</b> The input layer had already removed the "
                   "condition, so there was nothing left to subtract. The learned C × condition therefore <i>wrote "
                   "the condition back in</i> as a clean, simple signal (0.97), effectively a per-condition offset.",
                   "This is the second time explaining the model caught something RMSE alone would hide (the first "
                   "was Signal A). For the thesis, the improvement should be credited to the condition-aware input layer."]))

    # ---------- 12
    st += section(12, "Final Results, Ablation and Limitations")
    st.append(key("The condition-aware input layer is our main accuracy contribution; the wear model is our main "
                  "explainability contribution; and several Koopman ideas were tested and honestly ruled out."))
    st.append(fig(os.path.join(rep, "overview_vs_kepin.png"), "Figure 16. Every experiment compared with the original "
                  "KePIN (ensemble of 3, paper's method). Left of the line = better. Missing bars = not run on that dataset."))
    st.append(P("Everything we tried, in one table", "h2"))
    st.append(table([["Idea", "Result vs KePIN", "Why"],
                     ["Remove Koopman completely", "no significant change (all 4 datasets)", "accuracy comes from the encoder"],
                     ["Dynamic Koopman (condition / health / both)", "no improvement", "re-packaged known information; predictor ignored it"],
                     ["Train Koopman on each engine's real future (5–60 flights ahead)", "worse (FD001 14.02, FD002 15.79)",
                      "a fading rule cannot describe growing wear, so the shared hidden state was squeezed"],
                     ["Same, on 16 dedicated dims + growth term", "still worse (13.67, 15.57)", "the extra task competes with the main task"],
                     ["Time-to-failure (Koopman as predictor)", "slightly worse; never crosses", "a stable rule can only fade"],
                     ["Wear model (TTF + wear accumulator)", "ties on all 4 datasets", "accurate AND explains every prediction"],
                     ["<b>Condition-aware input layer</b>", "<b>−0.9 to −1.8 (FD002, FD004)</b>; ties on FD001/FD003", "<b>removes flying-condition noise before smoothing</b>"],
                     ["Koopman with control (on top)", "no reliable extra gain", "its \"wear state\" still carries the condition"]],
                    [0.36, 0.27, 0.37], highlight_rows=(7,)))
    st.append(P("What we learned", "h2"))
    st += bullets([
        "<b>Explainability found the real problem.</b> Explaining Signal A exposed the flying-condition shortcut; fixing "
        "it gave the accuracy gain.",
        "<b>KePIN's Koopman part does not drive its accuracy</b>: constant features, a predictor that ignores it, and a "
        "complete removal that changes nothing significantly.",
        "<b>Stable Koopman rules cannot represent growing wear</b>, which is why time-to-failure and future-training failed. "
        "A never-fading wear mode fixes it.",
        "<b>The paper's evaluation chooses its model with the test set</b>, which flatters its numbers by about 0.4–0.6."])
    st.append(P("Limitations (honest)", "h2"))
    st += bullets([
        "Condition-aware KePIN gives big gains only on FD002/FD004 (6 flying conditions); on FD001/FD003 it "
        "ties KePIN, because there is no condition noise to remove.",
        "Koopman with control adds no reliable accuracy and does not produce a condition-free state; it needs a "
        "redesign (for example, a penalty that keeps condition information out of s) or should be dropped.",
        "3 seeds per model, as in the paper: differences smaller than about 0.5 RMSE cannot be detected reliably.",
        "Recent results use the paper's method (test set picks the model) for comparability; a fair-test run of the "
        "final model would make the claim stronger.",
        "The accuracy gain comes from how the inputs are handled, not from Koopman itself."])
    st.append(P("Next steps", "h2"))
    st += bullets(["Decide the final model: condition-aware input alone, or with control (slightly better on 3 of 4 "
                   "datasets, but its \"wear state\" does not do what it was designed to do).",
                   "Put the wear model on top of condition-aware KePIN: the new accuracy <i>plus</i> per-prediction explanations.",
                   "Fair-test (clean) runs of the final model for the thesis."])

    # ---------- glossary
    st += [PageBreak(), P("13. Glossary", "h1")]
    st.append(table([["Term", "Plain meaning"],
                     ["RUL", "Remaining Useful Life: flights left before the engine fails."],
                     ["RMSE", "Typical prediction error in flights. Lower is better."],
                     ["NASA score", "An error measure that punishes late predictions (dangerous) more than early ones."],
                     ["Encoder", "The part of the network (convolution + LSTM + attention) that turns sensor data into a hidden state."],
                     ["Hidden state (z)", "The network's internal summary of the engine: a list of 96–128 numbers per flight."],
                     ["Koopman operator / rule (K)", "A learned matrix that moves the hidden state forward one flight: next = K × current."],
                     ["Mode / eigenvalue", "The Koopman rule splits into simple patterns (modes); each fades by its own factor per flight (eigenvalue)."],
                     ["Stable", "Every mode fades (factor below 1), so predictions never explode, but nothing can grow."],
                     ["Wear accumulator", "An extra mode with factor exactly 1 that adds the wear rate every flight; it never fades."],
                     ["Flying (operating) condition", "A combination of altitude, speed and throttle; FD002/FD004 have 6."],
                     ["Normalisation", "Rescaling a sensor so values are comparable (subtract an average, divide by a spread)."],
                     ["EMA smoothing", "A running average that reduces noise; each value is mostly the previous smoothed value plus a little of the new one."],
                     ["Seed / run", "One training from a random start. We use 3, like the paper."],
                     ["Ensemble", "Average the predictions of the 3 runs, then score once. The paper reports this."],
                     ["Paper's method vs fair test", "Paper: the test set chooses the final model. Fair: held-out training engines choose it."],
                     ["Ablation", "Remove one part and retrain, to measure whether it matters."],
                     ["p-value", "Chance that a difference this large happens by luck. Below 0.05 is usually called significant."],
                     ["Shapley values / Integrated Gradients", "Explainability methods that measure how much each input contributes to an output."],
                     ["Probe / R²", "A simple model that tests whether some information can be read from a hidden state; R² = 1 fully, 0 not at all."],
                     ["Shortcut", "When a model learns an easy pattern that happens to work, instead of the intended one."]],
                    [0.3, 0.7], bold_cols=(0,)))
    st.append(Spacer(1, 6))
    st.append(P("Code and data for everything in this guide are in <font name='DVM'>RCKePIN/dynamic_koopman/</font> "
                "(experiment log: README.md) and <font name='DVM'>RCKePIN/FULL_REPORT/REPORT.md</font>.", "caption"))

    doc = SimpleDocTemplate(OUT, pagesize=A4, leftMargin=18 * mm, rightMargin=18 * mm, topMargin=16 * mm,
                            bottomMargin=18 * mm, title="Making KePIN Better: An Easy Guide",
                            author="RC-KePIN project", subject="What we did and why")
    doc.build(st, onFirstPage=page_decor, onLaterPages=page_decor)
    print("written", OUT)


if __name__ == "__main__":
    build()
