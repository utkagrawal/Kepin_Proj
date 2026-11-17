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
    "box": ParagraphStyle("box", fontName="DV", fontSize=9.8, leading=14, textColor=INK, spaceAfter=4),
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
    return KeepTogether([t, Spacer(1, 8)]) if len(rows) <= 20 else t


def section(num, title):
    return [PageBreak(), P(f"{num}. {title}", "h1")]


def page_decor(canv, doc):
    canv.saveState()
    canv.setFont("DV", 8)
    canv.setFillColor(MUTED)
    canv.drawString(18 * mm, 10 * mm, "Making KePIN better: an easy guide to what we did and why (updated 10 Nov 2026)")
    canv.drawRightString(A4[0] - 18 * mm, 10 * mm, f"page {doc.page}")
    canv.restoreState()


# ---------------------------------------------------------------- Part 2 (22 Oct - 9 Nov 2026)
def part2(rep):
    st = []

    # ---------- 13
    st += section(13, "The New Question: Can the Koopman Rule Adapt to Engine Health?")
    st.append(key("Part 2 asks one thing: instead of one Koopman rule for the whole fleet, can each engine get its own "
                  "rule that depends on how healthy it is, and does that help or explain anything?"))
    st.append(P("The idea", "h2"))
    st.append(P("A Koopman rule moves the hidden state forward one step, and every pattern (mode) in it shrinks by its own "
                "factor ρ each step. In original KePIN there is <b>one</b> rule, so a brand-new engine and an engine about "
                "to fail are moved forward in exactly the same way. Our hypothesis was: a <b>healthy</b> engine changes "
                "slowly, so its rule should forget slowly (ρ close to 1); an engine <b>near failure</b> changes fast, so its "
                "rule should forget fast (smaller ρ). In short: <i>the latent space should shrink faster near failure</i>."))
    st.append(fig("13_adaptive_idea.png", "Figure 17. Left: a static rule, the same fading for every engine. Right: the "
                  "adaptive idea, where the fading speed depends on the engine's health.", 0.95))
    st.append(P("How a Koopman rule can be made to adapt", "h2"))
    st.append(P("Since section 5 we use an <b>eigen-parameterised</b> Koopman layer. It writes the rule as"))
    st.append(P("<b>K = P · diag(ρ<sub>k</sub> R(θ<sub>k</sub>)) · P<super>T</super></b>", "body"))
    st += bullets(["<b>P</b> is an orthogonal matrix: a fixed \"rotation\" of the 128-number hidden space into 64 mode pairs.",
                   "<b>ρ<sub>k</sub></b> (between 0 and 1) is how much mode k keeps per step; <b>θ<sub>k</sub></b> is how much it "
                   "turns (oscillates) per step.",
                   "Making K adaptive = giving each engine its own ρ<sub>k</sub> and θ<sub>k</sub>, while P stays shared. "
                   "The eigenvalues are then known exactly, so they are easy to read and plot."])
    st.append(why(["Part 1 taught us two conditions for any new part to matter: (1) it must bring <b>information the model "
                   "does not already have</b>, and (2) it must sit on a <b>path the predictor actually uses</b>.",
                   "Each step of Part 2 tests one way of meeting these two conditions. The table below is the map."]))
    st.append(table([["Section", "Where the Koopman rule sits", "What the rule adapts to", "Outcome"],
                     ["14", "condition-aware KePIN (extra features for the head)", "a health index from the window",
                      "ignored by the predictor (0.00 change)"],
                     ["15", "wear model (Koopman IS the predictor)", "the same health index", "K matters, adapting does not"],
                     ["16", "frozen / linear encoders", "the same health index", "gains, but not from health"],
                     ["17", "original KePIN + hypernetwork", "window, or engine age", "K starts to matter (with age)"],
                     ["18–19", "original KePIN + hypernetwork", "health from the engine's whole history",
                      "<b>11.77</b> (fair test); K carries health"]],
                    [0.11, 0.32, 0.26, 0.31], highlight_rows=(5,)))
    st.append(P("<b>Results in Part 2:</b> FD002 (6 flying conditions), 3 seeds, ensemble RMSE. Most experiments use the "
                "paper's method so they compare with Part 1; the final models (section 18) were re-run with the "
                "<b>fair test</b> as well.", "caption"))

    # ---------- 14
    st += section(14, "First Attempts: Three Adaptive Rules on Condition-Aware KePIN")
    st.append(key("We built a reliable health index without using RUL labels and three different adaptive rules on top of "
                  "it; none beat the static rule, and a knockout test showed the predictor ignores the rule completely."))
    st.append(P("Step 1: a health index h that really means health", "h2"))
    st.append(P("Every adaptive rule needs a number h that says how healthy the engine is (1 = healthy, 0 = near failure). "
                "When we let the network learn h freely, it collapsed to a constant (spread 0.000) because nothing told it "
                "what health means. We added one rule, the <b>mono loss</b>: take two windows of the same engine; the "
                "later one must have a lower h. It uses only the order of flights, never the RUL label."))
    st.append(code("""
# h_early: window at flight t, h_late: same engine at t + dt  (dt = 10..60 flights)
loss = tf.reduce_mean(tf.nn.relu(h_late - h_early + margin * frac))   # later must be lower, by a margin""",
                   "dynamic_koopman/regime_trainer.py:73"))
    st.append(table([["Check (FD002, 3 runs)", "h with mono loss (no RUL labels)"],
                     ["Correlation with true remaining life", "0.87–0.90"],
                     ["Explained by the flying-condition mix (R²)", "0.00 (no shortcut, unlike Signal A)"],
                     ["h falls over the life of the engine", "in 100% of the 260 training engines"],
                     ["Sensors that drive h (Integrated Gradients)", "s11, s15, s4, then s14, s17, s3: the known wear sensors"]],
                    [0.5, 0.5]))
    st.append(P("Signal A, done properly", "h2"))
    st.append(P("We also re-tested Signal A (section 4) with each sensor scaled <b>per flying condition</b> first. That "
                "removes the shortcut and turns its \"mean\" statistic into a real wear signal:"))
    st.append(table([["Signal A version", "R² of remaining life", "R² of the flying-condition mix"],
                     ["original (one global scaling)", "0.04 / 0.19", "0.50 / 0.69 (shortcut)"],
                     ["per-condition scaling, same 4 statistics", "0.55 / 0.63", "0.04 / 0.02"],
                     ["per-condition, mean + slope of every sensor (42)", "<b>0.82 / 0.85</b>", "−0.01 / 0.22"]],
                    [0.46, 0.25, 0.29]))
    st.append(P("Linear / random-forest R², tested on engines the model never saw. The last row is the seed of the "
                "engine-history features in section 18.", "caption"))
    st.append(callout("A bug we found and fixed", "Signal A first divided each sensor by its exact spread within a flying "
                      "condition. Some sensors barely move (spread 0.0001–0.001), but training adds noise of about 0.009, "
                      "up to 100× their spread, so in training the statistics were pure noise while at test time they were "
                      "clean. Fix: use the same floor (0.1) as the condition-aware layer. All earlier Signal A runs were "
                      "invalid and were re-run.", KEY_BG, KEY_EDGE))
    st.append(P("Step 2: three ways to adapt the rule", "h2"))
    st.append(table([["Adaptive rule", "How it works", "Ensemble RMSE", "What it learned"],
                     ["Static rule (+ control), baseline", "one K for all", "<b>13.87</b>", "-"],
                     ["Health gate", "decay of every mode shifted by h: lower h → faster fading", "14.17",
                      "a near-failure engine keeps only 0.1–8% of what a healthy one keeps after 10 flights"],
                     ["Mixture of regimes", "3 rules (healthy / middle / failing) blended by h", "14.17",
                      "collapses to 2 regimes: healthy (mean RUL ~100) and near failure (~30)"],
                     ["Local rule per engine", "eigenvalues fitted on each engine's own window", "13.97",
                      "modes seemed to fade more slowly near failure (see below)"]],
                    [0.22, 0.32, 0.14, 0.32]))
    st.append(P("Step 3: the knockout test", "h2"))
    st.append(P("A <b>knockout</b> switches one part off in a trained model, without retraining, and measures how much "
                "the error grows. We forced every engine to use the healthy rule (h = 1), then the failing rule (h = 0)."))
    st.append(table([["Knockout (FD002 test engines)", "Change in RMSE"],
                     ["Gate: every engine gets h = 1, or h = 0", "<b>0.00</b>"],
                     ["Mixture: every engine gets one regime", "±0.07"],
                     ["Local rule: every engine gets the fleet rule", "+0.02"]], [0.6, 0.4]))
    st.append(why(["The adaptive rule moves the hidden state z forward, but the predictor <b>already reads z directly</b>. "
                   "Anything the rule computes from z is information the predictor has anyway, so it learns to ignore it.",
                   "Side finding: the local rule suggested that modes fade <i>more slowly</i> near failure (\"critical "
                   "slowing down\", a known early-warning sign in physics). We checked: it came from the trend inside "
                   "the 30-flight window. After removing that trend the sensor noise looks the same at every age, so "
                   "there is no critical slowing down in FD002 and we do not claim it."]))
    st.append(P("<b>What it means:</b> to test adaptivity fairly, Koopman must sit on the path to the prediction. "
                "That is exactly the wear model (section 7)."))

    # ---------- 15
    st += section(15, "Adaptive Koopman Where Koopman Is the Predictor (Wear Model)")
    st.append(key("In the wear model the RUL comes only from the Koopman forecast, so Koopman clearly matters; but a single "
                  "static rule was as good as any adaptive one."))
    st.append(P("Set-up", "h2"))
    st.append(P("We combined the two Part 1 winners: the <b>wear model</b> (RUL = flights until the projected health "
                "crosses the failure line) and the <b>condition-aware input layer</b>. This gave <b>13.68</b> (score 748), "
                "the best result at that time. The only change between arms is the Koopman rule used for the forecast."))
    st.append(table([["Koopman rule in the wear model", "Ensemble RMSE", "NASA score", "vs static (95% range)"],
                     ["<b>static (one rule)</b>", "<b>13.68</b>", "<b>748</b>", "-"],
                     ["health gate", "14.17", "847", "−0.17 to +1.12"],
                     ["mixture of regimes", "14.04", "838", "−0.20 to +0.92"],
                     ["local rule per engine", "14.17", "798", "−0.01 to +1.02"],
                     ["wear rate from Signal A (bug fixed)", "14.38", "783", "+0.10 to +1.35 (worse)"]],
                    [0.4, 0.17, 0.15, 0.28], highlight_rows=(1,)))
    st.append(table([["Knockout (wear model)", "gate", "mixture", "local rule"],
                     ["Freeze the fading modes (Koopman part off)", "+1.5", "+3.0", "+2.3"],
                     ["Give every engine the healthy rule", "−0.01", "−0.00", "+0.15 (fleet rule)"]],
                    [0.46, 0.18, 0.18, 0.18]))
    st.append(why(["Here Koopman <b>does</b> matter: switching its fading modes off costs 1.5–3 RMSE. But making the rule "
                   "<b>different per engine</b> adds nothing: giving every engine the healthy rule changes the error by 0.00.",
                   "Three adaptive designs, two architectures, same answer: on FD002 one static rule is enough once the "
                   "encoder has done its job. Section 16 explains why."]))

    # ---------- 16
    st += section(16, "Why There Was Little to Adapt: the Encoder Straightens the Curve")
    st.append(key("Real wear speeds up near failure about 5×, but the trained encoder turns this into an almost straight "
                  "line (1.8×), so a single linear rule can already follow it."))
    st.append(P("Measuring the \"speed-up\" near failure", "h2"))
    st.append(P("For each layer of the encoder, we found the direction that best predicts the remaining flights, and "
                "compared how fast it moves in the <b>last 50 flights</b> with how fast it moves <b>100–150 flights "
                "before failure</b>. 1× means a straight line; 5× means it moves five times faster near the end."))
    st.append(fig("16_curvature_layers.png", "Figure 18. The cleaned sensors speed up 5× near failure (100% of engines). The "
                  "first ReLU removes the most; an encoder trained on the (straight-line, capped) RUL label ends at 1.8×, "
                  "a self-supervised encoder (trained only to rebuild its input) keeps 2.8×."))
    st.append(P("Testing adaptive Koopman on encoders that keep the curve", "h2"))
    st.append(table([["Encoder (speed-up it keeps)", "Static rule", "Gate (h per window)", "h re-read every forecast step"],
                     ["RUL-trained, deep (1.8×)", "13.68", "14.17", "13.97"],
                     ["frozen self-supervised (2.8×)", "18.61", "<b>16.73</b>", "18.55"],
                     ["linear, no activations (5×)", "19.25", "<b>16.32</b>", "19.25"]],
                    [0.34, 0.18, 0.24, 0.24]))
    st.append(why(["On the curved encoders the gate gives big gains (−1.9 and −2.9). But the gate's h was <b>not</b> health: "
                   "it had no relation to remaining life (R² ≈ 0) and fell over life in only about half of the engines. The arm "
                   "whose h <i>did</i> track health (≈ 0.8) gave no gain.",
                   "So the gain most likely came from extra capacity (a small non-linear network added to an otherwise linear "
                   "model), not from adapting to health. We report it as a confound, not a result."]))
    st.append(P("<b>What it means:</b> inside the window, a good encoder already straightens the degradation, so an "
                "adaptive rule fed from the window has nothing new to add. The health information has to come from "
                "<b>outside the window</b>. Sections 17–18 do exactly that."))

    # ---------- 17
    st += section(17, "Hypernetworks: Building a Koopman Rule for Every Engine")
    st.append(key("A hypernetwork is a small network that outputs the settings of another part of the model. We use one "
                  "to output each engine's Koopman eigenvalues; this finally makes the 12 Koopman features differ between "
                  "engines in original KePIN."))
    st.append(P("Where the idea comes from", "h2"))
    st.append(P("<b>Hyper-LR-PINN</b> (Cho et al., NeurIPS 2023) solves a family of physics equations with one network. "
                "Its weight matrices are written as U · diag(s) · V<super>T</super>: the bases U, V are shared, and a "
                "hypernetwork produces only the small diagonal s for each equation parameter. A ReLU on s switches parts "
                "off (adaptive rank). Our Koopman layer already has exactly this shape:"))
    st.append(table([["Hyper-LR-PINN", "Our hypernetwork Koopman"],
                     ["shared bases U, V", "shared orthogonal basis P (the same for all engines)"],
                     ["diagonal s(μ) made by the hypernetwork", "decay ρ<sub>k</sub> and rotation θ<sub>k</sub> of each mode, made per engine"],
                     ["equation parameter μ", "what we know about the engine (window, age, or health)"],
                     ["ReLU on s → adaptive rank", "gate g<sub>k</sub> ∈ [0, 1] → modes switched off per engine"]],
                    [0.42, 0.58]))
    st.append(P("The architecture we use", "h2"))
    st.append(fig("17_hyper_architecture.png", "Figure 19. Hypernetwork Koopman in original KePIN (final version, section "
                  "18). The encoder, the predictor and the Koopman training losses are original KePIN. New parts: the "
                  "health branch (purple) and the hypernetwork (orange)."))
    st.append(code("""
def hyper_params(self, z, signal=None):
    ctx = [mean_t(z), z[:, -1]] if self.hyper_use_window else []   # window context (section 17 runs)
    if self.hyper_ctx_dim: ctx.append(signal)                     # h, or h + age (section 18 runs)
    out = self.hyper_out(self.hyper_hidden(tf.concat(ctx, -1)))   # Dense 64 ReLU -> Dense (zero init)
    ds  = 2.0 * tf.tanh(out[:, :m])                               # change of each mode's decay
    dth = 0.5 * tf.tanh(out[:, m:2*m])                            # change of each mode's rotation
    log_rho = tf.math.log_sigmoid(self.s + ds)                    # rho_k = sigmoid(s_k + ds_k)
    if self.hyper_rank:                                           # adaptive rank
        g = tf.clip_by_value(tf.nn.relu(1.0 + out[:, 2*m:]), 0.0, 1.0)
        log_rho = log_rho + tf.math.log(g + 1e-6)                 # g = 0 -> mode switched off
    return log_rho, self.theta + dth, g""", "dynamic_koopman/eigen_koopman.py:192-207 (simplified)"))
    st += bullets(["<b>Size:</b> one hidden layer of 64 units; the output layer has 128 numbers (decay + rotation for 64 "
                   "modes), or 192 with the on/off gates.",
                   "<b>Starts as the static rule:</b> the output layer starts at zero, so at the beginning of training every "
                   "engine gets the shared rule (s, θ); the hypernetwork only learns <i>differences</i>.",
                   "<b>Bounded changes:</b> tanh keeps the change in decay within ±2 (in logit units) and in rotation within "
                   "±0.5 rad, so the rule stays stable (ρ &lt; 1).",
                   "<b>New road to the prediction:</b> the eigenvalues now differ per engine, so KePIN's 12 spectral "
                   "features (identical for all engines in section 2) now carry per-engine information."])
    st.append(P("First runs: what should the hypernetwork read?", "h2"))
    st.append(table([["Hypernetwork reads", "Average of 3 runs", "Ensemble RMSE", "NASA score"],
                     ["nothing (static rule, original KePIN)", "15.34", "14.91", "951"],
                     ["the window (average and last hidden state)", "15.13", "14.72", "854"],
                     ["window + engine age (flights flown so far)", "14.81", "14.39", "972"],
                     ["window + age, with on/off gates", "14.43", "<b>14.04</b>", "<b>818</b>"]],
                    [0.46, 0.18, 0.18, 0.18], highlight_rows=(4,)))
    st.append(table([["Knockout (change in RMSE)", "window only", "+ age", "+ age + gates"],
                     ["same rule for every engine", "0.00", "+2.9", "+2.4"],
                     ["12 spectral features held fixed", "0.00", "+2.9", "+2.2"],
                     ["age shuffled between engines", "-", "+5.9", "+7.7"]],
                    [0.4, 0.2, 0.2, 0.2]))
    st.append(why(["With the <b>window only</b>, the per-engine rule is again ignored (0.00): same story as section 14.",
                   "With <b>age</b>, the predictor suddenly depends on the per-engine rule (+2.4 to +2.9 when it is removed). "
                   "Age comes from <b>outside the window</b>; the predictor cannot see it any other way.",
                   "<b>The rule we found:</b> Koopman contributes only when its spectrum carries information the predictor "
                   "cannot already see. So the next step is to give the hypernetwork real <b>health</b> information from "
                   "outside the window, and to check with a control whether Koopman is needed or just passes it along."]))

    # ---------- 18
    st += section(18, "Feeding the Health Regime to the Hypernetwork")
    st.append(key("A health index built from the engine's whole history drives the hypernetwork; error falls to 11.77 "
                  "(fair test, from 15.34), but a control that gives the same health index straight to the predictor "
                  "does just as well."))
    st.append(P("Engine-history features (outside the window)", "h2"))
    st.append(P("Health is a property of the engine's whole life, not of the last 30 flights. For every window we compute "
                "42 numbers from <b>all flights so far</b> (only the past, so they are also available at test time). "
                "Each sensor is first scaled with its own flying condition's average and spread (the Signal A lesson), then:"))
    st += bullets(["<b>drift since new</b> (21 numbers): average of the last 10 flights minus average of the first 10 flights;",
                   "<b>current rate</b> (21 numbers): slope over the last 30 flights."])
    st.append(code("""
S = (sensors - mu[condition]) / sd[condition]          # per-condition scaling (training statistics)
base = S[:10].mean(0)                                  # the engine when new
for e in window_ends:                                  # only flights 1..e are used
    drift = S[e-10:e].mean(0) - base                   # how far each sensor has moved since new
    rate  = 10 * (t @ S[e-30:e]) / sum(t**2)           # how fast it is moving now (per 10 flights)
    features.append(concat([drift, rate]))             # 42 numbers""",
                   "dynamic_koopman/dk_data.py:64-89 (_history_features, simplified)"))
    st.append(P("On their own these 42 numbers explain <b>81%</b> of the variation in remaining life (linear R²), "
                "against <b>49%</b> for age alone."))
    st.append(P("From 42 numbers to one health index h", "h2"))
    st.append(P("A small <b>health net</b> (Dense 32 with ReLU → Dense 1 with sigmoid) turns them into h between 0 and 1. "
                "It is trained with the <b>mono loss</b> only (section 14), never with RUL labels. Result: h correlates "
                "<b>0.92–0.94</b> with true remaining life in every run."))
    st.append(P("Three arms", "h2"))
    st.append(table([["Arm", "What builds the Koopman rule", "What the predictor sees", "Question"],
                     ["A", "hypernetwork reads <b>only h</b>", "window + 12 spectral features",
                      "can health, through K alone, help?"],
                     ["B", "hypernetwork reads <b>h + age</b>, with on/off gates", "window + 12 spectral features",
                      "best version"],
                     ["C (control)", "<b>static</b> rule", "window + 12 features <b>+ h + age directly</b>",
                      "is K needed, or only a pipe?"]],
                    [0.13, 0.32, 0.3, 0.25]))
    st.append(P("In A, the hypernetwork sees nothing but h, so every change in the rule is caused by health by design. "
                "C uses a small hook that adds [h, age] to the predictor's input:"))
    st.append(code("""
head_input = tf.concat([pooled, spec_feats_batch], -1)          # original KePIN predictor input
extra = getattr(self, "_head_extra", None)                       # arm C: [h, age]
if extra is not None:
    head_input = tf.concat([head_input, tf.cast(extra, tf.float32)], axis=-1)""",
                   "dynamic_koopman/kepin_core/kepin_model.py:316-319"))
    st.append(P("Results", "h2"))
    st.append(fig("18_hyper_results.png", "Figure 20. Ensemble RMSE on FD002 (3 seeds). Blue: paper's method; orange: fair "
                  "test. The fair test is about 0.4–0.7 higher everywhere, but the order does not change."))
    st.append(table([["Model (FD002)", "Paper's method: runs / ensemble / score", "Fair test: runs / ensemble / score"],
                     ["original KePIN, static rule", "15.47, 15.08, 15.47 / 14.91 / 951", "15.44, 15.84, 16.09 / 15.34 / 1018"],
                     ["A: rule from h only", "13.31, 13.65, 13.19 / 13.09 / 668", "14.17, 13.78, 14.00 / 13.70 / 847"],
                     ["<b>B: rule from h + age, gates</b>", "11.58, 11.33, 11.44 / <b>11.03</b> / 543",
                      "12.39, 12.48, 12.12 / <b>11.77</b> / <b>610</b>"],
                     ["C: static rule, h + age to predictor", "11.62, 11.10, 11.23 / 11.04 / <b>465</b>",
                      "12.96, 12.06, 11.98 / 11.99 / 636"]],
                    [0.32, 0.34, 0.34], highlight_rows=(3,)))
    st.append(table([["Difference (fair test, bootstrap over 253 engines)", "Ensemble RMSE", "95% range"],
                     ["A − original KePIN", "−1.64", "−3.31 to −0.04"],
                     ["B − original KePIN", "−3.57", "−5.21 to −1.96"],
                     ["C − original KePIN", "−3.34", "−4.96 to −1.67"],
                     ["<b>B − C</b>", "<b>−0.22</b>", "<b>−0.76 to +0.32 (a tie)</b>"]],
                    [0.5, 0.2, 0.3]))
    st.append(why(["<b>The health information is the big win:</b> 15.34 → 11.77 under the fair test, the best result of the "
                   "whole project.",
                   "<b>Through Koopman alone it still helps:</b> arm A, where health can reach the prediction only through "
                   "the Koopman rule, beats original KePIN by 1.6.",
                   "<b>But Koopman is not needed for the gain:</b> B and C tie (paper's method: −0.01, range −0.56 to +0.52; "
                   "fair test: −0.22, range −0.76 to +0.32). The hypernetwork Koopman is a working pipe for the health "
                   "information, not an extra source of accuracy. We said in advance we would report it this way if they tied."]))

    # ---------- 19
    st += section(19, "Reading the Adaptive Koopman Rule (Arm A)")
    st.append(key("In arm A the Koopman rule carries all of the health information, and it changes over an engine's life "
                  "in a clear, consistent way, but in the opposite direction to our first idea."))
    st.append(P("Knockout: does the rule really carry health?", "h2"))
    st.append(table([["Knockout, fair-test models (runs 0 / 1 / 2)", "RMSE", "Change"],
                     ["normal", "14.17 / 13.78 / 14.00", "-"],
                     ["same rule for every engine (average rule)", "30.23 / 30.20 / 28.98", "+15.2"],
                     ["h shuffled between engines", "39.99 / 41.06 / 38.62", "+25.9"],
                     ["12 spectral features held fixed", "30.20 / 30.32 / 29.36", "+15.3"]],
                    [0.45, 0.33, 0.22]))
    st.append(P("Compare with section 2 (0.00) and section 14 (0.00): here the rule is essential. The paper's-method "
                "models give the same picture (+15, +26, +15)."))
    st.append(P("How the rule changes over life", "h2"))
    st.append(fig(os.path.join(rep, "hyper_health", "eig_hyper_hh_mono_clean.png"),
                  "Figure 21. Arm A, fair-test run 0, every 4th training engine over its full life. Left: average decay factor "
                  "ρ per step. Middle: number of slow modes (ρ > 0.3). Right: each mode's ρ against the health index h."))
    st += bullets(["The average decay factor rises from about <b>0.15</b> while healthy to about <b>0.36</b> at failure; "
                   "its correlation with h is −0.96 to −0.99 in all three runs.",
                   "Slow modes (ρ > 0.3): <b>0–1</b> while healthy, <b>49–64</b> in the last 30 flights, in every run, under "
                   "both protocols.",
                   "The change starts about 100 flights before failure, where degradation becomes visible in the sensors."])
    st.append(why(["<b>Opposite to our hypothesis.</b> We expected the latent space to shrink <i>faster</i> near failure. "
                   "Instead the rule squeezes it hardest while the engine is healthy and keeps more of it near failure. A "
                   "natural reading: a healthy engine has little going on, so the rule throws most of the state away; a "
                   "failing engine has more to say, so the rule keeps it.",
                   "<b>Honest caveat.</b> The decay factors are small: most modes lose half their size in under one step (median half-life 0.3–0.7 steps). So "
                   "the spectrum works as a readable <b>health code</b> that the predictor reads, more than as dynamics "
                   "played out over many steps.",
                   "<b>Arm B is not readable this way:</b> the direction in which its rule follows h flips between seeds "
                   "(+0.86, −0.64, +0.04), because age and the gates offer other ways to encode the same information."]))

    # ---------- 20 all datasets
    import glob as _glob
    import json as _json
    import numpy as _np

    def _ens(p, ds, arm):
        fs = sorted(_glob.glob(os.path.join(DK, "runs", p, ds, arm, "run*_predictions.npz")))
        d = [_np.load(f) for f in fs]
        y = d[0]["y_true"].ravel(); pr = _np.mean([x["y_pred"].ravel() for x in d], 0)
        sc = float(_np.sum(_np.where(pr - y < 0, _np.exp(-(pr - y) / 13) - 1, _np.exp((pr - y) / 10) - 1)))
        return y, pr, float(_np.sqrt(_np.mean((pr - y) ** 2))), sc, len(fs)

    def _boot(a, b, seed=0):
        (y, pa), (_, pb) = a, b
        ea, eb = (pa - y) ** 2, (pb - y) ** 2; n = len(y); rng = _np.random.default_rng(seed)
        dd = [_np.sqrt(ea[i].mean()) - _np.sqrt(eb[i].mean()) for i in (rng.integers(0, n, n) for _ in range(4000))]
        return _np.sqrt(ea.mean()) - _np.sqrt(eb.mean()), _np.percentile(dd, 2.5), _np.percentile(dd, 97.5)

    DSS = ["FD001", "FD002", "FD003", "FD004"]
    ARMS = [("svd_static", "KePIN"), ("eig_hyper_hh_mono", "A"), ("eig_hyper_hh_age_rank_mono", "B"),
            ("eig_static_hh_hctx_mono", "C")]
    R = {(p, ds, n): _ens(p, ds, arm) for p in ("clean", "paper") for ds in DSS for arm, n in ARMS}
    fmt = lambda r: f"{r[2]:.2f} / {r[3]:.0f}" + ("" if r[4] == 3 else f" ({r[4]} seed{'s' if r[4] > 1 else ''})")
    ci = lambda t: f"{t[0]:+.2f} ({t[1]:+.2f} to {t[2]:+.2f})"

    st += section(20, "Results on All Four Datasets")
    st.append(key("The same three arms on FD001, FD003 and FD004 give the same picture as FD002: history health gives a "
                  "large gain everywhere (biggest on FD004), and the hypernetwork Koopman (B) ties the control (C) "
                  "on every dataset."))
    st.append(P("Before training we checked the 42 history features on each dataset: they line up one-to-one with the "
                "training windows and explain 81% (FD001), 81% (FD002), 82% (FD003) and 83% (FD004) of the variation in "
                "remaining life of the test engines (simple linear fit). So the health information is equally strong "
                "everywhere."))
    st.append(fig("29_results_all.png", "Figure 22. Ensemble RMSE of the four models on all datasets (3 seeds each). Left: "
                  "fair test. Right: paper's method."))
    for p, title in [("clean", "Fair test"), ("paper", "Paper's method")]:
        rows = [[f"{title}: RMSE / NASA score", "original KePIN", "A: health only", "B: health + age", "C: control"]]
        for ds in DSS:
            rows.append([ds] + [fmt(R[(p, ds, n)]) for _, n in ARMS])
        st.append(table(rows, [0.2, 0.2, 0.2, 0.2, 0.2], bold_cols=(0,)))
    rows = [["Fair test, bootstrap (95% range)", "A − KePIN", "B − KePIN", "C − KePIN", "B − C"]]
    for ds in DSS:
        g = lambda n: R[("clean", ds, n)][:2]
        rows.append([ds, ci(_boot(g("A"), g("KePIN"))), ci(_boot(g("B"), g("KePIN"))), ci(_boot(g("C"), g("KePIN"))),
                     "<b>" + ci(_boot(g("B"), g("C"))) + "</b>"])
    st.append(table(rows, [0.16, 0.21, 0.21, 0.21, 0.21], bold_cols=(0,)))
    st.append(why(["<b>History health helps on every dataset.</b> B and C beat original KePIN by about 2 to 5 RMSE, with the "
                   "biggest gain on FD004, the hardest dataset (17.8 → 12.6–12.9 on the fair test; NASA score 1658 → "
                   "about 755). On FD003 the gain (about −2) is just short of significance, because only 100 test engines "
                   "make the 95% range wide.",
                   "<b>B ties C on all four datasets</b> (every B − C range includes 0). The hypernetwork Koopman is a "
                   "working pipe for the health information, on every dataset, not an extra source of accuracy.",
                   "<b>Arm A</b> (health only through Koopman) gains clearly on the multi-condition datasets (FD002 −1.7, "
                   "FD004 −2.6) but not on FD001/FD003. With the window alone the encoder already tracks wear well "
                   "there, so the narrow health code adds little."]))
    st.append(callout("A caveat we found", "In 9 of the 72 trained models the health number came out <b>flipped</b>: it rises "
                      "towards failure (correlation with remaining life about −0.9), so it measures \"damage\" instead of "
                      "\"health\". Most are in arm C on FD003/FD004. Accuracy is unaffected (the predictor can read either "
                      "direction), but for explanations the direction should be fixed, for example by anchoring h near 1 "
                      "on each engine's first flights. All arm A fair-test models, used in the next two sections, have the "
                      "correct direction.", KEY_BG, KEY_EDGE))

    # ---------- 21 easy explainer
    st += section(21, "How a Hypernetwork Makes Koopman Adaptive (the Easy Version)")
    st.append(key("A hypernetwork is a \"settings writer\": it reads how healthy an engine is and writes a personal "
                  "Koopman rule for that engine."))
    st.append(P("Step by step", "h2"))
    st += bullets(["<b>The hidden state</b> is a list of 128 numbers that summarises the engine's last 30 flights.",
                   "<b>A Koopman rule</b> says how that list changes from one step to the next. It splits the list into 64 "
                   "simple patterns (\"modes\"). Each pattern has two settings: a <b>fade speed ρ</b> (how much of it is "
                   "kept per step; 1 = kept forever, 0 = gone at once) and a <b>turn speed θ</b> (how much it rotates).",
                   "<b>In original KePIN</b> there is one rule with one set of 128 settings, the same for every engine.",
                   "<b>With a hypernetwork</b> the 64 patterns (the directions P) stay the same for every engine, but the "
                   "settings are <b>written fresh for every engine</b> by a small network that reads the engine's health "
                   "number h."])
    st.append(fig("22_hyper_concept.png", "Figure 23. The hypernetwork reads the health number and turns the 64 dials of "
                  "the Koopman rule. The directions P are shared; only the dial settings differ between engines."))
    st.append(P("What the generated rule looks like", "h2"))
    st.append(P("A standard way to draw a Koopman rule is to put each mode as a dot in a circle: the <b>distance from the "
                "centre is the fade factor ρ</b> (edge = never fades, centre = fades at once) and the angle is the turn "
                "speed. Here is the rule the hypernetwork wrote for the same FD002 engine at two moments of its life:"))
    st.append(fig("23_eig_circle.png", "Figure 24. Left: the engine is healthy; the hypernetwork pulls almost every mode "
                  "towards the centre. Right: the same engine at failure; most modes move out, beyond the dotted circle "
                  "(ρ = 0.3).", 0.9))
    st.append(why(["<b>Why this makes Koopman \"adaptive\":</b> the rule is no longer fixed. Every engine, at every moment, "
                   "gets its own rule, and the only thing that decides the rule is the health number.",
                   "<b>Why we built it this way:</b> keeping the directions P shared and only changing the dials (as in "
                   "Hyper-LR-PINN) keeps the rule simple, stable (all dots stay inside the circle) and easy to read: we can "
                   "always point at a mode and say how fast it fades for this engine.",
                   "<b>Why it now matters for the prediction:</b> KePIN's predictor reads 12 numbers computed from the dots "
                   "(fade speeds, turn speeds, size of the biggest dot). In original KePIN those 12 numbers were identical "
                   "for every engine (section 2). Now they differ, and they carry the engine's health."]))

    # ---------- 22 explainability
    st += section(22, "How This Helps Explain the Engine")
    st.append(key("Because the health-only Koopman rule (arm A) is built from one health number, we can read the "
                  "model's reasoning in four simple checks, and all four hold on every dataset."))
    exj = {ds: _json.load(open(os.path.join(rep, "hyper_explain", f"{ds}_clean.json"))) for ds in DSS}
    av = lambda ds, k: _np.mean([r[k] for r in exj[ds]["runs"]])
    rng_ = lambda ds, k: (min(r[k] for r in exj[ds]["runs"]), max(r[k] for r in exj[ds]["runs"]))
    st.append(P("Check 1: is the health number really health?", "h2"))
    st.append(fig("26_h_trajectories.png", "Figure 25. The health number h of six training engines per dataset, from their "
                  "first flight to failure (arm A, fair-test run 0)."))
    st.append(P("The health number was trained <b>without RUL labels</b> (only \"later must be lower\"). Still, it falls "
                "over the life of <b>100% of engines</b> on all four datasets, and correlates "
                + ", ".join(f"{av(ds, 'h_corr_rul_capped'):.2f} ({ds})" for ds in DSS) +
                " with the true remaining life. It stays high and flat while the engine is healthy and drops in the last "
                "100 or so flights, which is how C-MAPSS degradation behaves."))
    st.append(P("Check 2: does the Koopman rule follow health, the same way everywhere?", "h2"))
    st.append(fig("25_life_all.png", "Figure 26. Averages over training engines, by flights before failure; the shaded band "
                  "is the spread over 3 seeds. The same three-step story on every dataset."))
    rows = [["Arm A, fair test (3 seeds)", "corr(average ρ, h)", "average ρ: healthy → near failure",
             "slow modes (ρ > 0.3): healthy → near failure"]]
    for ds in DSS:
        a, b = rng_(ds, "corr_mean_rho_h")
        rows.append([ds, f"{a:.2f} to {b:.2f}", f"{av(ds, 'mean_rho_healthy'):.2f} → {av(ds, 'mean_rho_near_failure'):.2f}",
                     f"{av(ds, 'slow_modes_healthy'):.0f} → {av(ds, 'slow_modes_near_failure'):.0f} of 64"])
    st.append(table(rows, [0.22, 0.22, 0.28, 0.28], bold_cols=(0,)))
    st.append(P("Healthy = more than 150 flights before failure; near failure = the last 30 flights. In all 12 models the "
                "rule keeps <b>more</b> of the hidden state as health falls: modes fade less and more modes stay slow.",
                "caption"))
    st.append(P("Check 3: does the prediction really go through the Koopman rule?", "h2"))
    st.append(fig("27_knockout_all.png", "Figure 27. Knockout of arm A on every dataset: give every engine the same (average) "
                  "rule, hold the 12 Koopman features fixed, or shuffle the health numbers between engines."))
    st.append(P("Breaking the Koopman path costs <b>8 to 16 RMSE</b>; shuffling health costs <b>13 to 26</b>. In original "
                "KePIN the same tests gave 0.00 (section 2). Here every bit of the health information the predictor "
                "uses travels through the Koopman rule."))
    st.append(P("Check 4: the health dial (a what-if explanation)", "h2"))
    st.append(P("Because health is a single number, we can ask \"what if this engine were healthier, or sicker?\". We kept "
                "each engine's 30 flights fixed, fed in a chosen health number from 0 to 1 instead of the real one, and "
                "watched the prediction and the rule:"))
    st.append(fig("24_health_dial.png", "Figure 28. FD002, five test engines (dots = their real health number). Left: "
                  "predicted RUL as we turn the health dial. Right: number of slow modes in the generated rule."))
    st.append(why(["<b>The prediction follows the dial smoothly</b>, for every engine and every dataset (h = 0 → 1 moves "
                   "the prediction by roughly 35–95 flights). The ordering between engines stays: their own 30 flights "
                   "still matter, through the encoder.",
                   "<b>The rule changes in two ways.</b> Its overall fade factor shifts gradually with h (this is what the "
                   "12 features pass on), and below a low health level (h ≈ 0.25 on FD002) a large group of modes switches from \"fade at once\" to "
                   "\"slow\". So the rule has a clear <b>failure regime</b>: when health drops low, the latent space "
                   "suddenly opens up.",
                   "<b>This is a per-engine explanation:</b> for any engine we can say \"its health number is 0.2, so "
                   "its Koopman rule keeps 50 slow modes, and that is why its predicted life is short\"."]))
    st.append(fig("28_mode_heatmaps.png", "Figure 29. Fade factor of every mode against health (FD002 and FD004, run 0). "
                  "Almost all modes brighten together as h falls; one or two modes stay near ρ = 1 at every health "
                  "level (a 'memory' mode), and a few stay near 0.", 0.85))
    st.append(P("What the explanation can and cannot tell us", "h2"))
    st.append(table([["We CAN say", "We CANNOT (yet) say"],
                     ["how healthy each engine is, from its whole history, without RUL labels",
                      "which physical part of the engine each Koopman mode stands for"],
                     ["how the Koopman rule changes with health, the same way on all four datasets",
                      "that the rule is a multi-step forecast: most modes still fade within one step"],
                     ["that the prediction depends on health only through the Koopman rule (arm A)",
                      "that the Koopman route is more accurate than giving health to the predictor directly (B ties C)"],
                     ["what the prediction would be for a healthier or sicker engine (health dial)",
                      "that engines fade faster near failure: the data shows the opposite"]], [0.5, 0.5]))

    # ---------- 23
    st += section(23, "Where We Stand Now")
    st.append(key("History-based health information gives the best accuracy on all four datasets; a hypernetwork "
                  "Koopman rule driven by that health is a readable, consistent explanation of the model; Koopman is "
                  "not needed for the accuracy itself."))
    st.append(P("Everything in Part 2, in one table", "h2"))
    st.append(table([["Idea", "Result", "Why"],
                     ["Health index with the mono loss (no labels)", "falls in 100% of engines; corr 0.87–0.94", "the order of flights is enough to define health"],
                     ["Adaptive rule on condition-aware KePIN", "no gain; knockout 0.00", "the predictor already reads the hidden state"],
                     ["Adaptive rule in the wear model", "no gain; K matters (+1.5–3)", "one static rule is enough on FD002"],
                     ["Encoders that keep the curve", "gains, but h is not health", "extra capacity, not adaptation"],
                     ["Hypernetwork, window context", "14.72; knockout 0.00", "no new information"],
                     ["Hypernetwork, age context", "14.04; knockout +2.4", "age is new information"],
                     ["<b>Hypernetwork, history health (B)</b>", "<b>−2 to −5 RMSE vs KePIN on all 4 datasets</b>", "<b>health from outside the window</b>"],
                     ["Same information straight to the predictor (C)", "ties B on all 4 datasets", "the gain is the information"],
                     ["Health-only hypernetwork (A)", "knockout +8 to +16; same pattern on all 4", "Koopman carries all the health; readable"]],
                    [0.36, 0.34, 0.3], highlight_rows=(7,)))
    st.append(P("What we can claim", "h2"))
    st += bullets(["<b>Engine-history health</b> (per-condition drift since new and current rate, turned into one label-free "
                   "health number) improves KePIN on all four C-MAPSS datasets, by 2 to 5 RMSE under the fair test.",
                   "A <b>hypernetwork Koopman operator</b> driven only by that health number makes Koopman essential in KePIN "
                   "(+8 to +16 RMSE when removed, against 0.00 in original KePIN).",
                   "The generated operator is <b>readable and consistent</b>: on every dataset and seed its modes fade less "
                   "and more modes stay slow as health falls, with a clear failure regime below h ≈ 0.25.",
                   "<b>The rule behind all of it:</b> an adaptive Koopman operator helps only when it carries information the "
                   "predictor cannot otherwise see."])
    st.append(P("Limitations (honest)", "h2"))
    st += bullets(["The adaptive Koopman does not beat giving the predictor the same health information directly (B ties C on "
                   "all four datasets).",
                   "Most half-lives are below one step, so the operator acts as a health code more than as multi-step dynamics.",
                   "The \"faster forgetting near failure\" hypothesis is not supported; the data shows the opposite on every dataset.",
                   "The health number came out flipped in 9 of 72 models; its direction should be anchored.",
                   "3 seeds per model; differences below about 0.5 RMSE cannot be detected. FD001/FD003 have only 100 test "
                   "engines, so their ranges are wide.",
                   "The history features are hand-designed (drift and rate); their per-condition scaling uses all training "
                   "engines, including the fair test's validation engines (no labels involved)."])
    st.append(P("Possible next steps", "h2"))
    st += bullets(["Anchor the direction of h (e.g. h ≈ 1 on each engine's first flights) so every model gives a health, "
                   "not a damage, number.",
                   "Make Koopman <b>necessary</b>: e.g. the wear model's forecast with a hypernetwork rule driven by history "
                   "health, so the RUL can only come from the generated dynamics.",
                   "Thesis framing: \"adaptive Koopman operators contribute when, and only when, they carry information "
                   "from outside the encoder's view\", with arm A as the explainability case study."])
    return st


# ---------------------------------------------------------------- content
def build():
    sig = os.path.join(REPO, "RCKePIN_Signal_A", "results", "condition_net_interpretability_fd002")
    rep = os.path.join(DK, "reports")
    st = []

    # ---------- cover
    st += [Spacer(1, 12 * mm), P("Making KePIN Better", "title"),
           P("An easy guide to what we did, what we found, and <b>why</b>", "subtitle"), Spacer(1, 6 * mm),
           P("Remaining-useful-life prediction for jet engines (NASA C-MAPSS) · 18 October – 10 November 2026", "subtitle"),
           Spacer(1, 8 * mm)]
    st.append(key([
        "<b>Part 1 (sections 1–12).</b> We started out trying to improve KePIN by making its Koopman part \"dynamic\". "
        "By explaining what the model was really doing, we found that KePIN's Koopman part was not driving its "
        "accuracy, and that the real problem on the hardest datasets was <b>flying-condition noise</b> drowning out the "
        "<b>wear signal</b>. Fixing that inside the architecture (a <b>condition-aware KePIN</b>) beat the original "
        "KePIN and the published paper by <b>1.1 RMSE on FD002</b> and <b>1.8 RMSE on FD004</b>. Along the way we built "
        "a <b>wear model</b> that matches KePIN's accuracy and explains every single prediction.",
        "<b>Part 2 (sections 13–23).</b> The thesis question became: <b>how can the Koopman operator adapt to the "
        "engine's health?</b> Adaptive rules built from the 30-flight window were ignored or added nothing, because "
        "the window holds no information the predictor doesn't already have. A <b>hypernetwork</b> that builds a "
        "Koopman rule for every engine from a <b>health index computed from the engine's whole history</b> finally "
        "made Koopman matter in original KePIN. With this health regime, error fell by <b>2 to 5 RMSE on all four "
        "datasets</b> (fair test; FD002 15.4 → 11.8, FD004 17.8 → 12.9). A control shows the gain comes from the health information itself, not from routing it through "
        "Koopman, but the health-driven Koopman rule is readable: its modes fade more slowly as an engine nears failure."],
        "The whole story in two paragraphs"))
    st.append(P("How to read this guide", "h2"))
    st += bullets([
        "Each section follows the same pattern: <b>what we did</b>, the <b>proof</b> (a figure, real code or a table), "
        "a yellow <b>Why?</b> box, and what it means.",
        "Blue boxes give the short version. You can read just the blue and yellow boxes for a quick overview.",
        "All numbers are RMSE (how many flights off the predictions are, on average). <b>Lower is better.</b>"])
    st.append(P("Contents", "h2"))
    st.append(P("<b>Part 1: understanding and fixing KePIN</b>", "toc"))
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
                           "Final results, ablation and limitations (Part 1)",
                           None,
                           "The new question: can the Koopman rule adapt to engine health?",
                           "First attempts: three adaptive rules on condition-aware KePIN",
                           "Adaptive Koopman where Koopman is the predictor (wear model)",
                           "Why there was little to adapt: the encoder straightens the curve",
                           "Hypernetworks: building a Koopman rule for every engine",
                           "Feeding the health regime to the hypernetwork",
                           "Reading the adaptive Koopman rule",
                           "Results on all four datasets",
                           "How a hypernetwork makes Koopman adaptive (the easy version)",
                           "How this helps explain the engine",
                           "Where we stand now",
                           "Glossary"], 1):
        if t is None:
            st.append(P("<b>Part 2: making the Koopman operator adaptive</b>", "toc"))
            continue
        st.append(P(f"{i if i < 13 else i - 1}.&nbsp;&nbsp;{t}", "toc"))

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
    st.append(P("In our <i>dynamic</i> Koopman versions (section 5) the Koopman rule changes from engine to engine, "
                "so the 12 numbers <i>can</i> differ per engine. We tested whether the predictor actually uses them: "
                "we took the trained models, <b>replaced every engine's 12 numbers with one fixed vector</b>, and "
                "measured how much the predictions changed."))
    st.append(table([["Model (FD002, fair-test runs)", "How much the 12 numbers differ between engines", "Error: real features → constant", "Largest change in any prediction"],
                     ["Fixed rule (control), runs 0–2", "0 (identical by design)", "15.436 / 15.836 / 16.095 → unchanged", "0.000 flights"],
                     ["Dynamic A: operating condition, runs 0–2", "small (0.001–0.005)", "16.207 → 16.208; 16.181 and 16.094 unchanged", "0.090 / 0.002 / 0.002 flights"],
                     ["Dynamic B: learned health, runs 0–2", "essentially none (≈ 0.0000004)", "15.538 / 16.790 / 15.537 → unchanged", "0.000 flights"]],
                    [0.27, 0.25, 0.27, 0.21]))
    st.append(callout("Your questions, answered", [
        "<b>Which dynamic models did we test?</b> Dynamic version <b>A</b> (the Koopman rule changes with the operating "
        "condition at every flight) and dynamic version <b>B</b> (the rule's decay is shifted by a learned \"health\" "
        "value), each trained 3 times on FD002 with the fair-test protocol. The fixed-rule model was included as a "
        "control, because its 12 numbers are identical for every engine by design.",
        "<b>What constant did we replace the 12 numbers with?</b> For each trained model we computed its 12 numbers for "
        "all 253 FD002 test engines and took their <b>average</b>, giving one 12-number vector. Every engine then received "
        "that same vector. For dynamic B, run 0, it was "
        "[0.017, 0.017, 0.019, 0.019, 0.078, 0.160, 0.160, 0.011, 0.011, 0.080, 0.983, 0.000] "
        "(5 decay rates, 5 frequencies, spectral radius, spectral gap).",
        "<b>Was it a random constant or something derived?</b> <b>Derived</b>, not random: it is the model's own average "
        "over the test engines. We deliberately avoided zeros or random numbers. Those would feed the predictor values "
        "it never saw in training, and any change in error would then come from the unusual input, not from the lost "
        "information. With the average, the predictor still gets typical, in-range values; only the engine-to-engine "
        "<i>differences</i> are removed.",
        "<b>Where is the code?</b> <font name='DVM'>dynamic_koopman/analyze_spectral_freeze.py</font>. The constant is "
        "computed in the line <font name='DVM'>const = feats.mean(axis=0)</font>, and the replacement is the function "
        "<font name='DVM'>frozen</font>, which temporarily takes the place of "
        "<font name='DVM'>kepin_model.extract_spectral_features</font> (the function KePIN calls to build the 12 numbers). "
        "Results are saved in <font name='DVM'>dynamic_koopman/reports/spectral_freeze/summary.json</font>.",
        "<b>What does the table show?</b> Only dynamic A really tests the question, because only A's 12 numbers vary "
        "between engines, and even then only slightly. Removing that variation changed predictions by at most 0.09 flights. "
        "Dynamic B's learned health hardly moved the slowest Koopman modes (the ones the 12 numbers describe), so B's "
        "numbers were already effectively constant and the test tells us nothing extra about B."],
        WHY_BG, colors.HexColor("#6250d6")))
    st.append(code("""
feats = original(ko["eigenvalues"], top_k=m.spectral_k).numpy()  # 12 features x 253 engines
const = feats.mean(axis=0)                     # their AVERAGE: one 12-number vector

def frozen(eigenvalues, top_k=5, c=const):     # every engine gets that same vector
    c = tf.constant(c[None], tf.float32)
    return tf.tile(c, [tf.shape(eigenvalues)[0], 1])

kepin_model.extract_spectral_features = frozen  # KePIN now uses the constant
pred_c, _ = m(inp, training=False)              # predictions with frozen features""",
                   "dynamic_koopman/analyze_spectral_freeze.py (simplified)"))
    st.append(why(["Analogy: a doctor predicts each patient's health from their test results plus a \"Koopman note\". "
                   "If the note is the same printed sheet for every patient, it cannot help tell patients apart.",
                   "And when the note <i>did</i> differ a little per patient (dynamic A), the predictor had learned to "
                   "ignore it, because it already received the full hidden state (192–256 numbers) directly."]))
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

    st += part2(rep)

    # ---------- glossary
    st += [PageBreak(), P("24. Glossary", "h1")]
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
                     ["Shortcut", "When a model learns an easy pattern that happens to work, instead of the intended one."],
                     ["Adaptive Koopman operator", "A Koopman rule that is different for each engine (or each moment), instead of one rule for the whole fleet."],
                     ["Decay factor ρ / half-life", "How much a mode keeps per step (ρ = 0.9 keeps 90%). Half-life = steps until a mode is half its size."],
                     ["Health index h", "One number between 0 (near failure) and 1 (healthy) that the model computes for each engine."],
                     ["mono loss", "A training rule with no RUL labels: a later window of the same engine must have a lower h."],
                     ["Hypernetwork", "A small network whose output is the settings (weights) of another part of the model, here the Koopman rule of each engine."],
                     ["Adaptive rank / mode gate", "A per-engine on/off switch for each Koopman mode, so an engine can use fewer modes."],
                     ["Engine history features", "42 numbers per window that summarise the engine from its first flight up to now (drift since new, current rate)."],
                     ["Knockout test", "Switch one part off (or make it identical for every engine) in a trained model and see how much the error grows."],
                     ["Control arm", "A comparison model that gets the same information by a simpler route, to check what really causes a gain."],
                     ["Bootstrap CI", "Resample the 253 test engines many times to get a 95% range for a difference; if it includes 0, the difference may be luck."],
                     ["Curvature (speed-up)", "How much faster a degradation signal moves in the last 50 flights than in mid-life; 1× = a straight line."],
                     ["Self-supervised encoder", "An encoder trained without RUL labels, only to rebuild its own input."]],
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
