"""
analyze_ecg.py  --  analyse ONE ECG recording end to end: ensemble prediction + XAI + PDF report.

This is the backend for the web app. It is also a normal terminal command, so you can test
everything yourself before any UI exists.

    python analyze_ecg.py --input my_recording.npy
    python analyze_ecg.py --input data/Physionet_dataset/WFDBRecords/01/010/JS00001
    python analyze_ecg.py --input recording.csv --out results/case1 --fast

Accepted input
--------------
  .npy   shape (5000, 12), ALREADY preprocessed (e.g. one row of X_test.npy).
         Add --raw if your .npy is unfiltered signal that still needs preprocessing.
  .csv   raw 12-lead signal, 12 columns, one row per sample (header optional).
  WFDB   the record path with no extension (.hea and .mat must sit next to each other),
         or the path to the .hea / .mat file itself.
Raw inputs are run through the SAME preprocessing as training.

Output (in --out, default outputs/web_analysis/<input name>/)
------------------------------------------------------------
  report.pdf        the human-readable report
  results.json      every number, for the web UI to render
  *.png             the XAI figures the PDF embeds

For the web API, see serve.py, which imports analyze() from this file.
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import time

import numpy as np

import matplotlib
matplotlib.use("Agg")

# ---------------------------------------------------------------------------
# Plain-language class information
# ---------------------------------------------------------------------------

CLASS_INFO = {
    "SB":   ("Sinus Bradycardia",
             "A normal rhythm pattern, but slower than usual (under about 60 beats per minute). "
             "Common in athletes and during sleep; can also be caused by medication."),
    "AFIB": ("Atrial Fibrillation",
             "An irregular rhythm arising in the upper chambers of the heart. The regular P wave "
             "before each beat is absent and the spacing between beats is erratic."),
    "SR":   ("Sinus Rhythm",
             "The heart's normal rhythm at a normal rate (about 60-100 beats per minute)."),
    "ST":   ("Sinus Tachycardia",
             "A normal rhythm pattern, but faster than usual (over about 100 beats per minute). "
             "Common during exercise, fever, stress or dehydration."),
    "SVT":  ("Supraventricular Tachycardia",
             "A fast rhythm that starts above the ventricles. Episodes typically begin and end "
             "abruptly."),
}

DISCLAIMER = (
    "Research prototype - not a medical device. This report is produced by machine-learning "
    "models trained on a public ECG database. It is not a diagnosis, has not been clinically "
    "validated, and must not be used to make any medical decision. Any ECG finding must be "
    "reviewed by a qualified clinician. If you have symptoms that concern you, seek medical care."
)

_ENGINE = {}


# ---------------------------------------------------------------------------
# Loading a recording
# ---------------------------------------------------------------------------

def load_recording(input_path: str, force_raw: bool = False):
    """Returns (x, how) where x is a preprocessed (5000, 12) float32 array."""
    from config_physionet import PhysioNetConfig as PNConfig
    T, L = PNConfig.RECORD_LENGTH, PNConfig.NUM_LEADS
    ext = os.path.splitext(input_path)[1].lower()

    if ext == ".npy" and not force_raw:
        x = np.load(input_path)
        if x.ndim == 3 and x.shape[0] == 1:
            x = x[0]
        if x.shape != (T, L):
            raise ValueError(
                f"expected a preprocessed array of shape ({T}, {L}), got {tuple(x.shape)}. "
                f"If this is raw signal, pass --raw.")
        return np.nan_to_num(x.astype(np.float32)), "preprocessed .npy (used as-is)"

    # --- everything else is raw signal and gets the training-time pipeline ---
    from utils.preprocessing_physionet import (
        bandpass_filter, fix_length, load_signal, normalize_signal, wavelet_denoise,
    )

    if ext == ".npy":
        sig, how = np.load(input_path).astype(np.float32), "raw .npy"
    elif ext == ".csv":
        try:
            sig = np.loadtxt(input_path, delimiter=",", dtype=np.float32)
        except ValueError:                                  # has a header row
            sig = np.loadtxt(input_path, delimiter=",", dtype=np.float32, skiprows=1)
        how = "raw .csv"
    else:
        base = input_path
        if ext in (".hea", ".mat", ".dat"):
            base = os.path.splitext(input_path)[0]
        sig, how = load_signal(base).astype(np.float32), "raw WFDB record"

    if sig.ndim != 2:
        raise ValueError(f"expected a 2-D signal (samples x leads), got shape {tuple(sig.shape)}")
    if sig.shape[0] == L and sig.shape[1] != L:             # (12, N) -> (N, 12)
        sig = sig.T
    if sig.shape[1] != L:
        raise ValueError(f"expected {L} leads, got {sig.shape[1]}")

    sig = bandpass_filter(sig)
    sig = wavelet_denoise(sig)
    sig = normalize_signal(sig)
    sig = fix_length(sig)
    return np.nan_to_num(sig.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0), how + " (preprocessed)"


# ---------------------------------------------------------------------------
# Models, loaded once per process
# ---------------------------------------------------------------------------

def get_engine(combiner_path="ensemble_combiner.json"):
    if _ENGINE:
        return _ENGINE
    from config_physionet import PhysioNetConfig as PNConfig
    from ensemble_combiner import StackedCombiner
    from ensemble_xai.adapters import build_adapters

    PNConfig.load_label_vocab()
    comb = StackedCombiner.load(combiner_path)
    adapters = build_adapters()
    if comb.model_names != [a.name for a in adapters]:
        raise RuntimeError("combiner model order does not match adapter order")
    _ENGINE.update(combiner=comb, adapters=adapters,
                   class_names=list(PNConfig.CLASS_NAMES), lead_names=list(PNConfig.LEAD_NAMES))
    return _ENGINE


def load_reference_metrics():
    """System-level performance, read from the files you already produced. Missing files are fine."""
    m = {}
    try:
        d = json.load(open("ensemble_improve_results.json"))
        chosen = d.get("chosen_on_val")
        r = d["methods"][chosen]
        m["method"] = chosen
        m["test_macro_f1"] = r["test_f1"]
        m["test_accuracy"] = r["test_acc"]
        m["test_ece"] = r["test_ece"]
        m["per_class_f1"] = r["per_class_f1"]
        m["best_single_model"] = d.get("best_single")
        m["best_single_f1"] = d.get("best_single_test_f1")
        sig = d.get("significance_chosen_vs_best_single", {})
        m["mcnemar_p"] = sig.get("mcnemar_p")
    except Exception:
        pass
    try:
        d = json.load(open("ensemble_deletion_results.json"))
        m["deletion"] = {"aopc_most_relevant": d["AOPC_MoRF"], "aopc_random": d["AOPC_random"],
                         "aopc_least_relevant": d["AOPC_LeRF"],
                         "share_ok": d["share_recordings_MoRF_below_random"],
                         "n_recordings": d["n_recordings"]}
    except Exception:
        pass
    return m


# ---------------------------------------------------------------------------
# The analysis itself
# ---------------------------------------------------------------------------

def analyze(input_path: str, out_dir: str | None = None, fast: bool = False,
            force_raw: bool = False, make_pdf: bool = True,
            combiner_path: str = "ensemble_combiner.json") -> dict:
    """Analyse one recording. Returns the results dict (also written to results.json)."""
    from ensemble_xai.pipeline import generate_ensemble_assets_for_recording

    t0 = time.time()
    eng = get_engine(combiner_path)
    comb, adapters = eng["combiner"], eng["adapters"]
    class_names, lead_names = eng["class_names"], eng["lead_names"]

    x, how = load_recording(input_path, force_raw=force_raw)

    name = os.path.splitext(os.path.basename(input_path))[0] or "recording"
    out_dir = out_dir or os.path.join("outputs", "web_analysis", name)
    os.makedirs(out_dir, exist_ok=True)

    ws = 500 if fast else 250
    xai = generate_ensemble_assets_for_recording(
        x, adapters, np.ones(len(adapters)), None, out_dir,
        class_names, lead_names, window_size=ws, stride=ws, combiner=comb,
    )

    pred = xai["predicted_class"]
    full_name, description = CLASS_INFO.get(pred, (pred, ""))
    leads_sorted = sorted(xai["ensemble_lead_importance"].items(), key=lambda t: -t[1])

    results = {
        "input": {"path": input_path, "name": name, "interpreted_as": how},
        "generated_at": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
        "prediction": {
            "class": pred,
            "full_name": full_name,
            "description": description,
            "confidence": xai["confidence"],
            "probabilities": xai["probs"],
        },
        "models": {
            "predictions": xai["per_model_prediction"],
            "confidences": xai["per_model_confidence"],
            "agrees_with_ensemble": xai["per_model_agrees_with_ensemble"],
            "n_agreeing": xai["n_models_agreeing"],
            "n_total": xai["n_models_total"],
            "shares": xai["ensemble_weights"],
        },
        "explanation": {
            "lead_importance": xai["ensemble_lead_importance"],
            "top_leads": [k for k, _ in leads_sorted[:3]],
            "window_size_samples": ws,
            "figures": {
                "lead_importance": xai["lead_importance_figure"],
                "lead_time_heatmap": xai["lead_time_heatmap_figure"],
                "model_agreement": xai["model_agreement_figure"],
                "per_model_lead_importance": xai["per_model_lead_importance_figure"],
                "per_lead_saliency": {k: v["saliency_figure"] for k, v in xai["leads"].items()},
            },
        },
        "system_performance": load_reference_metrics(),
        "disclaimer": DISCLAIMER,
        "seconds_taken": round(time.time() - t0, 1),
    }

    if make_pdf:
        pdf_path = os.path.join(out_dir, "report.pdf")
        build_pdf(results, pdf_path)
        results["report_pdf"] = pdf_path

    with open(os.path.join(out_dir, "results.json"), "w") as fh:
        json.dump(results, fh, indent=2)
    results["output_dir"] = out_dir
    return results


# ---------------------------------------------------------------------------
# The PDF report
# ---------------------------------------------------------------------------

def build_pdf(r: dict, out_path: str) -> str:
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_JUSTIFY
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import mm
    from reportlab.lib.utils import ImageReader
    from reportlab.platypus import (Image, PageBreak, Paragraph, SimpleDocTemplate,
                                    Spacer, Table, TableStyle)

    ss = getSampleStyleSheet()
    body = ParagraphStyle("body", parent=ss["BodyText"], fontSize=10, leading=14, alignment=TA_JUSTIFY)
    small = ParagraphStyle("small", parent=body, fontSize=8, leading=11, textColor=colors.HexColor("#555555"))
    h1 = ParagraphStyle("h1", parent=ss["Heading1"], fontSize=14, spaceBefore=14, spaceAfter=6)
    warn = ParagraphStyle("warn", parent=body, fontSize=9, leading=13,
                          textColor=colors.HexColor("#7a1c1c"), backColor=colors.HexColor("#fdf0f0"),
                          borderPadding=8, borderWidth=0.8, borderColor=colors.HexColor("#c98b8b"))

    PAGE_W = A4[0] - 36 * mm

    def fig(path, caption, max_h_mm=95):
        """Embed a PNG scaled to the page width, capped in height."""
        if not path or not os.path.exists(path):
            return []
        iw, ih = ImageReader(path).getSize()
        w = PAGE_W
        h = w * ih / iw
        if h > max_h_mm * mm:
            h = max_h_mm * mm
            w = h * iw / ih
        return [Image(path, width=w, height=h), Paragraph(caption, small), Spacer(1, 8)]

    def table(rows, widths=None, highlight_row=None):
        t = Table(rows, colWidths=widths, hAlign="LEFT")
        style = [
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 9),
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#eeeeee")),
            ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#bbbbbb")),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("TOPPADDING", (0, 0), (-1, -1), 4), ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ]
        if highlight_row:
            style.append(("BACKGROUND", (0, highlight_row), (-1, highlight_row), colors.HexColor("#e6f0fb")))
            style.append(("FONTNAME", (0, highlight_row), (-1, highlight_row), "Helvetica-Bold"))
        t.setStyle(TableStyle(style))
        return t

    p = r["prediction"]
    conf_pct = f"{p['confidence'] * 100:.1f}%"
    perf = r.get("system_performance", {})
    s = []

    # ---- cover ----
    s += [Paragraph("TrustECG - ECG Rhythm Analysis", ss["Title"]),
          Paragraph(f"Recording: <b>{r['input']['name']}</b> &nbsp;|&nbsp; "
                    f"Generated: {r['generated_at'][:16].replace('T', ' ')}", small),
          Spacer(1, 10), Paragraph(DISCLAIMER, warn), Spacer(1, 14)]

    # ---- 1. result ----
    s += [Paragraph("1. Result", h1),
          Paragraph(f"The ensemble classified this recording as <b>{p['full_name']} ({p['class']})</b>, "
                    f"with a confidence of <b>{conf_pct}</b>.", body),
          Spacer(1, 4), Paragraph(p["description"], body), Spacer(1, 10)]

    probs = sorted(p["probabilities"].items(), key=lambda t: -t[1])
    rows = [["Rhythm", "Full name", "Probability"]]
    hl = None
    for i, (c, v) in enumerate(probs, start=1):
        rows.append([c, CLASS_INFO.get(c, (c, ""))[0], f"{v * 100:.1f}%"])
        if c == p["class"]:
            hl = i
    s += [table(rows, [60, 230, 80], highlight_row=hl), Spacer(1, 6),
          Paragraph("The five rhythms above are the only ones these models were trained to "
                    "recognise. Any other rhythm will still be forced into one of these five, so a "
                    "high probability is not evidence that no other condition is present.", small)]

    # ---- 2. the three models ----
    s += [Paragraph("2. What each model said", h1)]
    m = r["models"]
    rows = [["Model", "Its own prediction", "Its confidence", "Weight in ensemble"]]
    for n in m["predictions"]:
        rows.append([n, m["predictions"][n], f"{m['confidences'][n] * 100:.1f}%",
                     f"{m['shares'].get(n, 0) * 100:.0f}%"])
    s += [table(rows, [110, 110, 90, 110]), Spacer(1, 8)]

    if m["n_agreeing"] == m["n_total"]:
        agree_txt = (f"All {m['n_total']} models independently reached the same conclusion. "
                     f"Agreement does not guarantee the answer is correct, but disagreement is a "
                     f"useful warning sign, and there is none here.")
    else:
        agree_txt = (f"Only {m['n_agreeing']} of {m['n_total']} models agreed with the final answer. "
                     f"This recording is one the models find ambiguous, so the result deserves "
                     f"closer review than the confidence figure alone suggests.")
    s += [Paragraph(agree_txt, body), Spacer(1, 8)]
    s += fig(r["explanation"]["figures"].get("model_agreement"),
             "Figure 1. What each model predicted, and how confident it was.")

    # ---- 3. how much to trust it ----
    s += [PageBreak(), Paragraph("3. How reliable is this system?", h1)]
    if perf.get("test_macro_f1"):
        rows = [["Measure", "Value", "What it means"]]
        rows.append(["Accuracy", f"{perf['test_accuracy'] * 100:.1f}%",
                     Paragraph("Share of unseen test recordings classified correctly.", small)])
        rows.append(["Macro F1", f"{perf['test_macro_f1'] * 100:.1f}%",
                     Paragraph("Averages performance across all five rhythms equally, so rare "
                               "rhythms count as much as common ones.", small)])
        rows.append(["Calibration error", f"{perf['test_ece']:.3f}",
                     Paragraph("How closely the stated confidence matches reality. Near zero means "
                               "a stated 90% really is right about 90% of the time.", small)])
        s += [table(rows, [90, 60, 260]), Spacer(1, 8)]

        pc = perf.get("per_class_f1", {}).get(p["class"])
        if pc is not None:
            worst = min(perf["per_class_f1"], key=perf["per_class_f1"].get)
            line = (f"For <b>{p['class']}</b> specifically, the system scores an F1 of "
                    f"{pc:.3f} on unseen data.")
            if p["class"] == worst:
                line += (" This is the system's <b>weakest</b> class, so treat this particular "
                         "result with extra caution.")
            s += [Paragraph(line, body), Spacer(1, 8)]

    d = perf.get("deletion")
    if d:
        s += [Paragraph(
            f"The explanations below were tested for faithfulness on {d['n_recordings']} recordings. "
            f"Blanking the regions the system called important reduced its confidence about "
            f"{d['aopc_most_relevant'] / max(d['aopc_random'], 1e-9):.1f} times more than blanking "
            f"random regions, which means the highlights track evidence the models genuinely use. "
            f"That expected pattern held on {d['share_ok'] * 100:.0f}% of recordings individually, "
            f"so the highlighting is informative on average but not guaranteed for every recording.",
            body), Spacer(1, 8)]

    # ---- 4. what the models looked at ----
    s += [Paragraph("4. Which parts of the ECG mattered", h1),
          Paragraph("Each small section of each lead was blanked out in turn to see how much the "
                    "system's confidence fell. A large fall means that section carried evidence the "
                    "decision depended on. Brighter colour means more important.", body), Spacer(1, 8)]

    top = r["explanation"]["top_leads"]
    li = r["explanation"]["lead_importance"]
    s += [Paragraph(f"The three most influential leads were <b>{', '.join(top)}</b> "
                    f"(carrying {sum(li[k] for k in top) * 100:.0f}% of the total importance).", body),
          Spacer(1, 8)]
    s += fig(r["explanation"]["figures"].get("lead_importance"),
             "Figure 2. How much each of the 12 leads contributed to the decision.", 80)
    s += fig(r["explanation"]["figures"].get("lead_time_heatmap"),
             "Figure 3. Importance across every lead over the 10 seconds of the recording.")
    s += [PageBreak()]
    s += fig(r["explanation"]["figures"].get("per_model_lead_importance"),
             "Figure 4. The same analysis per model. Similar patterns mean the models agree on "
             "what to look at, not just on the answer.")

    figs = r["explanation"]["figures"].get("per_lead_saliency", {})
    for i, lead in enumerate(top, start=5):
        s += fig(figs.get(lead), f"Figure {i}. Lead {lead} with the important regions highlighted.", 70)

    # ---- 5. limitations ----
    s += [Paragraph("5. Limitations", h1)]
    bullets = [
        "This is a research prototype. It is not a medical device, has not been clinically "
        "validated, and must not be used to make a medical decision.",
        "The models were trained on one public 12-lead ECG database. Performance on recordings from "
        "different equipment, populations or recording conditions is unknown and likely lower.",
        "Only five rhythms can be recognised. Any other rhythm, and any condition outside rhythm "
        "classification, will be misreported as one of the five.",
        "The highlighting measures what happens when one region is removed at a time. Where two "
        "leads carry the same information, removing just one changes little, so genuinely useful "
        "regions can be under-credited.",
        "Regions that argued against the prediction are shown as zero importance, not as negative, "
        "so the figures show supporting evidence only.",
        "One model reconstructs individual heartbeats before classifying. Blanking part of the "
        "signal can change which beats it finds, not only how they look, which adds noise to its "
        "share of the explanation.",
        "A single recording is 10 seconds. Intermittent rhythms may simply not be present in it.",
    ]
    for b in bullets:
        s += [Paragraph(f"&bull;&nbsp; {b}", body), Spacer(1, 3)]

    s += [Spacer(1, 10), Paragraph(DISCLAIMER, warn)]

    SimpleDocTemplate(out_path, pagesize=A4,
                      leftMargin=18 * mm, rightMargin=18 * mm,
                      topMargin=16 * mm, bottomMargin=16 * mm,
                      title=f"TrustECG report - {r['input']['name']}").build(s)
    return out_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Analyse one ECG recording: prediction + XAI + PDF report.")
    ap.add_argument("--input", required=True, help=".npy / .csv / WFDB record path")
    ap.add_argument("--out", default=None, help="output folder (default outputs/web_analysis/<name>)")
    ap.add_argument("--fast", action="store_true", help="coarser analysis, roughly twice as fast")
    ap.add_argument("--raw", action="store_true", help="the .npy is raw signal, not preprocessed")
    ap.add_argument("--no_pdf", action="store_true", help="results.json and figures only")
    ap.add_argument("--combiner", default="ensemble_combiner.json")
    a = ap.parse_args()

    r = analyze(a.input, a.out, fast=a.fast, force_raw=a.raw,
                make_pdf=not a.no_pdf, combiner_path=a.combiner)
    p = r["prediction"]
    m = r["models"]
    print(f"\n  input      : {r['input']['interpreted_as']}")
    print(f"  prediction : {p['class']} ({p['full_name']})  confidence {p['confidence'] * 100:.1f}%")
    print(f"  models     : {m['n_agreeing']}/{m['n_total']} agree -> " +
          ", ".join(f"{k}={v}" for k, v in m["predictions"].items()))
    print(f"  top leads  : {', '.join(r['explanation']['top_leads'])}")
    print(f"  took       : {r['seconds_taken']}s")
    print(f"\n  {r['output_dir']}\\results.json")
    if r.get("report_pdf"):
        print(f"  {r['report_pdf']}")


if __name__ == "__main__":
    main()