"""
build_xai_report.py
====================
Assembles the final XAI Word report from the figures + summary JSON
produced by generate_xai_report_assets.py. Run that script FIRST.

Every number in this report (confidence, heart rate, beat count, which
lead mattered most) is read from xai_report_summary.json — nothing here
is hand-typed or guessed, so re-running generate_xai_report_assets.py on
a different checkpoint and re-running this script always reflects your
actual current results.

Scope decision (read before assuming something is missing)
------------------------------------------------------------
generate_xai_report_assets.py produces ~50 figures PER CLASS (12 leads x
4 figure types). Embedding all of them in one document would make it
unreadable and enormous. This report embeds, per class:
  - the 12-lead overview grid and the lead-attention chart (whole-signal
    views)
  - a full deep-dive (technical annotation, Grad-CAM++ saliency,
    comparison-to-normal) for Lead II (the standard clinical reference
    lead) AND for whichever lead the model's own attention weighted most
    highly, if different from Lead II
The report explicitly tells the reader where to find the complete
12-lead breakdown for anyone who wants to inspect every lead.

Usage
-----
    python build_xai_report.py --model cnn_bilstm2
"""

from __future__ import annotations

import argparse
import json
import os

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Inches, Pt, RGBColor
from docx.oxml.ns import qn
from docx.oxml import OxmlElement

from config_physionet import PhysioNetConfig as PNConfig

ACCENT = RGBColor(0x1F, 0x4E, 0x79)
IMG_WIDTH = Inches(6.2)
IMG_WIDTH_HALF = Inches(3.0)

# Generic, textbook rhythm descriptions — safe, standard definitions used
# only to describe the class taxonomy itself, not to diagnose anyone.
CLASS_DESCRIPTIONS = {
    "SB": {
        "full_name": "Sinus Bradycardia",
        "plain": "The heart is beating more slowly than the typical resting rate, "
                 "but in an otherwise normal, coordinated pattern — each heartbeat "
                 "still follows the usual sequence, it's just happening less often "
                 "per minute than usual.",
        "clinical": "Sinus rhythm with a resting rate below 60 bpm. P-wave morphology, "
                    "PR interval, and QRS morphology are otherwise consistent with "
                    "normal sinus conduction; the defining feature is rate alone.",
    },
    "AFIB": {
        "full_name": "Atrial Fibrillation",
        "plain": "The heart's upper chambers quiver rapidly and irregularly instead "
                 "of contracting in a single coordinated beat. This produces a "
                 "heartbeat rhythm that is irregular in an unpredictable way — the "
                 "time between beats varies beat to beat rather than staying steady.",
        "clinical": "Characterised by the absence of distinct P-waves (replaced by "
                    "fibrillatory activity) and an irregularly irregular ventricular "
                    "response — R-R intervals show no consistent pattern.",
    },
    "SR": {
        "full_name": "Sinus Rhythm (Normal)",
        "plain": "This is the normal, healthy heartbeat pattern — regular timing "
                 "between beats, and each beat follows the expected sequence. This "
                 "is used as the reference/baseline throughout this report.",
        "clinical": "Regular rhythm at 60-100 bpm, with a P-wave preceding every "
                    "QRS complex, a normal PR interval, and normal QRS/T morphology.",
    },
    "ST": {
        "full_name": "Sinus Tachycardia",
        "plain": "The heart is beating faster than the typical resting rate, but "
                 "still in a regular, coordinated pattern — the same rhythm as "
                 "normal, just sped up.",
        "clinical": "Sinus rhythm with a resting rate above 100 bpm. Regular R-R "
                    "intervals and normal P-QRS-T morphology; rate is the "
                    "distinguishing feature.",
    },
    "SVT": {
        "full_name": "Supraventricular Tachycardia",
        "plain": "An abnormally fast heart rhythm originating above the heart's "
                 "main pumping chambers — typically noticeably faster than sinus "
                 "tachycardia, and can start or stop abruptly.",
        "clinical": "A fast rhythm (often >150 bpm) originating above the "
                    "ventricles. Depending on subtype, P-waves may be absent, "
                    "retrograde, or hidden within the preceding T-wave.",
    },
}


def add_heading(doc, text, level=1):
    h = doc.add_heading(text, level=level)
    for run in h.runs:
        run.font.color.rgb = ACCENT
    return h


def add_caption(doc, text):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = p.add_run(text)
    run.italic = True
    run.font.size = Pt(9)
    run.font.color.rgb = RGBColor(0x55, 0x55, 0x55)
    return p


def add_figure(doc, path, caption_text, width=IMG_WIDTH):
    if not os.path.exists(path):
        doc.add_paragraph(f"[Figure not found: {path}]")
        return
    doc.add_picture(path, width=width)
    doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
    add_caption(doc, caption_text)


def add_dual_box(doc, plain_text, technical_text):
    """A visually distinct 'plain language / technical' pair of paragraphs."""
    p1 = doc.add_paragraph()
    r1 = p1.add_run("In plain language:  ")
    r1.bold = True
    r1.font.color.rgb = RGBColor(0x16, 0x7A, 0x3D)
    p1.add_run(plain_text)

    p2 = doc.add_paragraph()
    r2 = p2.add_run("Technical / clinical detail:  ")
    r2.bold = True
    r2.font.color.rgb = RGBColor(0x1F, 0x4E, 0x79)
    p2.add_run(technical_text)


def add_page_number_footer(doc):
    section = doc.sections[0]
    footer = section.footer
    p = footer.paragraphs[0]
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = p.add_run()
    fld = OxmlElement("w:fldSimple")
    fld.set(qn("w:instr"), "PAGE")
    run._r.append(fld)


def fmt_bpm(v):
    return f"{v:.0f} bpm" if v is not None else "not determinable from this example"


def add_recording_section(doc, section_label, class_name, cdata, report_dir,
                           normal_class_name, true_label_known=True,
                           extra_note=None):
    """
    Writes one full recording's explanation section (used for both a
    known test-set class example and an arbitrary new user recording).

    Parameters
    ----------
    section_label : str
        e.g. "2.1" for a batch report, or "" for a single-recording report.
    class_name : str
        The class this recording is labelled/predicted as, for figure
        titles and the CLASS_DESCRIPTIONS lookup.
    true_label_known : bool
        If True, `class_name` is a ground-truth label and the section
        reports "correctly/incorrectly classified" against it. If False
        (a genuinely new, unlabelled recording), only the prediction
        itself is reported — there is no ground truth to check against.
    extra_note : str, optional
        An additional sentence shown at the top of the section (e.g. to
        flag that this is a brand-new, unlabelled recording).
    """
    desc = CLASS_DESCRIPTIONS.get(class_name, {"full_name": class_name, "plain": "", "clinical": ""})

    heading_text = f"{section_label} {class_name} — {desc['full_name']}".strip()
    add_heading(doc, heading_text, level=2 if section_label else 1)

    if extra_note:
        p = doc.add_paragraph()
        r = p.add_run(extra_note)
        r.italic = True

    if desc["plain"] or desc["clinical"]:
        add_dual_box(doc, desc["plain"], desc["clinical"])

    # --- Prediction result ---
    if true_label_known:
        correct_word = "correctly" if cdata["was_correctly_classified"] else "INCORRECTLY"
        doc.add_paragraph(
            f"Model result on this example: predicted \"{cdata['predicted_class']}\" "
            f"with {cdata['confidence']:.1%} confidence — this example's true label "
            f"({class_name}) was classified {correct_word}."
        )
        if not cdata["was_correctly_classified"]:
            p = doc.add_paragraph()
            r = p.add_run(
                "This is reported honestly rather than cherry-picking only correct "
                "examples — an incorrect prediction's explanation is often as "
                "informative as a correct one."
            )
            r.italic = True
    else:
        doc.add_paragraph(
            f"Model prediction: \"{cdata['predicted_class']}\" with "
            f"{cdata['confidence']:.1%} confidence. No ground-truth label is "
            f"available for this recording, so this reports the model's own "
            f"output only — it is not being checked against a known diagnosis."
        )

    probs_line = ", ".join(f"{k}: {v:.1%}" for k, v in cdata["probs"].items())
    p = doc.add_paragraph()
    p.add_run("Full probability breakdown: ").bold = True
    p.add_run(probs_line)

    # --- 12-lead overview ---
    add_heading(doc, "All 12 leads (overview)", level=3)
    add_figure(doc, cdata["twelve_lead_grid_figure"],
               f"Figure — all 12 leads of this {class_name} example, stacked on a shared time axis.")

    # --- Lead-attention chart ---
    add_heading(doc, "Which leads the model relied on most", level=3)
    sorted_leads = sorted(cdata["lead_weights"].items(), key=lambda kv: kv[1], reverse=True)
    top_lead_name, top_lead_weight = sorted_leads[0]
    add_dual_box(
        doc,
        f"Out of all 12 leads, the model paid the most attention to lead "
        f"{top_lead_name} when making this prediction (the tallest bar below).",
        f"Lead-attention weight ranking (highest first): "
        + ", ".join(f"{n} ({w:.3f})" for n, w in sorted_leads[:5]) + ", ...",
    )
    add_figure(doc, cdata["lead_attention_figure"],
               f"Figure — learned lead-attention weights for this {class_name} example.")

    # --- Deep-dive leads: Lead II (clinical standard) + top-attention lead ---
    deep_dive_leads = ["II"]
    if top_lead_name != "II":
        deep_dive_leads.append(top_lead_name)

    for lead_name in deep_dive_leads:
        reason = "the standard clinical reference lead" if lead_name == "II" else "the lead the model itself weighted most heavily"
        add_heading(doc, f"Detailed view — Lead {lead_name} ({reason})", level=3)

        lead_data = cdata["leads"].get(lead_name)
        if lead_data is None:
            doc.add_paragraph(f"[No data found for lead {lead_name}]")
            continue

        tech = lead_data["technical"]
        hr_text = fmt_bpm(tech.get("heart_rate_bpm"))
        add_dual_box(
            doc,
            f"In this lead, the recording is {tech['duration_s']:.0f} seconds long "
            f"({tech['num_samples']} samples at {tech['fs']} samples per second). "
            f"The system automatically detected {tech['num_beats_detected']} heartbeats "
            f"in this window, giving an estimated heart rate of {hr_text}.",
            f"R-peak detection on Lead {lead_name}: {tech['num_beats_detected']} beats "
            f"detected over a {tech['duration_s']:.1f}s window "
            f"({tech['num_samples']} samples @ {tech['fs']} Hz), estimated rate "
            f"{hr_text}. P/QRS/T window boundaries shown in the figure below are "
            f"approximate (typical-timing heuristic, not a dedicated delineation "
            f"algorithm — see Section 1.2).",
        )
        folder = lead_data["folder"]
        add_figure(doc, os.path.join(folder, "technical.png"),
                   f"Figure — Lead {lead_name}, {class_name}: heartbeat detection and approximate P/QRS/T regions.")
        add_figure(doc, os.path.join(folder, "gradcam_saliency.png"),
                   f"Figure — Lead {lead_name}, {class_name}: Grad-CAM++ saliency (where the model looked).")

        if class_name != normal_class_name and "comparison_to_normal" in lead_data:
            comp = lead_data["comparison_to_normal"]
            add_dual_box(
                doc,
                f"Compared side-by-side with a normal heartbeat, this {class_name} "
                f"example shows an estimated rate of {fmt_bpm(comp['class_heart_rate_bpm'])}, "
                f"versus {fmt_bpm(comp['normal_heart_rate_bpm'])} for the normal reference — "
                f"looking at both waveforms together makes the difference easy to see "
                f"directly, not just as two separate numbers.",
                f"Rate comparison (Lead {lead_name}): {class_name} = "
                f"{fmt_bpm(comp['class_heart_rate_bpm'])} "
                f"({comp['class_beats_detected']} beats detected in the window) vs. "
                f"Normal Sinus Rhythm = {fmt_bpm(comp['normal_heart_rate_bpm'])} "
                f"({comp['normal_beats_detected']} beats detected).",
            )
            add_figure(doc, os.path.join(folder, "vs_normal.png"),
                       f"Figure — Lead {lead_name}: {class_name} vs. Normal Sinus Rhythm, same time axis.")

    # --- Pointer to the complete per-lead breakdown ---
    class_folder = os.path.join(report_dir, class_name) if true_label_known else report_dir
    p = doc.add_paragraph()
    r = p.add_run(
        f"Note: the figures above cover Lead II and the model's top-attention "
        f"lead for readability. The complete breakdown — all 12 leads, each with "
        f"its own technical annotation, Grad-CAM++ saliency, temporal-attention, "
        f"and normal-comparison figure — is available in: {class_folder}"
    )
    r.italic = True
    r.font.size = Pt(9)
    r.font.color.rgb = RGBColor(0x55, 0x55, 0x55)

    doc.add_page_break()


def build_single_recording_report(summary: dict, save_dir: str, output_path: str):
    """
    Builds a report for ONE new, arbitrary recording (explain_user_recording.py),
    reusing add_recording_section() — same figures, same dual-tier
    explanations, same rigor as the batch report, just one recording
    instead of five.
    """
    doc = Document()
    add_page_number_footer(doc)

    title = doc.add_paragraph()
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = title.add_run("TrustECG")
    run.bold = True
    run.font.size = Pt(40)
    run.font.color.rgb = ACCENT

    sub = doc.add_paragraph()
    sub.alignment = WD_ALIGN_PARAGRAPH.CENTER
    sub.add_run("Explanation Report — Uploaded Recording").font.size = Pt(18)
    doc.add_page_break()

    add_heading(doc, "How to Read This Report", level=1)
    doc.add_paragraph(
        "Every section below has two paired explanations: one in plain language "
        "for a reader with no ECG background, and one in standard clinical/"
        "technical terminology. Every number quoted (confidence, estimated heart "
        "rate, beat count, which lead mattered most) is computed directly from "
        "the model's own output on this specific recording."
    )
    doc.add_page_break()

    cdata = summary["recording"]
    add_recording_section(
        doc, section_label="", class_name=cdata["predicted_class"], cdata=cdata,
        report_dir=save_dir, normal_class_name=summary.get("normal_reference", {}).get("class"),
        true_label_known=False,
        extra_note=(
            "This is a newly provided recording with no known ground-truth "
            "label — the section below reports the model's prediction and "
            "explanation for it, not a check against a known diagnosis."
        ),
    )

    add_heading(doc, "Limitations", level=1)
    doc.add_paragraph(
        "P-wave and T-wave locations are approximated from typical timing "
        "windows relative to each detected R-peak — not diagnostic-grade "
        "wave-delineation output."
    )
    doc.add_paragraph(
        "This report reflects one automated model's output on one recording. "
        "It is a research/educational explainability tool, not a medical "
        "diagnosis."
    )

    doc.save(output_path)
    return output_path


def build_report(summary: dict, report_dir: str, output_path: str):
    doc = Document()
    add_page_number_footer(doc)

    # ------------------------------------------------------------------
    # Title page
    # ------------------------------------------------------------------
    title = doc.add_paragraph()
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = title.add_run("TrustECG")
    run.bold = True
    run.font.size = Pt(40)
    run.font.color.rgb = ACCENT

    sub = doc.add_paragraph()
    sub.alignment = WD_ALIGN_PARAGRAPH.CENTER
    sub.add_run("Explainability (XAI) Report").font.size = Pt(20)

    sub2 = doc.add_paragraph()
    sub2.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = sub2.add_run(
        "Per-class, per-lead explanation of model predictions, written for both "
        "general readers and readers with ECG/clinical background."
    )
    r.italic = True
    r.font.color.rgb = RGBColor(0x55, 0x55, 0x55)

    doc.add_paragraph()
    meta = doc.add_paragraph()
    meta.alignment = WD_ALIGN_PARAGRAPH.CENTER
    meta.add_run(f"Model: {summary['model']}")
    doc.add_page_break()

    # ------------------------------------------------------------------
    # How to read this report
    # ------------------------------------------------------------------
    add_heading(doc, "How to Read This Report", level=1)
    doc.add_paragraph(
        "Every section below has two paired explanations: one written in plain "
        "language for a reader with no ECG background, and one written with "
        "standard clinical/technical terminology, for a reader who wants the "
        "precise detail. Both describe the same figure — read whichever suits you, "
        "or both."
    )
    doc.add_paragraph(
        "Every number quoted in this report (confidence, estimated heart rate, "
        "beat count, which lead mattered most) is computed directly from the "
        "model's own output on a real example from the test set — nothing here "
        "is a hypothetical or illustrative number."
    )

    # ------------------------------------------------------------------
    # ECG Fundamentals
    # ------------------------------------------------------------------
    add_heading(doc, "1. ECG Fundamentals", level=1)

    add_heading(doc, "1.1 What the numbers on these graphs mean", level=2)
    add_dual_box(
        doc,
        "An ECG is just a recording of the heart's electrical activity over time, "
        "turned into a wiggly line. The X-axis of most figures in this report is "
        "either \"time in seconds\" or \"sample number\" — think of a sample as "
        "one single measurement, taken many times per second, like a very fast "
        "series of snapshots.",
        "Recordings in this dataset are sampled at 500 Hz — 500 samples per "
        "second — for a fixed duration of 10 seconds, giving exactly 5,000 "
        "samples per recording per lead. A sample index of 2,500 therefore "
        "corresponds to the 5-second mark (2500 / 500 = 5.0s). The Y-axis "
        "(\"amplitude\") is the recorded electrical voltage, normalised (z-scored) "
        "during preprocessing so different recordings and leads are on a "
        "comparable scale.",
    )

    add_heading(doc, "1.2 What a \"heartbeat\" looks like on this graph", level=2)
    add_dual_box(
        doc,
        "Each heartbeat shows up as a small cluster of bumps repeating over and "
        "over. The tallest, sharpest spike in each cluster is the moment the "
        "heart's main pumping chambers (the ventricles) contract — that one sharp "
        "spike is the easiest part of a heartbeat to spot by eye, and it's what "
        "this report uses to count heartbeats and estimate heart rate.",
        "Each cardiac cycle produces three landmark waveform components: the "
        "P-wave (atrial depolarisation), the QRS complex (ventricular "
        "depolarisation — the sharp, high-amplitude spike), and the T-wave "
        "(ventricular repolarisation). This report detects the R-peak (the QRS "
        "complex's peak) via a prominence-based peak finder, then estimates heart "
        "rate from the average spacing between consecutive R-peaks (60 / mean "
        "R-R interval in seconds).",
    )

    add_heading(doc, "1.3 The 12-lead system", level=2)
    add_dual_box(
        doc,
        "A standard ECG doesn't just record from one place on the body — it "
        "records from 12 different \"viewpoints\" (called leads) at once, each "
        "one looking at the heart's electrical activity from a slightly "
        "different angle. Some rhythm problems show up more clearly in some "
        "leads than others, which is part of why doctors look at all 12 "
        "together rather than just one.",
        "The 12 standard leads are: I, II, III (limb leads), aVR, aVL, aVF "
        "(augmented limb leads), and V1-V6 (precordial/chest leads). This report's "
        "model includes a learned lead-attention mechanism that assigns each "
        "recording its own per-lead importance weighting — shown as a bar chart "
        "in every class section below — rather than treating all 12 leads as "
        "equally informative.",
    )

    add_heading(doc, "1.4 What the explainability (XAI) figures show", level=2)
    add_dual_box(
        doc,
        "Beyond just predicting a diagnosis, this system tries to show its work. "
        "The red-shaded overlay on a waveform (\"Grad-CAM++ saliency\") highlights "
        "which parts of that specific recording most influenced the model's "
        "decision — the taller the red shading, the more that moment mattered to "
        "the prediction. This is similar in spirit to a heatmap.",
        "Grad-CAM++ (Chattopadhay et al., 2018) is a gradient-based class "
        "activation mapping technique: it weights the final convolutional layer's "
        "feature maps by a second-order approximation of each feature map's "
        "importance to the target class's logit, producing a localisation map "
        "upsampled to the input's temporal resolution. This report computes it "
        "per lead — see Section 1.3 — using a batched, single-forward-pass "
        "architecture (see the accompanying project report's Methodology section "
        "for the vectorisation detail).",
    )

    doc.add_page_break()

    # ------------------------------------------------------------------
    # Per-class sections
    # ------------------------------------------------------------------
    class_names = list(summary["classes"].keys())
    normal_class = summary["normal_reference"]["class"]

    add_heading(doc, "2. Per-Class Explanations", level=1)
    doc.add_paragraph(
        f"Each class below is illustrated with ONE representative recording from "
        f"the test set (the first correctly-classified example of that class, or "
        f"the first available example if none were classified correctly for that "
        f"class — stated honestly per class below). '{normal_class}' "
        f"({CLASS_DESCRIPTIONS.get(normal_class, {}).get('full_name', normal_class)}) "
        f"is used as the normal reference for comparison throughout."
    )

    for class_idx, class_name in enumerate(class_names, start=1):
        cdata = summary["classes"][class_name]
        add_recording_section(
            doc, section_label=f"2.{class_idx}", class_name=class_name, cdata=cdata,
            report_dir=report_dir, normal_class_name=normal_class, true_label_known=True,
        )

    # ------------------------------------------------------------------
    # Summary comparison table
    # ------------------------------------------------------------------
    add_heading(doc, "3. Summary Comparison Table", level=1)
    doc.add_paragraph(
        "All figures below are computed from Lead II of each class's representative "
        "example, for direct comparability."
    )
    table = doc.add_table(rows=1, cols=5)
    table.style = "Light Grid Accent 1"
    hdr = table.rows[0].cells
    for i, text in enumerate(["Class", "Predicted", "Correct?", "Confidence", "Est. Heart Rate (Lead II)"]):
        hdr[i].text = text

    for class_name in class_names:
        cdata = summary["classes"][class_name]
        lead_ii = cdata["leads"].get("II", {})
        hr = lead_ii.get("technical", {}).get("heart_rate_bpm")
        row = table.add_row().cells
        row[0].text = class_name
        row[1].text = cdata["predicted_class"]
        row[2].text = "Yes" if cdata["was_correctly_classified"] else "No"
        row[3].text = f"{cdata['confidence']:.1%}"
        row[4].text = fmt_bpm(hr)

    doc.add_paragraph()

    # ------------------------------------------------------------------
    # Limitations
    # ------------------------------------------------------------------
    add_heading(doc, "4. Limitations", level=1)
    doc.add_paragraph(
        "P-wave and T-wave locations shown in the technical figures are "
        "approximated from typical physiological timing windows relative to each "
        "detected R-peak — they are NOT the output of a dedicated wave-delineation "
        "algorithm, and should not be treated as diagnostic-grade boundaries."
    )
    doc.add_paragraph(
        "Temporal-attention figures show the SAME attention weights on every lead "
        "of a given class's example — this architecture fuses all 12 leads before "
        "computing temporal attention, so it is not an independently-computed "
        "per-lead quantity (unlike Grad-CAM++ saliency, which genuinely is)."
    )
    doc.add_paragraph(
        "Each class is illustrated with a single representative example, not an "
        "aggregate over many recordings — this report shows how the model "
        "explains one specific case per class, not a statistical summary of its "
        "behaviour across the whole class."
    )

    doc.save(output_path)
    return output_path


def parse_args():
    parser = argparse.ArgumentParser(description="Build the XAI Word report from generated assets.")
    parser.add_argument("--model", type=str, default=None, help="Only used to locate the summary JSON if it differs from the config default.")
    return parser.parse_args()


def main():
    args = parse_args()
    report_dir = os.path.join(PNConfig.PLOTS_PATH, "xai_report")
    summary_path = os.path.join(report_dir, "xai_report_summary.json")
    if not os.path.exists(summary_path):
        raise FileNotFoundError(
            f"{summary_path} not found. Run generate_xai_report_assets.py first."
        )
    with open(summary_path) as fh:
        summary = json.load(fh)

    output_path = os.path.join(report_dir, "TrustECG_XAI_Report.docx")
    build_report(summary, report_dir, output_path)
    print(f"Report saved -> {output_path}")


if __name__ == "__main__":
    main()