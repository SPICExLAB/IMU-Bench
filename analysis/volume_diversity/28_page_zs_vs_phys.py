#!/usr/bin/env python
"""Build the "Probe Input Scaling" page: z-score vs physical probe input on the vd_ek grid.

    python analysis/volume_diversity/28_page_zs_vs_phys.py <out.html>

Reads out/zs_vs_phys_deltas.json + out/zs_vs_phys_4src_{light,dark}.png (write them first with
27_plot_zs_vs_phys.py) and bench/results/. Republish the same file path to update the artifact.
"""
from __future__ import annotations
import base64, datetime, json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
OUT = os.path.join(HERE, "out")
KEY = {"har_wisdm": ("WISDM", "macro_f1", True, "HAR · WISDM (macro-F1)"),
       "har_utd_mhad": ("UTD-MHAD", "macro_f1", True, "HAR · UTD-MHAD (macro-F1)"),
       "har_pamap2": ("PAMAP2", "macro_f1", True, "HAR · PAMAP2 (macro-F1)"),
       "har_recofit": ("RecoFit", "macro_f1", True, "HAR · RecoFit (macro-F1)"),
       "microgesture_openwatch": ("Gesture", "clip_macro_f1", True, "Microgesture (clip F1)"),
       "arm_pose_nymeria": ("Arm pose", "pos_err_cm", False, "Arm pose (pos err, cm)"),
       "full_body_pose_nymeria": ("Full-body", "mpjpe_mm", False, "Full-body pose (MPJPE, mm)"),
       "odometry_nymeria_worldframe": ("Odometry", "ate_m", False, "Odometry (ATE, m)"),
       "har_hhar": ("HHAR", "macro_f1", True, "HAR · HHAR (macro-F1)"),
       "har_mhealth": ("MHEALTH", "macro_f1", True, "HAR · MHEALTH (macro-F1)"),
       "har_samosa": ("SAMoSA", "macro_f1", True, "HAR · SAMoSA (macro-F1)"),
       "har_ut_watch": ("UT-Watch", "macro_f1", True, "HAR · UT-Watch (macro-F1)"),
       "har_prism": ("PRISM", "macro_f1", True, "HAR · PRISM (macro-F1)")}
TOTAL = 24 * len(KEY)
NTAG = {"0p01": "0.01%", "0p1": "0.1%", "1": "1%", "10": "10%", "100": "100%"}


def parse(arm):
    k, f = arm[len("vd_ek"):].split("_f")
    return int(k), f


def val(task, arm, k):
    p = os.path.join(REPO, "bench", "results", task, f"{arm}__probe__s42.json")
    return json.load(open(p))["metrics"].get(k) if os.path.exists(p) else None


def main(out_path):
    D = json.load(open(os.path.join(OUT, "zs_vs_phys_deltas.json")))
    cells = sorted(json.load(open(os.path.join(OUT, "enc_type_volume_grid.json")))["arms"],
                   key=lambda a: (parse(a)[0], list(NTAG).index(parse(a)[1])))
    rows, n_pairs = [], 0
    for a in cells:
        K, f = parse(a); tds = []
        for t, (nm, k, hib, _l) in KEY.items():
            z, ph = val(t, a + "_zs", k), val(t, a, k)
            if z is None or ph is None:
                tds.append('<td class="pend">&#183;</td>'); continue
            d = (z - ph) * (1 if hib else -1); n_pairs += 1
            txt = f"{d:+.3f}" if abs(d) >= 0.001 else f"{d:+.4f}"
            tds.append(f'<td class="{"pos" if d > 0 else "neg"}">{txt}</td>')
        rows.append(f'<tr><th scope="row">K {K} &#183; N {NTAG[f]}</th>{"".join(tds)}</tr>')
    sumrows = ""
    for t, (nm, k, hib, lab) in KEY.items():
        d = D[lab]
        if d["mean"] is None:
            sumrows += f'<tr><td>{nm}</td><td>0/24</td><td colspan="3" class="pend">physical-input twins still running</td></tr>'; continue
        sumrows += (f'<tr><td>{nm}</td><td>{d["n"]}/24</td><td class="{"pos" if d["mean"] > 0 else "neg"}">{d["mean"]:+.4f}</td>'
                    f'<td>{d["min"]:+.4f} … {d["max"]:+.4f}</td><td>{d["z_better"]}/{d["n"]}</td></tr>')
    L = base64.b64encode(open(os.path.join(OUT, "zs_vs_phys_4src_light.png"), "rb").read()).decode()
    Dk = base64.b64encode(open(os.path.join(OUT, "zs_vs_phys_4src_dark.png"), "rb").read()).decode()
    n_zs = sum(1 for t in KEY for a in cells
               if os.path.exists(os.path.join(REPO, "bench", "results", t, f"{a}_zs__probe__s42.json")))
    now = datetime.datetime.now().strftime("%d %b %H:%M")
    done = n_zs == TOTAL
    status = (f'<span class="status ok">COMPLETE &#183; all {TOTAL} z-score cells scored &#183; updated {now}</span>' if done
              else f'<span class="status">IN PROGRESS &#183; {n_zs} of {TOTAL} z-score cells scored &#183; updated {now}</span>')
    pend = [f"{nm} {D[lab]['n']}/24" for t, (nm, k, h, lab) in KEY.items() if D[lab]["n"] < 24]
    pend_line = ("Every task is complete." if not pend else "Paired cells still incomplete: " + ", ".join(pend) + ".")
    _sm = D["HAR · SAMoSA (macro-F1)"]
    samosa_line = (f"So far SAMoSA ({_sm['n']} pairs) favours physical input by {-_sm['mean']:.3f} macro-F1, z-score better in {_sm['z_better']}/{_sm['n']}."
                   if _sm["mean"] is not None else "")
    alt = ("Fourteen panels: a bar summary of the mean effect of z-score input per task, then thirteen task panels with solid "
           "z-score lines over dashed physical-input lines across motion-type clusters and data volumes.")
    html = f'''<title>Probe Input Scaling</title>
<link rel="preconnect" href="https://fonts.googleapis.com"><link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Archivo:wght@500;600;700&family=IBM+Plex+Mono:wght@400;500;600&family=Source+Serif+4:opsz,wght@8..60,400;8..60,600&display=swap">
<style>
:root{{--paper:#EDF0F1;--panel:#F7F9F9;--ink:#12171A;--ink-2:#3D4A50;--muted:#66747A;--rule:#D2DADD;--rule-soft:#E1E7E9;
--accent:#0B6B6E;--accent-ink:#075053;--pos:#1B6B4A;--neg:#9A3B2C;--warn-fill:#B98A1E1C;--warn:#9A6E12;--ok-fill:#0B6B6E14;
--sans:"Archivo","Helvetica Neue",-apple-system,sans-serif;--serif:"Source Serif 4",Georgia,serif;--mono:"IBM Plex Mono",ui-monospace,Menlo,monospace}}
@media (prefers-color-scheme:dark){{:root:not([data-theme="light"]){{--paper:#0D1113;--panel:#151B1E;--ink:#E7ECEE;--ink-2:#B7C2C7;--muted:#8A979D;
--rule:#263034;--rule-soft:#1D2528;--accent:#4FB3B6;--accent-ink:#7FCDCF;--pos:#6FC48F;--neg:#D98070;--warn-fill:#D9A83A22;--warn:#E0B45A;--ok-fill:#4FB3B61F}}}}
:root[data-theme="dark"]{{--paper:#0D1113;--panel:#151B1E;--ink:#E7ECEE;--ink-2:#B7C2C7;--muted:#8A979D;--rule:#263034;--rule-soft:#1D2528;
--accent:#4FB3B6;--accent-ink:#7FCDCF;--pos:#6FC48F;--neg:#D98070;--warn-fill:#D9A83A22;--warn:#E0B45A;--ok-fill:#4FB3B61F}}
*{{box-sizing:border-box}}
body{{margin:0;background:var(--paper);color:var(--ink);font-family:var(--serif);font-size:16.5px;line-height:1.62}}
.wrap{{max-width:1040px;margin:0 auto;padding-inline:20px;padding-block:0 72px}}
p{{margin:0 0 1em;max-width:68ch}} h1,h2{{font-family:var(--sans);text-wrap:balance;margin:0}}
.eyebrow{{font-family:var(--mono);font-size:11px;letter-spacing:.14em;text-transform:uppercase;color:var(--muted)}}
header.top{{border-bottom:1px solid var(--rule);padding-block:40px 26px;margin-bottom:28px}}
h1{{font-size:clamp(32px,6vw,52px);font-weight:700;letter-spacing:-.022em;line-height:1.05;margin-top:8px}}
.sub{{margin-top:12px;font-size:18.5px;color:var(--ink-2);max-width:64ch}}
.status{{display:inline-flex;margin-top:14px;font-family:var(--mono);font-size:12px;color:var(--warn);border:1px dashed var(--warn);border-radius:2px;padding:4px 10px;background:var(--warn-fill)}}
.status.ok{{color:var(--accent-ink);border:1px solid var(--accent);background:var(--ok-fill)}}
section{{margin-block:0 44px}} h2{{font-size:24px;font-weight:700;margin:6px 0 8px}}
.fig{{background:var(--panel);border:1px solid var(--rule-soft);border-radius:3px;padding:14px;margin:16px 0 8px}}
.fig img{{display:block;width:100%;height:auto;border-radius:2px}} .fig .img-dark{{display:none}}
@media (prefers-color-scheme:dark){{:root:not([data-theme="light"]) .fig .img-light{{display:none}} :root:not([data-theme="light"]) .fig .img-dark{{display:block}}}}
:root[data-theme="dark"] .fig .img-light{{display:none}} :root[data-theme="dark"] .fig .img-dark{{display:block}}
figcaption{{font-family:var(--mono);font-size:11.5px;color:var(--muted);margin-top:9px;line-height:1.5}}
.scroll{{overflow-x:auto;margin:12px 0 8px}}
table{{border-collapse:collapse;width:100%;min-width:620px;font-size:14px}}
caption{{text-align:left;font-family:var(--mono);font-size:11px;letter-spacing:.13em;text-transform:uppercase;color:var(--muted);padding-bottom:8px}}
th,td{{padding:6px 12px 6px 0;text-align:right;border-bottom:1px solid var(--rule-soft)}}
th:first-child,td:first-child{{text-align:left}}
thead th{{font-family:var(--mono);font-weight:500;font-size:11.5px;color:var(--muted);border-bottom:1px solid var(--rule);white-space:nowrap}}
tbody td{{font-family:var(--mono);font-variant-numeric:tabular-nums;color:var(--ink-2)}}
th[scope=row]{{font-family:var(--mono);font-size:12.5px;color:var(--ink);white-space:nowrap;text-align:left}}
td.pos{{color:var(--pos);font-weight:600}} td.neg{{color:var(--neg);font-weight:600}} td.pend{{color:var(--muted)}}
.cap{{font-size:13.5px;color:var(--muted);margin:8px 2px 0;max-width:78ch}}
ul{{margin:0 0 1em;padding-left:1.15em;max-width:70ch}} li{{margin-bottom:.45em}}
code{{font-family:var(--mono);font-size:.88em;background:var(--rule-soft);padding:1px 5px;border-radius:2px}}
footer{{border-top:1px solid var(--rule);padding-top:20px;color:var(--muted);font-size:13.5px}}
@media (max-width:520px){{.wrap{{padding-inline:16px}}}}
</style>
<div class="wrap">
<header class="top">
  <div class="eyebrow">IMU-Bench &#183; evaluation protocol &#183; vd_ek grid, 4-source pool</div>
  <h1>Probe Input Scaling</h1>
  <p class="sub">The same 24 pretrained checkpoints, the same probe recipe and seed, scored twice: once on physically scaled input, once on the per-window z-score its pretraining used.</p>
  {status}
</header>
<section>
  <div class="eyebrow">Why this exists</div>
  <h2>A protocol mismatch, measured</h2>
  <p>Every vd_ek encoder was pretrained on per-window z-scored input. From 20 Aug to 14 Sep the bench fed all of them <b>physically scaled</b> input at probe time, so the whole grid was scored under a train/eval mismatch. The bench now honours each encoder's declared normalization, and these runs measure what that choice was worth. Nothing was re-pretrained and no old result was overwritten: the z-score runs use alias encoders <code>vd_ek*_zs</code> that point at the same checkpoints.</p>
</section>
<section>
  <div class="eyebrow">Figure</div>
  <h2>Solid = z-score input, dashed = physical input</h2>
  <figure class="fig">
    <img class="img-light" src="data:image/png;base64,{L}" alt="{alt}">
    <img class="img-dark" src="data:image/png;base64,{Dk}" alt="{alt}">
    <figcaption>vd_ek grid, 24 cells &#215; 13 tasks, seed 42. x = motion-type clusters the arm may draw from; one line per pretraining volume. First panel: mean effect of z-score input per task, as a percentage of that task's mean physical-input score. {pend_line}</figcaption>
  </figure>
</section>
<section>
  <div class="eyebrow">Result</div>
  <h2>Physical input wins on 10 of 13 tasks</h2>
  <div class="scroll"><table>
    <caption>Effect of z-score input, per task (positive = z-score better)</caption>
    <thead><tr><th>task</th><th>paired cells</th><th>mean effect</th><th>range</th><th>z-score better</th></tr></thead>
    <tbody>{sumrows}</tbody>
  </table></div>
  <p class="cap">Signed so positive always means z-score input scored better: for arm pose, full-body pose and odometry, where lower is better, the sign is flipped.</p>
  <ul>
    <li><b>Every paired cell favours physical input</b> on WISDM, UTD-MHAD, RecoFit, MHEALTH, SAMoSA, PRISM, arm pose and odometry (0 of 24 cells each); HHAR and UT-Watch are 3 of 24.</li>
    <li><b>Size, relative to each task's physical-input mean:</b> SAMoSA &#8722;28%, PRISM &#8722;24%, odometry &#8722;22% (0.028 m), MHEALTH &#8722;17%, UTD-MHAD &#8722;15%, HHAR &#8722;13%, RecoFit and WISDM &#8722;10 to &#8722;11%, UT-Watch &#8722;9%.</li>
    <li><b>The five HAR sets added last</b> (HHAR, MHEALTH, SAMoSA, UT-Watch, PRISM, none in the pretraining pool) follow the same direction as the first four, and include the two largest gaps. Shoaib was dropped: it has a single subject, so its subject-disjoint split leaves no validation labels.</li>
    <li><b>Arm pose</b> &#8212; only &#8722;1.8% relative, but 0.148 cm is larger than the 0.108 cm run-to-run noise band measured on byte-identical configs, and it holds in all 24 cells.</li>
    <li><b>Microgesture</b> &#8212; the one clear exception: z-score is better in 20 of 24 cells, +0.060 clip F1 (+12%).</li>
    <li><b>PAMAP2 and full-body pose</b> &#8212; a wash: &#8722;0.007 macro-F1 (13 of 24 cells favour z-score) and +0.19 mm on a ~92 mm metric (16 of 24).</li>
    <li>The reading: tasks that need gravity, posture or absolute intensity gain from physical input, which keeps the DC component per-window z-score removes. Microgesture depends on fine waveform shape and prefers the input its encoder was trained on.</li>
  </ul>
</section>
<section>
  <div class="eyebrow">Every paired cell</div>
  <h2>Per-cell effect of z-score input</h2>
  <div class="scroll"><table>
    <caption>z-score minus physical, per cell ({n_pairs} paired cells)</caption>
    <thead><tr><th>cell</th>{"".join(f"<th>{v[0]}</th>" for v in KEY.values())}</tr></thead>
    <tbody>{"".join(rows)}</tbody>
  </table></div>
  <p class="cap">K = clusters the pretraining subset may draw from; N = share of the 1,738,094-window pool. Green means z-score input scored better, red means physical did. A dot marks a pair whose z-score cell is still queued.</p>
</section>
<section>
  <div class="eyebrow">Method and caveats</div>
  <h2>What is and is not controlled</h2>
  <ul>
    <li><b>Controlled:</b> same checkpoint, same probe template and recipe, same 20,000-step budget, same subject-disjoint split, seed 42. Only <code>preprocess.normalize.method</code> differs.</li>
    <li><b>Single seed.</b> Per-cell differences smaller than run-to-run noise are not resolvable; the per-task direction, consistent across 24 cells, is the reliable part.</li>
    <li><b>Consequence for the other pages:</b> Five Data Axes and the forecast page report physical-input numbers. Comparisons inside those pages hold; their absolute values are not comparable with protocol-consistent cells.</li>
    <li><b>The trade-off:</b> matching the encoder's pretraining is the methodologically clean choice, and on this benchmark it costs accuracy on most tasks. That belongs in the paper rather than in a silent default.</li>
  </ul>
</section>
<footer><p>Figure: <code>analysis/volume_diversity/27_plot_zs_vs_phys.py</code>; page: <code>28_page_zs_vs_phys.py</code>; deltas in <code>out/zs_vs_phys_deltas.json</code>. Results: <code>bench/results/&lt;task&gt;/vd_ek*_zs__probe__s42.json</code> against <code>vd_ek*__probe__s42.json</code>. Companion pages: <a href="https://claude.ai/code/artifact/016bae4c-8545-42ad-86f1-7c1b14133410" style="color:var(--accent-ink)">Five Data Axes</a>, <a href="https://claude.ai/artifact/HxduXXNwonJ3eSe2mgYJoa" style="color:var(--accent-ink)">Random-Type Grid Forecast</a>.</p></footer>
</div>
'''
    open(out_path, "w").write(html)
    print(f"page written: {len(html)} chars; {n_pairs} paired cells; {n_zs}/{TOTAL} z-score cells")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else os.path.join(OUT, "zscore_protocol.html"))
