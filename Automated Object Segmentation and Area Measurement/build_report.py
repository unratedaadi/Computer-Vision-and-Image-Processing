"""Build the offline report and case viewer from the current benchmark CSV."""
import csv
import json
from collections import defaultdict
from html import escape
from statistics import mean

from segment import ROOT


METHOD_NAMES = {"global": "Fixed global", "otsu": "Otsu"}


def case_label(image, condition):
    if image.startswith("scene_"):
        return condition
    return f"Leaf {image.split('_')[1]} · {condition}"


def main():
    with (ROOT / "outputs" / "summary.csv").open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    by_case = defaultdict(dict)
    by_condition = defaultdict(list)
    for row in rows:
        by_case[row["image"]][row["method"]] = row
        by_condition[(row["condition"], row["method"])].append(row)

    initial = "leaf_04_original"
    condition_rows, count_rows, options = [], [], []
    for (condition, method), items in by_condition.items():
        raw, cleaned, dice = [mean(float(row[key]) for row in items) for key in ("raw_iou", "iou", "dice")]
        area = mean(abs(float(row["area_error_pct"])) for row in items)
        matches = sum(row["objects"] == row["reference_objects"] for row in items)
        condition_rows.append(
            f"<tr><th scope='row'>{escape(condition)}</th><td>{METHOD_NAMES[method]}</td>"
            f"<td>{matches}/{len(items)}</td><td>{raw:.3f}</td><td>{cleaned:.3f}</td>"
            f"<td>{dice:.3f}</td><td>{area:.1f}%</td></tr>"
        )

    for image, comparison in by_case.items():
        fixed, otsu = comparison["global"], comparison["otsu"]
        label = escape(case_label(image, fixed["condition"]))
        selected = " selected" if image == initial else ""
        options.append(f'<option value="{image}"{selected}>{label}</option>')
        count_rows.append(
            f"<tr><th scope='row'>{label}</th><td>{fixed['reference_objects']}</td>"
            f"<td>{int(fixed['reference_area_px']):,}</td>"
            f"<td>{fixed['objects']}</td><td>{int(fixed['area_px']):,}</td>"
            f"<td>{otsu['objects']}</td><td>{int(otsu['area_px']):,}</td></tr>"
        )

    original = {method: mean(float(row["iou"]) for row in by_condition[("Original lighting", method)])
                for method in METHOD_NAMES}
    dim_fixed = mean(float(row["iou"]) for row in by_condition[("Simulated dim light", "global")])
    dim_otsu = mean(float(row["iou"]) for row in by_condition[("Simulated dim light", "otsu")])
    shadow_otsu = mean(float(row["iou"]) for row in by_condition[("Simulated side shadow", "otsu")])
    full_image = int(by_case["leaf_04_dim"]["global"]["area_px"])
    separated = by_case["scene_separated"]["otsu"]
    touching = by_case["scene_touching"]["otsu"]
    total = len(by_case)
    case_json = json.dumps(by_case).replace("<", "\\u003c")
    html = PAGE.format(
        options="\n".join(options), condition_rows="\n".join(condition_rows),
        count_rows="\n".join(count_rows), case_json=case_json, total=total,
        initial=initial, original_fixed=original["global"], original_otsu=original["otsu"],
        dim_fixed=dim_fixed, dim_otsu=dim_otsu, shadow_otsu=shadow_otsu, full_image=full_image,
        separated=separated["objects"], touching=touching["objects"], touching_iou=float(touching["iou"]),
    )
    (ROOT / "report.html").write_text(html, encoding="utf-8")
    print("Wrote report.html")


PAGE = """<!doctype html>
<html lang="en">
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Mini Project 5 | Automated Object Segmentation and Area Measurement</title>
<style>
:root {{ color-scheme: light; --ink: #1b2c23; --muted: #48584e; --paper: #f5f5ee; --line: #cdd6ca; --accent: #205739; }}
* {{ box-sizing: border-box; }}
body {{ margin: 0; background: var(--paper); color: var(--ink); font: 16px/1.55 system-ui, sans-serif; }}
header {{ border-bottom: 1px solid var(--line); background: #e3ecdf; }}
.wrap {{ max-width: 1120px; margin: auto; padding: 2.4rem clamp(1rem, 4vw, 3rem); }}
h1, h2 {{ font-family: Georgia, serif; font-weight: normal; line-height: 1.1; }}
h1 {{ max-width: 840px; font-size: clamp(2.4rem, 5vw, 4.1rem); margin: .35rem 0 .75rem; }}
h2 {{ font-size: 2rem; margin: 0 0 1rem; }}
p {{ margin: .55rem 0 .9rem; }}
header p:last-child, section > p {{ max-width: 80ch; }}
.kicker {{ font-size: .82rem; font-weight: 700; letter-spacing: .12em; text-transform: uppercase; }}
section + section {{ border-top: 1px solid var(--line); margin-top: 2.6rem; padding-top: 2.2rem; }}
.layout {{ display: grid; grid-template-columns: minmax(0, 1.15fr) minmax(0, 1fr); gap: 1.5rem; align-items: start; }}
figure {{ margin: 0; }}
figure img {{ width: 100%; height: auto; display: block; border: 1px solid #26392c; border-radius: .35rem; }}
figcaption, .note {{ color: var(--muted); font-size: .88rem; margin-top: .65rem; }}
label {{ display: block; font-weight: 650; margin-bottom: .35rem; }}
select {{ width: 100%; font: inherit; color: inherit; padding: .55rem .65rem; border: 1px solid #aebdac; border-radius: .3rem; background: #fff; }}
.table {{ overflow-x: auto; margin: .8rem 0; }}
table {{ width: 100%; border-collapse: collapse; background: #fff; font-variant-numeric: tabular-nums; }}
th, td {{ border-bottom: 1px solid var(--line); padding: .58rem .65rem; text-align: left; vertical-align: top; }}
thead th {{ background: #e3ecdf; }}
tbody th {{ font-weight: 550; }}
td {{ white-space: nowrap; }}
a {{ color: var(--accent); text-underline-offset: .2em; }}
li + li {{ margin-top: .48rem; }}
code {{ background: #e7ece3; padding: .1rem .2rem; }}
@media (max-width: 760px) {{ .layout {{ grid-template-columns: 1fr; }} }}
@media print {{ body {{ background: white; font-size: 11pt; }} .wrap {{ padding: 1rem 0; }} section {{ break-inside: auto; }} tr, figure {{ break-inside: avoid; }} }}
</style>
<header><div class="wrap">
  <p class="kicker">Computer Vision and Image Processing · Mini Project 5</p>
  <h1>Automated Object Segmentation and Area Measurement</h1>
  <p>Fixed global thresholding and Otsu thresholding on five real leaf scans and twelve derived test cases.</p>
</div></header>
<main class="wrap">
<section>
  <h2>Working demonstration</h2>
  <div class="layout">
    <figure>
      <a id="image-link" href="outputs/{initial}/comparison.png">
        <img id="comparison" src="outputs/{initial}/comparison.png" width="800" height="834"
          alt="Input, reference foreground, fixed global and Otsu mask and annotated output">
      </a>
      <figcaption>Click to open the full-size comparison. The green outline marks predicted foreground.</figcaption>
    </figure>
    <div>
      <label for="case">Benchmark image</label>
      <select id="case">{options}</select>
      <div class="table"><table aria-label="Selected measurements">
      <thead><tr><th scope="col">Method</th><th scope="col">Threshold</th><th scope="col">Count</th><th scope="col">Area (px²)</th><th scope="col">IoU</th></tr></thead>
      <tbody id="selected-results"></tbody></table></div>
      <p id="reference" class="note" aria-live="polite"></p>
      <p><a id="input" href="data/inputs/{initial}.png">Input image</a> ·
      <a id="raw" href="outputs/{initial}/otsu/raw_mask.png">Otsu raw mask</a> ·
      <a id="clean" href="outputs/{initial}/otsu/mask.png">Otsu cleaned mask</a></p>
      <p class="note">Try the original, dim, and side-shadow variants of Leaf 04, then compare the separated and touching scenes.</p>
      <noscript><p>Use the full count and area table below if JavaScript is disabled.</p></noscript>
    </div>
  </div>
</section>
<section>
  <h2>Experimental method</h2>
  <p>OpenCV decodes the image, converts it to grayscale and applies a 5 × 5 Gaussian blur. The fixed method uses intensity 200, selected as a baseline for the bright-background scans. Otsu estimates one threshold per image by maximizing the variance between the two intensity classes. Intensities at or below the chosen threshold become foreground.</p>
  <p>Both methods apply a 5 × 5 elliptical opening and closing. Opening removes small spots, while closing fills small gaps. Eight-connected components smaller than 100 pixels are discarded. The area of each retained region is the number of pixels assigned to that component. Its perimeter is traced for the final overlay. Every output uses the benchmark image's pixel resolution.</p>
  <p>The benchmark has {total} cases. The five source scans are evaluated at 465 × 638. For each scan, uniform dimming applies a gain of 0.62 and subtracts 12 intensity levels. The side shadow applies a horizontal gain ramp from 0.42 to 1.03. Both preserve leaf geometry. Two composites paste three resized leaf cutouts at different offsets on the same plain background. Reference masks use manually traced polygon outlines. Composite masks use the union of the projected silhouettes, while their reference instance count comes from the number of pasted leaves.</p>
  <p>IoU is the intersection of predicted and reference foreground divided by their union. Dice is twice the intersection divided by the sum of foreground areas. Mean absolute area error averages the absolute difference between measured and reference areas as a percentage of reference area. Count matches report the number of images where the component count equals the expected instance count.</p>
</section>
<section>
  <h2>Results by condition</h2>
  <div class="table"><table>
    <thead><tr><th scope="col">Condition</th><th scope="col">Method</th><th scope="col">Count matches</th><th scope="col">Raw IoU</th><th scope="col">Final IoU</th><th scope="col">Dice</th><th scope="col">Mean absolute area error</th></tr></thead>
    <tbody>{condition_rows}</tbody>
  </table></div>
  <p>IoU and Dice are per-image means within each condition. Raw IoU is measured before opening, closing and component filtering.</p>
  <ul>
    <li><strong>Original scans.</strong> Mean IoU was {original_fixed:.3f} for the fixed threshold and {original_otsu:.3f} for Otsu. Both counted each original leaf correctly. Mask cleanup changed mean IoU only slightly on these scans because most errors are along contours and areas of low contrast.</li>
    <li><strong>Uniform dimming.</strong> Otsu retained {dim_otsu:.3f} mean IoU, while the fixed method fell to {dim_fixed:.3f}. The fixed method marked all {full_image:,} pixels in each dim scan as foreground. Its count of one matched the expected number despite the failed mask.</li>
    <li><strong>Side shadow.</strong> The leaf and dark background occupied overlapping intensity ranges. Otsu's mean IoU fell to {shadow_otsu:.3f}. More even lighting, illumination correction or local thresholding could address the background gradient. Opening and closing cannot remove a connected background band of this size.</li>
    <li><strong>Overlap.</strong> Otsu detected {separated} regions in the separated scene and {touching} in the touching scene despite a foreground IoU of {touching_iou:.3f} there. The fixed method returned the same counts. Marker-controlled watershed could help with lightly touching leaves. Heavy overlap may require instance segmentation, and the hidden leaf area cannot be recovered from a single view without additional assumptions.</li>
  </ul>
  <p><strong>Limitations.</strong> Five related flatbed scans provide a small sample, and the derived images share source objects. Manual boundaries approximate fine tips and petioles. The stress tests simulate lighting changes on the scans. Field performance would need independent, annotated camera images. Converting projected area to physical units requires calibration at the object plane.</p>
</section>
<section>
  <h2>Object counts and estimated area</h2>
  <div class="table"><table>
    <thead><tr><th scope="col">Image</th><th scope="col">Ref. count</th><th scope="col">Ref. px²</th><th scope="col">Fixed count</th><th scope="col">Fixed px²</th><th scope="col">Otsu count</th><th scope="col">Otsu px²</th></tr></thead>
    <tbody>{count_rows}</tbody>
  </table></div>
  <p>Areas in this table sum all regions found by the method. Individual region measurements and bounding boxes are in the object CSV.</p>
  <p><a href="outputs/summary.csv">Image-level results CSV</a> ·
     <a href="outputs/objects.csv">Per-object measurements CSV</a> ·
     <a href="VIVA.md">Demonstration and viva guide</a></p>
</section>
<section>
  <h2>Source images</h2>
  <p>John Lagergren et al., <em>Few-Shot Learning Enables Population-Scale Analysis of Leaf Traits in Populus trichocarpa</em>, Plant Phenomics. The five original JPEGs are included under <code>data/originals</code>. <a href="https://doi.org/10.34133/plantphenomics.0072">Publication</a> · <a href="https://github.com/jlager/few-shot-leaf-segmentation">Dataset repository</a></p>
</section>
</main>
<script>
const cases = {case_json};
const dropdown = document.getElementById("case");
const integer = new Intl.NumberFormat("en");

function updateCase() {{
  const image = dropdown.value;
  const comparison = cases[image];
  const rows = [
    ["Fixed global", comparison.global],
    ["Otsu", comparison.otsu],
  ].map(([name, row]) => {{
    const tr = document.createElement("tr");
    [name, row.threshold, row.objects, integer.format(row.area_px), Number(row.iou).toFixed(3)]
      .forEach((value, index) => {{
        const cell = document.createElement(index === 0 ? "th" : "td");
        if (index === 0) cell.scope = "row";
        cell.textContent = value;
        tr.append(cell);
      }});
    return tr;
  }});
  document.getElementById("selected-results").replaceChildren(...rows);
  const source = `outputs/${{image}}/comparison.png`;
  document.getElementById("comparison").src = source;
  document.getElementById("image-link").href = source;
  document.getElementById("input").href = `data/inputs/${{image}}.png`;
  document.getElementById("raw").href = `outputs/${{image}}/otsu/raw_mask.png`;
  document.getElementById("clean").href = `outputs/${{image}}/otsu/mask.png`;
  document.getElementById("reference").textContent =
    `Reference — count ${{comparison.otsu.reference_objects}}, visible area ${{integer.format(comparison.otsu.reference_area_px)}} px².`;
}}
dropdown.addEventListener("change", updateCase);
updateCase();
</script>
</html>
"""


if __name__ == "__main__":
    main()
