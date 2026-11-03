#!/usr/bin/env node
/* Builds the panel deck. Run: NODE_PATH=<node_modules with pptxgenjs, sharp, react, react-icons> node presentation/build_deck.js */
const path = require("path");
const pptxgen = require("pptxgenjs");
const sharp = require("sharp");
const React = require("react");
const ReactDOMServer = require("react-dom/server");
const fa = require("react-icons/fa");
const { applyTheme } = require(process.env.PPTX_THEME_SCRIPT);

const HERE = __dirname;
const FIG = (name) => path.join(HERE, "figures", name);
const REPORT = (name) => path.join(HERE, "..", "report", "figures", name);
const OUTPUT = path.join(HERE, "BRCA_Neurosymbolic_Panel_Deck.pptx");
const DEMO_URL = "https://claude.ai/artifact/6sRCTPzwZyMXaUrjas5XNj";

const THEME = {
  name: "Progression Reasoning",
  headFontFace: "Cambria",
  bodyFontFace: "Calibri",
  colors: { dk1: "12263A", lt1: "FFFFFF", dk2: "12355B", lt2: "E6EEF7", accent1: "1C5CAB", accent2: "C93534", accent3: "12875C", accent4: "C9501F", accent5: "6250D6", accent6: "5B6B7B", hlink: "1C5CAB", folHlink: "6250D6" },
};
/* Listed by part, in speaking order. Parts rise and fall in difficulty: Part 3 is hardest, then 2, 4, and 1. */
const SPEAKERS = ["Sehar Majid Abdi", "Surya Pratap Singh", "Shaurya Pratap Singh", "Vishesh Vigh"];
const AUTHORS = ["Shaurya Pratap Singh", "Surya Pratap Singh", "Vishesh Vigh", "Sehar Majid Abdi"];
const PARTS = ["The problem, the data, and the graphs", "Neural models and what we ruled out", "Symbolic reasoning and our novelty", "From model to patient"];

async function imageSize(file) {
  const { width, height } = await sharp(file).metadata();
  return { width, height };
}

async function iconData(Icon, color) {
  const svg = ReactDOMServer.renderToStaticMarkup(React.createElement(Icon, { color: `#${color}`, size: 256 }));
  const png = await sharp(Buffer.from(svg)).resize(256, 256).png().toBuffer();
  return `image/png;base64,${png.toString("base64")}`;
}

async function main() {
  const pres = new pptxgen();
  pres.layout = "LAYOUT_WIDE";
  pres.title = "Neuro-Symbolic Reasoning over Patient-Specific Knowledge Graphs for Breast Cancer Progression Prediction";
  pres.author = AUTHORS.join(", ");
  pres.company = "MNNIT Allahabad";
  pres.theme = { headFontFace: THEME.headFontFace, bodyFontFace: THEME.bodyFontFace };
  const C = pres.SchemeColor;

  pres.defineSlideMaster({
    title: "TITLE",
    background: { color: C.text2 },
    objects: [
      { placeholder: { options: { name: "title", type: "title", x: 0.8, y: 1.15, w: 11.7, h: 2.5, fontSize: 38, bold: true, color: C.background1, align: "left", valign: "top", margin: 0 }, text: "" } },
      { placeholder: { options: { name: "body", type: "body", x: 0.8, y: 3.95, w: 11.7, h: 2.6, fontSize: 18, color: C.background2, align: "left", valign: "top", margin: 0 }, text: "" } },
    ],
  });
  pres.defineSlideMaster({
    title: "SECTION",
    background: { color: C.text2 },
    objects: [
      { placeholder: { options: { name: "title", type: "title", x: 0.8, y: 2.45, w: 11.7, h: 1.3, fontSize: 40, bold: true, color: C.background1, align: "left", valign: "top", margin: 0 }, text: "" } },
      { placeholder: { options: { name: "body", type: "body", x: 0.8, y: 3.85, w: 11.7, h: 1.8, fontSize: 20, color: C.background2, align: "left", valign: "top", margin: 0 }, text: "" } },
    ],
  });
  ["1", "2", "3", "4", "B"].forEach((part) => {
    const label = part === "B" ? "Backup" : `Part ${part}: ${PARTS[Number(part) - 1]}`;
    pres.defineSlideMaster({
      title: `CONTENT${part}`,
      background: { color: C.background1 },
      objects: [
        { placeholder: { options: { name: "title", type: "title", x: 0.6, y: 0.35, w: 12.13, h: 0.9, fontSize: 28, bold: true, color: C.text2, align: "left", valign: "middle", margin: 0 }, text: "" } },
        { text: { text: label, options: { x: 0.6, y: 7.02, w: 6, h: 0.3, fontSize: 11, color: C.accent6, margin: 0 } } },
        { text: { text: "MNNIT Allahabad, BTech project", options: { x: 6.7, y: 7.02, w: 5.4, h: 0.3, fontSize: 11, color: C.accent6, align: "right", margin: 0 } } },
      ],
      slideNumber: { x: 12.2, y: 7.02, w: 0.53, h: 0.3, fontSize: 11, color: THEME.colors.accent6, align: "right" },
    });
  });

  const sizes = new Map();
  const fit = async (slide, file, x, y, w, h, altText) => {
    if (!sizes.has(file)) sizes.set(file, await imageSize(file));
    const size = sizes.get(file);
    const scale = Math.min(w / size.width, h / size.height);
    const iw = size.width * scale, ih = size.height * scale;
    slide.addImage({ path: file, x: x + (w - iw) / 2, y: y + (h - ih) / 2, w: iw, h: ih, altText });
  };
  const card = (slide, x, y, w, h, fill = C.background2, name = "Card") => slide.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w, h, fill: { color: fill }, line: { color: fill, width: 0 }, rectRadius: 0.08, objectName: name });
  const text = (slide, value, x, y, w, h, options = {}) => slide.addText(value, { x, y, w, h, fontSize: 16, color: C.text1, margin: 0, valign: "top", isTextBox: true, ...options });
  const bullets = (slide, items, x, y, w, h, size = 16, gap = 8) => slide.addText(items.map((item, i) => ({ text: item, options: { bullet: { indent: 16 }, breakLine: i < items.length - 1, paraSpaceAfter: gap } })), { x, y, w, h, fontSize: size, color: C.text1, margin: 0, valign: "top", isTextBox: true });
  const stat = (slide, value, label, x, y, w, h, fill = C.background2, ink = C.text2) => {
    card(slide, x, y, w, h, fill, `Stat ${label}`);
    text(slide, value, x + 0.15, y + 0.14, w - 0.3, h * 0.55, { fontFace: "+mj-lt", fontSize: 34, bold: true, color: ink, valign: "middle" });
    text(slide, label, x + 0.15, y + h * 0.6, w - 0.3, h * 0.36, { fontSize: 14, color: fill === C.text2 ? C.background1 : C.text1 });
  };
  const icon = async (slide, Icon, x, y, size = 0.56, fill = C.text2) => {
    slide.addShape(pres.shapes.OVAL, { x, y, w: size, h: size, fill: { color: fill }, line: { color: fill, width: 0 }, objectName: "Icon circle" });
    slide.addImage({ data: await iconData(Icon, "FFFFFF"), x: x + size * 0.24, y: y + size * 0.24, w: size * 0.52, h: size * 0.52, altText: "" });
  };
  const head = (slide, value, x, y, w, color = C.text2) => text(slide, value, x, y, w, 0.4, { fontSize: 18, bold: true, color });
  const table = (slide, rows, x, y, w, colW, size = 14, rowH = 0.42) => {
    const body = rows.map((row, r) => row.map((cell, c) => {
      const options = typeof cell === "object" ? cell.options || {} : {};
      const value = typeof cell === "object" ? cell.text : cell;
      return { text: String(value), options: { fill: { color: r === 0 ? C.text2 : options.fill || C.background1 }, color: r === 0 ? C.background1 : C.text1, bold: r === 0 || options.bold, align: c === 0 ? "left" : options.align || "left", valign: "middle", fontSize: size, margin: [0.03, 0.08, 0.03, 0.08] } };
    }));
    slide.addTable(body, { x, y, w, colW, rowH, border: { type: "solid", pt: 0.75, color: "D8E0E8" }, autoPage: false, objectName: "Table" });
  };
  const barChart = (slide, labels, values, colors, x, y, w, h, options = {}) => slide.addChart(pres.charts.BAR, [{ name: options.series || "Value", labels, values }], {
    x, y, w, h, barDir: "bar", chartColors: colors, invertedColors: colors, barGapWidthPct: 45, showLegend: false, showValue: true, dataLabelPosition: "outEnd", dataLabelFontSize: 13, dataLabelColor: THEME.colors.dk1, dataLabelFontFace: "+mn-lt", dataLabelFormatCode: options.format || "0.000",
    catAxisLabelFontSize: 13, catAxisLabelColor: THEME.colors.dk1, catAxisLabelFontFace: "+mn-lt", catAxisOrientation: "maxMin", valAxisHidden: true, valAxisMinVal: options.min ?? 0, valAxisMaxVal: options.max, valGridLine: { style: "none" }, catGridLine: { style: "none" },
    showTitle: !!options.title, title: options.title, titleFontSize: 15, titleColor: THEME.colors.dk2, titleFontFace: "+mn-lt", altText: options.alt,
  });

  const SEC = ["Part 1: Problem and data", "Part 2: Neural models", "Part 3: Symbolic reasoning", "Part 4: Model to patient", "Backup"];
  const content = (part, sectionIndex) => pres.addSlide({ masterName: `CONTENT${part}`, sectionTitle: SEC[sectionIndex] });

  // ===== Part 1 =====
  pres.addSection({ title: SEC[0] });
  let s = pres.addSlide({ masterName: "TITLE", sectionTitle: SEC[0] });
  s.addText("Neuro-Symbolic Reasoning over Patient-Specific Knowledge Graphs for Breast Cancer Progression Prediction", { placeholder: "title" });
  s.addText([
    { text: "Shaurya Pratap Singh (20233262)   Surya Pratap Singh (20233283)", options: { breakLine: true } },
    { text: "Vishesh Vigh (20233583)   Sehar Majid Abdi (20233561)", options: { breakLine: true } },
    { text: " ", options: { breakLine: true } },
    { text: "Supervisor: Dr. Shashwati Banerjea", options: { breakLine: true } },
    { text: "Department of Computer Science and Engineering, MNNIT Allahabad", options: {} },
  ], { placeholder: "body" });
  s.addNotes("Good morning, respected panel members. We are presenting our BTech project on neuro-symbolic reasoning for breast cancer progression, carried out under the supervision of Dr. Shashwati Banerjea. Four of us will each take one part. The full mathematics is in our report, so today we will focus on the ideas and the results.");

  s = content(1, 0);
  s.addText("What we will cover", { placeholder: "title" });
  const topics = [[fa.FaHospitalUser, "The clinical question, the data, and how each patient becomes a graph"], [fa.FaProjectDiagram, "Graph neural networks, what we dropped, and why accuracy saturates"], [fa.FaBalanceScale, "Learned logic in place of IF-THEN rules, and our novelty"], [fa.FaFileMedicalAlt, "The final model, explanations, patient reports, and a live demo"]];
  for (let i = 0; i < 4; i++) {
    const x = 0.6 + i * 3.08;
    card(s, x, 1.55, 2.89, 4.9);
    await icon(s, topics[i][0], x + 0.25, 1.8, 0.7);
    text(s, `Part ${i + 1}`, x + 0.25, 2.7, 2.4, 0.35, { fontSize: 14, bold: true, color: C.accent6 });
    text(s, PARTS[i], x + 0.25, 3.05, 2.4, 1.0, { fontSize: 18, bold: true, color: C.text2 });
    text(s, topics[i][1], x + 0.25, 4.1, 2.4, 1.5, { fontSize: 15 });
    text(s, SPEAKERS[i], x + 0.25, 5.8, 2.4, 0.4, { fontSize: 14, bold: true });
  }
  s.addNotes("Here is the plan. I will set up the clinical problem and the data. Surya will cover the neural models and what we ruled out. Shaurya will explain the logic reasoning, which is where our novelty lies. Vishesh will bring it together, show the patient reports, and run the live demo.");

  s = content(1, 0);
  s.addText("The question: will it progress, and when?", { placeholder: "title" });
  const why = [[fa.FaUsers, "Common", "About 2.3 million new breast cancer cases in 2020"], [fa.FaHourglassHalf, "Unpredictable", "Some never relapse, some progress in months, some after a decade"], [fa.FaQuestionCircle, "Needs reasons", "Doctors can only act on a prediction they can check"]];
  for (let i = 0; i < 3; i++) {
    const y = 1.55 + i * 1.65;
    card(s, 0.6, y, 5.7, 1.5);
    await icon(s, why[i][0], 0.82, y + 0.42, 0.66);
    text(s, why[i][1], 1.7, y + 0.2, 4.4, 0.4, { fontSize: 20, bold: true, color: C.text2 });
    text(s, why[i][2], 1.7, y + 0.66, 4.4, 0.75);
  }
  head(s, "Our system must", 6.7, 1.55, 6);
  ["Estimate the risk of progression", "Estimate the month it may happen", "Explain why, in checkable steps", "Write a clinical report for each patient"].forEach((task, i) => {
    const y = 2.1 + i * 0.86;
    s.addShape(pres.shapes.OVAL, { x: 6.7, y, w: 0.56, h: 0.56, fill: { color: C.text2 }, line: { color: C.text2, width: 0 }, objectName: `Step ${i + 1}` });
    text(s, String(i + 1), 6.7, y, 0.56, 0.56, { align: "center", valign: "middle", color: C.background1, bold: true, fontSize: 17 });
    text(s, task, 7.45, y, 5.2, 0.56, { valign: "middle", fontSize: 18 });
  });
  text(s, "Two firm rules: no outcome data in the inputs, and no patient is predicted by a model that saw them.", 6.7, 5.65, 6.03, 0.8, { fontSize: 16, bold: true, color: C.text2 });
  s.addNotes("Breast cancer is the most commonly diagnosed cancer worldwide, and its course after treatment is hard to predict. Doctors want to know whether it will progress and when, and a score alone is not enough: they need reasons they can check. So our system has four jobs: estimate risk, estimate the month, explain why, and write a report. We held two rules throughout. No outcome data enters the inputs, and no patient is ever predicted by a model that saw them.");

  s = content(1, 0);
  s.addText("What a tumour record tells a clinician", { placeholder: "title" });
  const domain = [["Stage: how far it has spread", ["T: size of the tumour", "N: spread to lymph nodes", "M: spread to distant organs", "Stage I to IV sums these up"], fa.FaLayerGroup], ["Subtype: how it behaves", ["Luminal A: slower growing", "Luminal B: faster growing", "HER2-enriched: too much ERBB2", "Basal-like: aggressive"], fa.FaMicroscope], ["Alterations: what changed in the DNA", ["Mutation: a small change", "Amplification: extra gene copies", "Deletion: lost gene copies"], fa.FaDna]];
  for (let i = 0; i < 3; i++) {
    const x = 0.6 + i * 4.11;
    card(s, x, 1.55, 3.91, 4.9);
    await icon(s, domain[i][2], x + 0.25, 1.78, 0.62);
    text(s, domain[i][0], x + 0.25, 2.55, 3.45, 0.8, { fontSize: 19, bold: true, color: C.text2 });
    bullets(s, domain[i][1], x + 0.25, 3.45, 3.45, 2.8, 16, 9);
  }
  s.addNotes("A quick piece of background. Stage tells us how far the cancer has spread, using the T, N, and M codes. Subtype tells us how the tumour tends to behave. Alterations tell us what changed in the tumour's DNA. Our patient graphs are built from these three kinds of information.");

  s = content(1, 0);
  s.addText("The hard part: most patients have no final answer", { placeholder: "title" });
  await fit(s, FIG("fig_censoring.png"), 0.6, 1.5, 7.6, 4.9, "Illustration of six patient timelines, some ending in progression and some censored");
  card(s, 8.5, 1.55, 4.23, 4.75);
  stat(s, "145", "patients progressed", 8.72, 1.75, 3.8, 1.3, C.background1);
  stat(s, "938", "left follow-up with no event", 8.72, 3.2, 3.8, 1.3, C.background1);
  text(s, "This is called censoring. No event yet does not mean cured.", 8.75, 4.72, 3.75, 1.4, { fontSize: 17, bold: true, color: C.text2 });
  s.addNotes("This idea drives most of our design decisions. 145 patients have a recorded progression. The other 938 simply left follow-up with nothing recorded. Patient B here left at month 30, and we do not know what happened next. This is called censoring. It is like checking who has finished a book after one week: still reading does not mean never finishing.");

  s = content(1, 0);
  s.addText("Data: real patients and a public knowledge graph", { placeholder: "title" });
  const stats = [["1,083", "breast cancer patients from TCGA"], ["145", "recorded progressions"], ["500", "genes in our panel, including 27 known drivers"], ["8.1M", "connections in PrimeKG"], ["4,570", "PrimeKG connections kept for our genes"], ["3", "kinds of patient data: clinical, mutation, copy number"]];
  stats.forEach(([value, label], i) => stat(s, value, label, 0.6 + (i % 3) * 4.11, 1.55 + Math.floor(i / 3) * 2.45, 3.91, 2.25));
  s.addNotes("Our patient data is the public TCGA breast cancer study. We use clinical records, mutations, and copy-number changes. To keep graphs manageable we use a panel of 500 genes, including 27 known drivers. PrimeKG is a public map of biomedical knowledge linking genes to pathways, drugs, and diseases. We keep the 4,570 connections that touch our genes.");

  s = content(1, 0);
  s.addText("Each patient becomes a knowledge graph", { placeholder: "title" });
  await fit(s, REPORT("fig_fused_graph.png"), 0.6, 1.45, 7.2, 5.2, "Fused graph linking a patient's altered genes to proteins, pathways, and a disease");
  head(s, "How it is built", 8.1, 1.6, 4.6);
  bullets(s, ["Patient connects to the tumour sample", "Sample connects to the altered genes", "Genes connect to PrimeKG pathways, drugs, and diseases"], 8.1, 2.15, 4.63, 2.2, 17, 10);
  card(s, 8.1, 4.45, 4.63, 1.95);
  text(s, "Why it helps: two patients with different genes can meet at the same pathway.", 8.33, 4.62, 4.2, 1.6, { fontSize: 18, bold: true, color: C.text2, valign: "middle" });
  s.addNotes("Each patient becomes a small graph: the patient connects to the tumour sample, which connects to the altered genes. We then connect those genes to PrimeKG pathways, drugs, and breast diseases. The benefit is simple: two patients with different genes can meet at the same pathway, which a plain gene list can never show. That completes the data side. Surya will now take you through the neural models.");

  // ===== Part 2 =====
  pres.addSection({ title: SEC[1] });
  s = content(2, 1);
  s.addText("How we measured everything honestly", { placeholder: "title" });
  await fit(s, FIG("fig_cross_validation.png"), 0.6, 1.45, 7.4, 3.1, "Five-fold cross-validation grid");
  text(s, "Every patient is predicted by models that never saw them.", 0.6, 4.75, 7.4, 0.5, { fontSize: 18, bold: true, color: C.text2 });
  text(s, "A tested guard blocks outcome columns from every input.", 0.6, 5.3, 7.4, 0.5);
  card(s, 8.3, 1.55, 4.43, 4.75);
  head(s, "Main score: C-index", 8.55, 1.75, 3.95);
  text(s, "Take two patients where we know who progressed first. Did the model give that patient the higher risk?", 8.55, 2.3, 3.95, 1.65, { fontSize: 17 });
  stat(s, "0.5", "coin flip", 8.55, 4.1, 1.88, 1.95, C.background1);
  stat(s, "1.0", "perfect", 10.62, 4.1, 1.88, 1.95, C.background1);
  s.addNotes("Thank you, Sehar. I will start with how we measured everything. We split patients into five groups, train on four, and test on the fifth, so every patient is predicted by models that never saw them. Our main score is the C-index. Take two patients where we know who progressed first, and ask whether the model gave that patient the higher risk. 0.5 is a coin flip and 1.0 is perfect.");

  s = content(2, 1);
  s.addText("The first label failed, so we built a better one", { placeholder: "title" });
  card(s, 0.6, 1.55, 4.1, 4.9);
  head(s, "First try", 0.85, 1.75, 3.6);
  text(s, "0.54", 0.85, 2.2, 3.6, 0.9, { fontFace: "+mj-lt", fontSize: 44, bold: true, color: C.accent2 });
  text(s, "AUROC, close to chance, because censored patients were counted as negatives", 0.85, 3.15, 3.6, 1.2);
  head(s, "Better question", 0.85, 4.45, 3.6);
  text(s, "Did the patient progress within 36 months?", 0.85, 4.9, 3.6, 1.0);
  barChart(s, ["12 months (916 patients)", "24 months (629)", "36 months (498)", "48 months (406)", "60 months (344)"], [0.572, 0.674, 0.700, 0.719, 0.654], ["9FB6CC", "9FB6CC", THEME.colors.dk2, "9FB6CC", "9FB6CC"], 5.0, 1.5, 7.73, 4.1, { title: "Clinical AUROC by time horizon", min: 0.4, max: 0.8, series: "AUROC", alt: "Clinical AUROC at five horizons, 36 months chosen" });
  text(s, "36 months balances enough events against too few patients.", 5.0, 5.75, 7.73, 0.6, { fontSize: 16 });
  s.addNotes("Our first try counted censored patients as negatives, and the model was close to chance. So we asked a cleaner question: did the patient progress within 36 months? We tested several horizons. Short ones have too few events, and long ones lose too many patients. 48 months looks slightly higher, but that gap is within noise, so 36 months is the balanced choice. Later, survival models use all 1,083 patients.");

  s = content(2, 1);
  s.addText("No graph model beat the clinical baseline", { placeholder: "title" });
  barChart(s, ["Clinical baseline", "RGAT", "GCN", "GraphSAGE", "GATv2", "R-GCN", "Graph transformer", "GIN", "Gene-list MLP"], [0.703, 0.694, 0.688, 0.679, 0.638, 0.627, 0.613, 0.590, 0.570], [THEME.colors.accent6, THEME.colors.accent1, THEME.colors.accent1, THEME.colors.accent1, THEME.colors.accent1, THEME.colors.accent1, THEME.colors.accent1, THEME.colors.accent1, "9FB6CC"], 0.6, 1.4, 7.9, 5.3, { title: "AUROC at 36 months", min: 0.4, max: 0.78, series: "AUROC", alt: "Clinical baseline 0.703 leads every graph model" });
  card(s, 8.8, 1.55, 3.93, 4.95);
  head(s, "What a GNN does", 9.03, 1.72, 3.5);
  text(s, "Each node learns from its neighbours. After three layers, pathway information reaches the patient.", 9.03, 2.2, 3.5, 1.65);
  head(s, "What we found", 9.03, 3.95, 3.5);
  text(s, "Seven GNNs came close, but none beat age, stage, and subtype. Gaps are smaller than the noise.", 9.03, 4.43, 3.5, 1.8);
  s.addNotes("A graph neural network lets each node learn from its neighbours, so after three layers pathway information reaches the patient. We tested seven architectures. The clinical baseline, which uses only age, sex, subtype, and stage, stayed at the top. RGAT and GCN came close, but the gaps are smaller than the variation between folds. Training on survival-focused losses did not change that.");

  s = content(2, 1);
  s.addText("Why we did not use graph structure learning", { placeholder: "title" });
  await fit(s, FIG("fig_hops.png"), 0.6, 1.45, 6.3, 3.3, "Hops around a patient and the fixed graph boundary");
  text(s, "GSL lets the model learn which graph connections to keep.", 0.6, 4.95, 6.3, 0.75, { fontSize: 17, bold: true, color: C.text2 });
  head(s, "We tried it twice", 7.25, 1.5, 5.4);
  bullets(s, ["First version switched off useful connections, and GCN got worse", "Second version recovered, but stayed within noise"], 7.25, 2.05, 5.48, 1.75, 17, 9);
  head(s, "Why we left it out", 7.25, 3.9, 5.4, C.accent2);
  bullets(s, ["No reliable gain", "Reports should show real, stored connections", "Only 3 of our 7 models can use it"], 7.25, 4.45, 5.48, 1.9, 17, 9);
  s.addNotes("A natural question is why we did not use graph structure learning. It lets the model learn which connections matter for each patient. We tried it twice. The first version switched off useful connections, and GCN got worse. The second version recovered, but its gain stayed within noise. We left it out because there was no reliable gain, our explanations should show real stored connections, and only three of our seven models can use it. The numbers are in the report.");

  s = content(2, 1);
  s.addText("Other ideas we tested and set aside", { placeholder: "title" });
  table(s, [["Idea", "What happened", "Decision"], ["RGAT on all patients", "Too slow on a laptop", "Dropped"], ["Deeper or shallower graphs", "Almost no change", "Kept 2 hops"], ["SMOTE oversampling", "GNNs near chance", "Dropped"], ["External GSE2034 graphs", "Linear model still won", "Same lesson"], ["Random survival forest", "Lowest score of all", "Baseline only"]], 0.6, 1.55, 12.13, [4.3, 4.8, 3.03], 17, 0.72);
  s.addNotes("Here are other ideas we tested. RGAT was too slow for all patients on our hardware. Changing graph depth barely mattered. Oversampling did not help. On an external cohort, a linear model again beat the graph models. The random survival forest was the weakest of all, because 145 events is too few for it. The report has the numbers.");

  s = content(2, 1);
  s.addText("Every model family stops near 0.67", { placeholder: "title" });
  await fit(s, FIG("fig_saturation.png"), 0.6, 1.4, 8.0, 5.3, "C-index of twelve models with spread and the tie bar at 0.659");
  card(s, 8.85, 1.55, 3.88, 4.95, C.text2);
  text(s, "So we changed the goal", 9.08, 1.78, 3.45, 0.8, { fontSize: 22, bold: true, color: C.background1, fontFace: "+mj-lt" });
  text(s, "With 145 events, clinical facts carry most of the signal.", 9.08, 2.75, 3.45, 1.2, { fontSize: 17, color: C.background1 });
  text(s, "More model power did not help, so explanation became the goal.", 9.08, 4.1, 3.45, 1.6, { fontSize: 17, color: C.background1 });
  s.addNotes("This is our main finding. GNNs, logic, forests, and Cox models all finish in the same range, and the lines overlap. With only 145 events, clinical facts carry most of the signal. We accepted that instead of chasing noise. Following our supervisor's direction, the question changed from how to raise the score to how to explain it. Shaurya will explain how.");

  // ===== Part 3 =====
  pres.addSection({ title: SEC[2] });
  s = content(3, 2);
  s.addText("How a survival rule works, and where it falls short", { placeholder: "title" });
  const node = (x, y, w, h, fill, lines, name) => {
    card(s, x, y, w, h, fill, name);
    s.addText(lines, { x: x + 0.15, y, w: w - 0.3, h, margin: 0, valign: "middle", align: "center", isTextBox: true, objectName: `${name} text` });
  };
  const link = (x, y, w, h, flipH, label, lx) => {
    s.addShape(pres.shapes.LINE, { x, y, w, h, flipH, line: { color: THEME.colors.accent6, width: 1.75, endArrowType: "triangle" }, objectName: `Arrow ${label}` });
    text(s, label, lx, y + 0.1, 0.6, 0.35, { fontSize: 15, bold: true, color: C.accent6, align: "center" });
  };
  node(2.3, 1.5, 4.0, 0.72, C.text2, [{ text: "865 training patients", options: { fontSize: 18, bold: true, color: C.background1 } }], "All patients");
  s.addShape(pres.shapes.LINE, { x: 4.3, y: 2.22, w: 0, h: 0.38, line: { color: THEME.colors.accent6, width: 1.75, endArrowType: "triangle" }, objectName: "Arrow to rule" });
  node(0.75, 2.6, 7.1, 0.95, C.background2, [{ text: "IF nodal stage is N3 AND aneuploidy is below 26.5", options: { fontSize: 19, bold: true, color: C.text2, fontFace: "+mj-lt" } }], "Rule");
  link(2.32, 3.55, 1.98, 0.7, true, "yes", 2.45);
  link(4.3, 3.55, 1.82, 0.7, false, "no", 5.55);
  node(0.6, 4.25, 3.45, 1.6, "F9E3E2", [{ text: "Rule holds", options: { fontSize: 15, bold: true, color: C.accent2, breakLine: true } }, { text: "46 patients", options: { fontSize: 24, bold: true, color: C.text1, fontFace: "+mj-lt", breakLine: true } }, { text: "13 progressed: 5 times the hazard", options: { fontSize: 15, color: C.text1 } }], "Covered group");
  node(4.4, 4.25, 3.45, 1.6, C.background2, [{ text: "Rule does not hold", options: { fontSize: 15, bold: true, color: C.accent6, breakLine: true } }, { text: "819 patients", options: { fontSize: 24, bold: true, color: C.text1, fontFace: "+mj-lt", breakLine: true } }, { text: "everyone else", options: { fontSize: 15, color: C.text1 } }], "Other group");
  text(s, "RuleKit picks the conditions that split survival most sharply.", 0.6, 6.05, 7.25, 0.5, { fontSize: 16, color: C.accent6 });
  head(s, "Where IF-THEN falls short", 8.4, 1.5, 4.33);
  [["8", "rules per fold"], ["10%", "use the knowledge graph"], ["0", "weights: a rule is on or off"]].forEach(([value, label], i) => {
    const y = 2.05 + i * 1.18;
    card(s, 8.4, y, 4.33, 1.02, C.background2, `Limit ${label}`);
    text(s, value, 8.6, y, 1.3, 1.02, { fontFace: "+mj-lt", fontSize: 32, bold: true, color: C.text2, valign: "middle" });
    text(s, label, 9.9, y, 2.7, 1.02, { fontSize: 16, valign: "middle" });
  });
  text(s, "Readable, but the rules hardly used our knowledge graph.", 8.4, 5.7, 4.33, 0.85, { fontSize: 17, bold: true, color: C.text2 });
  s.addNotes("Thank you, Surya. Let me start with what a survival rule does. A rule takes all the training patients and splits them into two groups. Here is a real rule our baseline learned: if nodal stage is N3 and aneuploidy is below 26.5. It picks out 46 of 865 patients, and 13 of them progressed, at about five times the hazard of everyone else. RuleKit learns rules like this by choosing the conditions that split survival most sharply, and that is why rules are attractive: a doctor can read them. But they have limits. RuleKit kept only about eight rules per fold, each one is simply on or off with no weight, and only 10 percent used the knowledge graph we had built. Our supervisor suggested three ways to go further: Markov logic, soft logic, and reinforcement learning. We built all three.");

  s = content(3, 2);
  s.addText("Idea 1: Markov logic gives every rule a weight", { placeholder: "title" });
  card(s, 0.6, 1.55, 5.95, 2.75);
  head(s, "An ordinary rule", 0.85, 1.72, 5.4);
  text(s, "Fires or does not fire. Like a voter who can only say yes or no.", 0.85, 2.25, 5.45, 1.8, { fontSize: 18 });
  card(s, 6.78, 1.55, 5.95, 2.75, C.text2);
  text(s, "A Markov logic formula", 7.03, 1.72, 5.4, 0.4, { fontSize: 18, bold: true, color: C.background1 });
  text(s, "Has a learned weight. Like a voter with a volume knob, and many can speak at once.", 7.03, 2.25, 5.45, 1.8, { fontSize: 18, color: C.background1 });
  head(s, "Our change", 0.6, 4.55, 7.3, C.accent2);
  text(s, "We learn the weights from survival times, so a patient who has not relapsed yet is never treated as cured. Useless formulas drop to zero.", 0.6, 5.0, 7.5, 1.5, { fontSize: 18 });
  stat(s, "422", "facts about each patient, 80% from PrimeKG", 8.4, 4.55, 4.33, 1.9);
  s.addNotes("The first idea is Markov logic. An ordinary rule is like a voter who can only say yes or no. A Markov logic formula has a learned weight, like a volume knob, and many can speak at once. Our change is how the weights are learned. A standard version would treat a patient who has not relapsed yet as cured. We learn from survival times instead, and useless formulas drop to zero. All reasoners read the same 422 facts about each patient. The equations are in the report.");

  s = content(3, 2);
  s.addText("Idea 2: soft logic lets similar patients share risk", { placeholder: "title" });
  card(s, 0.6, 1.55, 5.95, 2.5);
  head(s, "Soft truth", 0.85, 1.72, 5.4);
  text(s, "Facts can be partly true, anywhere from 0 to 1, instead of only true or false.", 0.85, 2.25, 5.45, 1.6, { fontSize: 18 });
  card(s, 6.78, 1.55, 5.95, 2.5);
  head(s, "Reasoning across patients", 7.03, 1.72, 5.4);
  text(s, "If a similar patient is high risk, this patient may be too.", 7.03, 2.25, 5.45, 1.6, { fontSize: 18 });
  stat(s, "≈ 0.01", "weight learned for the similar-patient rule", 0.6, 4.3, 4.3, 2.1, C.background2, C.accent2);
  text(s, "A real finding: in this cohort, similar patients add nothing beyond a patient's own facts.", 5.2, 4.3, 7.53, 2.1, { fontSize: 20, bold: true, color: C.text2, valign: "middle" });
  s.addNotes("The second idea is soft logic. Facts can be partly true, and one rule lets similar patients share risk. We expected this to help, but the model gave that rule a weight of about 0.01 in every fold. So in this cohort, similar patients add nothing beyond a patient's own facts. That is a finding, and we report it as one.");

  s = content(3, 2);
  s.addText("Idea 3: reinforcement learning writes the rules", { placeholder: "title" });
  await fit(s, FIG("fig_rl_loop.png"), 0.6, 1.4, 12.13, 3.75, "RL loop from rule writer to reward, validation filter, and MLN");
  card(s, 0.6, 5.3, 5.95, 1.3);
  text(s, "A rule is rewarded only if it holds on two separate groups of patients.", 0.85, 5.4, 5.45, 1.1, { fontSize: 17, valign: "middle" });
  card(s, 6.78, 5.3, 5.95, 1.3);
  text(s, "Alone it overfits: over 20,000 rules tried per fold, so some look good by luck.", 7.03, 5.4, 5.45, 1.1, { fontSize: 17, valign: "middle", color: C.accent2, bold: true });
  s.addNotes("The third idea is a small neural network that writes rules one fact at a time. It is rewarded only if a rule works on two separate groups of patients, and the rule must then pass patients it never saw. Even so, alone it overfits: it tries over 20,000 rules per fold, so some look good by luck. It searches well but judges badly. That led us to combine it with Markov logic.");

  s = content(3, 2);
  s.addText("Our engine: RL proposes, Markov logic decides", { placeholder: "title" });
  card(s, 0.6, 1.55, 5.95, 4.9);
  await icon(s, fa.FaRobot, 0.85, 1.8, 0.66, C.accent4);
  text(s, "RL searches", 1.68, 1.8, 4.7, 0.66, { fontSize: 20, bold: true, color: C.text2, valign: "middle" });
  text(s, "It finds multi-step knowledge-graph rules that simple search cannot reach.", 0.85, 2.7, 5.45, 1.2, { fontSize: 17 });
  await icon(s, fa.FaBalanceScale, 0.85, 4.05, 0.66, C.text2);
  text(s, "Markov logic judges", 1.68, 4.05, 4.7, 0.66, { fontSize: 20, bold: true, color: C.text2, valign: "middle" });
  text(s, "Good rules earn weight. Bad ones get zero, so RL's weakness stops mattering.", 0.85, 4.95, 5.45, 1.2, { fontSize: 17 });
  s.addChart(pres.charts.DOUGHNUT, [{ name: "Formulas", labels: ["Mined clinical and PrimeKG pairs", "Single facts", "RL-proposed multi-step"], values: [81, 51, 44] }], { x: 6.9, y: 1.45, w: 5.83, h: 4.7, holeSize: 58, chartColors: [THEME.colors.accent1, THEME.colors.accent6, THEME.colors.accent4], showLegend: true, legendPos: "b", legendFontSize: 14, legendColor: THEME.colors.dk1, legendFontFace: "+mn-lt", showPercent: false, showValue: true, dataLabelColor: "FFFFFF", dataLabelFontSize: 14, dataLabelFontFace: "+mn-lt", dataLabelFormatCode: "0", showTitle: true, title: "176 formulas kept over five folds", titleFontSize: 15, titleColor: THEME.colors.dk2, titleFontFace: "+mn-lt", altText: "81 mined, 51 single fact, 44 RL formulas" });
  s.addNotes("So we combined them, and this is the core of our novelty. RL searches for candidate rules, including multi-step knowledge-graph rules. Markov logic then judges them with survival data. Good rules earn weight, and bad ones get zero. Across the five folds the engine kept 176 formulas, and 44 of them came from RL.");

  s = content(3, 2);
  s.addText("Accuracy is a tie; the reasoning is not", { placeholder: "title" });
  const labels = ["Markov logic", "RL + Markov logic (ours)", "RuleKit (baseline)", "Soft logic", "RL alone"];
  const engineColors = (values) => values.map((_, i) => (i === 1 ? THEME.colors.dk2 : i === 2 ? THEME.colors.accent6 : "9FB6CC"));
  barChart(s, labels, [0.630, 0.628, 0.619, 0.604, 0.563], engineColors(labels), 0.6, 1.4, 6.0, 4.0, { title: "C-index", min: 0.4, max: 0.72, series: "C-index", alt: "Markov logic 0.630, ours 0.628, RuleKit 0.619, soft logic 0.604, RL 0.563" });
  barChart(s, labels, [63, 69, 10, 63, 91], engineColors(labels), 6.73, 1.4, 6.0, 4.0, { title: "Formulas using PrimeKG (%)", min: 0, max: 115, format: "0", series: "Share", alt: "Share of formulas using PrimeKG, ours 69 percent, RuleKit 10 percent" });
  card(s, 0.6, 5.5, 12.13, 1.05);
  text(s, "The top three are statistically tied, so we chose the one with the richest reasoning.", 0.85, 5.55, 11.65, 0.95, { fontSize: 18, bold: true, color: C.text2, valign: "middle" });
  s.addNotes("Here is the honest comparison. On the left, the top three are within noise, so we do not claim better accuracy. On the right is the real difference: 69 percent of our formulas use the knowledge graph, against 10 percent for RuleKit. RL alone is higher, but it is not tied on accuracy. So among the tied reasoners we chose the richest one.");

  s = content(3, 2);
  s.addText("What the system learned, in plain words", { placeholder: "title" });
  const examples = [["No lymph-node spread", "lowers risk", "-0.30", "408 patients", C.accent1], ["N3 nodes, and an altered gene linked to ductal carcinoma", "raises risk", "+0.15", "43 patients", C.accent2], ["Altered genes in both the VEGFA-VEGFR2 and ion homeostasis pathways, found by RL", "lowers risk", "-0.14", "93 patients", C.accent1]];
  examples.forEach(([rule, effect, weight, support, color], i) => {
    const y = 1.55 + i * 1.62;
    card(s, 0.6, y, 7.8, 1.46);
    text(s, weight, 0.8, y + 0.12, 1.5, 0.8, { fontFace: "+mj-lt", fontSize: 28, bold: true, color, valign: "middle" });
    text(s, effect, 0.8, y + 0.9, 1.5, 0.4, { fontSize: 14 });
    text(s, rule, 2.4, y + 0.14, 5.8, 0.85, { fontSize: 17, bold: true, color: C.text2, valign: "middle" });
    text(s, support, 2.4, y + 0.98, 5.8, 0.35, { fontSize: 14, color: C.accent6 });
  });
  await fit(s, REPORT("fig_rule_km.png"), 8.7, 1.5, 4.03, 3.6, "Kaplan-Meier curve of patients for whom the RL formula holds");
  text(s, "Patients who match the RL rule, against all patients. These are associations, not causes.", 8.7, 5.2, 4.03, 1.2, { fontSize: 15, color: C.accent6 });
  s.addNotes("Here are real learned formulas, in plain words. No lymph-node spread lowers risk, as doctors would expect. The second joins N3 stage with a knowledge-graph link and raises risk. The third was proposed by RL and joins two pathways, which RuleKit could not express this way. On the right is the survival curve for patients who match it. These are associations, not causes.");

  s = content(3, 2);
  s.addText("Our novelty", { placeholder: "title" });
  const novelty = [[fa.FaWeightHanging, "Weighted logic", "About 35 weighted formulas instead of 8 on/off rules"], [fa.FaHeartbeat, "Survival-aware logic", "Weights learned from survival times, so censoring is respected"], [fa.FaRobot, "RL finds the rules", "With a reward that guards against lucky rules"], [fa.FaProjectDiagram, "Knowledge-graph reasoning", "69% of formulas use PrimeKG, against 10%"], [fa.FaRoute, "Rules inside explanations", "Every path combines neural and logic evidence"], [fa.FaClipboardCheck, "An honest comparison", "All three methods tested, failures included"]];
  for (let i = 0; i < 6; i++) {
    const x = 0.6 + (i % 3) * 4.11, y = 1.55 + Math.floor(i / 3) * 2.5;
    card(s, x, y, 3.91, 2.3);
    await icon(s, novelty[i][0], x + 0.22, y + 0.22, 0.58);
    text(s, novelty[i][1], x + 0.95, y + 0.22, 2.8, 0.7, { fontSize: 17, bold: true, color: C.text2, valign: "middle" });
    text(s, novelty[i][2], x + 0.22, y + 1.02, 3.47, 1.15, { fontSize: 16 });
  }
  s.addNotes("To sum up the novelty. Weighted formulas replaced on-off rules. The logic learns from survival times, so censoring is respected. Reinforcement learning finds the rules, with a reward that guards against luck. Most formulas reason over the knowledge graph, and they now feed every explanation. And we tested all three methods fairly, including the ones that did not work. Vishesh will now show how this reaches the patient.");

  // ===== Part 4 =====
  pres.addSection({ title: SEC[3] });
  s = content(4, 3);
  s.addText("The final model is tied with clinical Cox and can explain itself", { placeholder: "title" });
  await fit(s, FIG("fig_fusion.png"), 0.6, 1.4, 12.13, 3.2, "Clinical, GNN, and logic inputs enter one Cox model that drives risk, month, and paths");
  stat(s, "0.664", "our model", 0.6, 4.8, 2.9, 1.8, C.text2, C.background1);
  stat(s, "0.675", "best clinical model", 3.68, 4.8, 2.9, 1.8);
  card(s, 6.76, 4.8, 5.97, 1.8);
  text(s, "The gap is smaller than the noise, so they are tied. GNN and logic add explanation, not accuracy.", 7.0, 4.9, 5.5, 1.6, { fontSize: 18, bold: true, color: C.text2, valign: "middle" });
  s.addNotes("Thank you, Shaurya. The final prediction comes from one Cox model with three inputs: clinical data, the GNN risk, and the logic risk. It reaches 0.664, against 0.675 for the best clinical model. That gap is smaller than the noise, so the two are statistically tied. The GNN and logic add explanation, not accuracy, and we say so openly. We pay nothing measurable, and we gain reasons.");

  s = content(4, 3);
  s.addText("Estimating the month of progression", { placeholder: "title" });
  stat(s, "48%", "within 12 months of the true month (a constant guess gets 37%)", 0.6, 1.55, 3.7, 2.2);
  stat(s, "33%", "in the right time window (a constant guess gets 19%)", 4.45, 1.55, 3.7, 2.2);
  bullets(s, ["Most patients never progress, so we estimate time only for those who do", "Learned from past progressors only", "Coarse for very early or very late cases, so reports always show the true month beside it"], 0.6, 4.05, 7.55, 2.5, 17, 9);
  await fit(s, REPORT("fig_timing.png"), 8.45, 1.45, 4.28, 4.6, "Estimated against recorded month");
  s.addNotes("Next, the month of progression. Most patients never progress, so we estimate the month only for those who do, using a time-to-event model learned from past progressors. It lands within 12 months for 48 percent of progressors, against 37 percent for a constant guess. It is coarse, so reports always print the true month beside it.");

  s = content(4, 3);
  s.addText("Each path shows why, using two kinds of evidence", { placeholder: "title" });
  card(s, 0.6, 1.5, 12.13, 0.85);
  text(s, "Path score = 0.6 × neural + 0.4 × rule", 0.85, 1.5, 11.6, 0.85, { fontFace: "+mj-lt", fontSize: 26, bold: true, color: C.text2, valign: "middle", align: "center" });
  await fit(s, REPORT("fig_paths_example.png"), 0.6, 2.5, 7.7, 4.25, "Top paths for patient TCGA-BH-A1EV with score split");
  bullets(s, ["Neural: how much the GNN relied on this path", "Rule: learned formulas that hold for this patient and involve this path", "Reports show the strongest 3 to 5 paths"], 8.55, 2.65, 4.18, 3.8, 17, 12);
  s.addNotes("Each path is a real walk in the patient's graph, from an altered gene to a pathway, drug, or disease. Its score combines two things: how much the GNN relied on the path, and the learned formulas that hold for this patient and involve the path. The 0.6 and 0.4 follow our supervisor's design. Paths show what the model relied on, and they are not causal claims.");

  s = content(4, 3);
  s.addText("A report for every patient, grouped by progression time", { placeholder: "title" });
  await fit(s, REPORT("fig_report_p1.png"), 0.6, 1.4, 3.75, 5.4, "Patient report first page");
  await fit(s, REPORT("fig_report_p2.png"), 4.45, 1.4, 3.75, 5.4, "Patient report second page with paths and formulas");
  table(s, [["Group", "Recorded", "Patients"], ["A", "up to 12 months", { text: "32", options: { align: "right" } }], ["B", "12 to 24 months", { text: "30", options: { align: "right" } }], ["C", "24 to 36 months", { text: "27", options: { align: "right" } }], ["D", "over 36 months", { text: "56", options: { align: "right" } }], ["N", "no progression", { text: "938", options: { align: "right" } }]], 8.45, 1.5, 4.28, [0.9, 2.28, 1.1], 15, 0.42);
  text(s, "1,083 PDF reports: record, risk, month, survival curve, paths, and formulas.", 8.45, 4.3, 4.28, 1.3, { fontSize: 17 });
  s.addNotes("The system writes a two to three page PDF for all 1,083 patients, grouped by when they progressed. The first page gives the record, risk, month estimate, and survival curve. The second gives the paths and the formulas that fire. Reports for patients with no recorded progression state that this is not proof of remission, and drug links are marked as research context, not treatment advice.");

  s = content(4, 3);
  s.addText("Live demo: the model running in the browser", { placeholder: "title" });
  await fit(s, FIG("fig_demo.png"), 0.6, 1.4, 7.4, 5.35, "Screenshot of the BRCA Progression Reasoner demo");
  head(s, "Two ways in", 8.3, 1.55, 4.4);
  bullets(s, ["Pick any of the 1,083 real patients", "Or start from a synthetic example and change a few fields"], 8.3, 2.1, 4.43, 1.9, 17, 10);
  head(s, "Checked", 8.3, 4.05, 4.4);
  text(s, "It matches the pipeline's risk for all 1,083 patients.", 8.3, 4.55, 4.43, 1.1, { fontSize: 17 });
  text(s, [{ text: "Open the demo", options: { hyperlink: { url: DEMO_URL, tooltip: "Open the live demo" } } }], 8.3, 5.85, 4.43, 0.4, { fontSize: 18, bold: true, color: C.accent1 });
  s.addNotes("Now the demo. Open the link, or demo/index.html if the internet is unavailable. Step one: show TCGA-BH-A1EV, at the 98th percentile, and point out that its 23-month estimate misses the recorded window, which shows we do not hide errors. Step two: read one path and one formula. Step three: press Change this patient, set N stage to N0, and watch the result update. Step four: open New patient and select the Basal-like example. For new patients the GNN term is held at its average, because a GNN cannot run in the browser.");

  s = content(4, 3);
  s.addText("Limitations we state openly", { placeholder: "title" });
  const limits = [["Accuracy ceiling", "Scores stop near 0.67 with only 145 events."], ["Coarse timing", "Within a year for about half of progressors."], ["Reasoning, not accuracy", "Logic explains more, but does not score higher."], ["Not causal", "Paths show what the model used, not proven biology."], ["Not treatment advice", "Drug links are research context only."], ["One cohort", "No external validation of the reports yet."]];
  limits.forEach(([title, body], i) => {
    const x = i % 2 ? 6.78 : 0.6, y = 1.55 + Math.floor(i / 2) * 1.68;
    card(s, x, y, 5.95, 1.5);
    text(s, title, x + 0.25, y + 0.15, 5.45, 0.4, { fontSize: 18, bold: true, color: C.text2 });
    text(s, body, x + 0.25, y + 0.65, 5.45, 0.75, { fontSize: 17 });
  });
  s.addNotes("We want to be direct about the limits. Accuracy stops near 0.67. Timing is coarse. The logic adds reasoning, not accuracy. Paths are not causal, and drug links are not treatment advice. We used one cohort, and the reports have not been validated externally. This is a research prototype.");

  s = content(4, 3);
  s.addText("Conclusion and future work", { placeholder: "title" });
  card(s, 0.6, 1.55, 6.95, 5.0, C.text2);
  text(s, "What we delivered", 0.9, 1.75, 6.35, 0.5, { fontSize: 22, bold: true, color: C.background1, fontFace: "+mj-lt" });
  s.addText([
    "A leakage-checked pipeline from raw data to patient knowledge graphs",
    "A survival-aware logic engine with RL-proposed formulas",
    "A model tied with the best clinical model, and able to explain itself",
    "1,083 explained reports and a live demo",
  ].map((item, i) => ({ text: item, options: { bullet: { indent: 16 }, breakLine: i < 3, paraSpaceAfter: 14 } })), { x: 0.9, y: 2.45, w: 6.35, h: 3.9, fontSize: 19, color: C.background1, margin: 0, valign: "top", isTextBox: true });
  head(s, "Future work", 7.85, 1.65, 4.85);
  bullets(s, ["Let RL walk knowledge-graph paths directly", "Add gene expression data", "Validate on another cohort such as METABRIC", "Ask oncologists to rate the reports"], 7.85, 2.25, 4.88, 4.2, 18, 13);
  s.addNotes("To conclude: we built a complete pipeline, and turned a saturated prediction problem into an explained one. On this cohort, knowledge-graph reasoning adds explanation rather than accuracy, and the system is designed around that finding. Next, we would let RL walk the knowledge graph directly, add more data, validate on another cohort, and ask oncologists to assess the reports.");

  s = pres.addSlide({ masterName: "SECTION", sectionTitle: SEC[3] });
  s.addText("Thank you", { placeholder: "title" });
  s.addText([{ text: "We welcome your questions.", options: { breakLine: true } }, { text: "Shaurya Pratap Singh, Surya Pratap Singh, Vishesh Vigh, Sehar Majid Abdi", options: { breakLine: true } }, { text: "Supervisor: Dr. Shashwati Banerjea", options: {} }], { placeholder: "body" });
  s.addNotes("Thank you. We welcome your questions. Any of us can take them. The derivations are in our report, and a backup slide has all the model scores.");

  // ===== Backup =====
  pres.addSection({ title: SEC[4] });
  s = content("B", 4);
  s.addText("Backup: all models", { placeholder: "title" });
  table(s, [["Model", "C-index", "Model", "C-index"], ["Clinical Cox (19 fields)", "0.675 ± 0.035", "GraphSAGE, C-index loss", "0.622 ± 0.067"], ["Clinical Cox (extended)", "0.667 ± 0.042", "RuleKit rules", "0.619 ± 0.048"], ["Neuro-symbolic Cox", "0.664 ± 0.021", "GCN, Cox loss", "0.612 ± 0.044"], ["Neuro-symbolic Cox (ext.)", "0.662 ± 0.042", "Survival PSL", "0.604 ± 0.059"], ["GNN + logic, no clinical", "0.642 ± 0.056", "GraphSAGE, Cox loss", "0.603 ± 0.062"], ["Survival MLN", "0.630 ± 0.055", "R-GCN, Cox loss", "0.586 ± 0.041"], ["RL-structured MLN", "0.628 ± 0.055", "RL rules", "0.563 ± 0.046"], ["", "", "Random survival forest", "0.554 ± 0.023"]], 0.6, 1.5, 12.13, [3.9, 2.16, 3.9, 2.17], 15, 0.52);
  s.addNotes("Backup slide for exact numbers. All values are five-fold means on the same patients.");

  await pres.writeFile({ fileName: OUTPUT });
  await applyTheme(OUTPUT, THEME);
  console.log("wrote", OUTPUT);
}

main().catch((error) => { console.error(error); process.exit(1); });
