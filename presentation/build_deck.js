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
const EQ = (name) => path.join(HERE, "figures", "eq", `${name}.png`);
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
    const label = part === "B" ? "Backup" : `Part ${part}: ${SPEAKERS[Number(part) - 1]}`;
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
  /* Equations keep one optical size: `height` is the height of a tall display line. */
  const eq = async (slide, name, x, y, maxW, h, altText) => {
    if (!sizes.has(name)) sizes.set(name, await imageSize(EQ(name)));
    const size = sizes.get(name);
    const scale = Math.min(maxW / size.width, h / size.height);
    slide.addImage({ path: EQ(name), x, y: y + (h - size.height * scale) / 2, w: size.width * scale, h: size.height * scale, altText });
  };
  const card = (slide, x, y, w, h, fill = C.background2, name = "Card") => slide.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w, h, fill: { color: fill }, line: { color: fill, width: 0 }, rectRadius: 0.08, objectName: name });
  const text = (slide, value, x, y, w, h, options = {}) => slide.addText(value, { x, y, w, h, fontSize: 16, color: C.text1, margin: 0, valign: "top", isTextBox: true, ...options });
  const bullets = (slide, items, x, y, w, h, size = 16, gap = 8) => slide.addText(items.map((item, i) => ({ text: item, options: { bullet: { indent: 16 }, breakLine: i < items.length - 1, paraSpaceAfter: gap } })), { x, y, w, h, fontSize: size, color: C.text1, margin: 0, valign: "top", isTextBox: true });
  const stat = (slide, value, label, x, y, w, h, fill = C.background2, ink = C.text2) => {
    card(slide, x, y, w, h, fill, `Stat ${label}`);
    text(slide, value, x + 0.15, y + 0.14, w - 0.3, h * 0.55, { fontFace: "+mj-lt", fontSize: 34, bold: true, color: ink, valign: "middle" });
    text(slide, label, x + 0.15, y + h * 0.6, w - 0.3, h * 0.36, { fontSize: 14, color: C.text1 });
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
  s.addNotes("Good morning, respected panel members. We are presenting our BTech project on neuro-symbolic reasoning for breast cancer progression, carried out under the supervision of Dr. Shashwati Banerjea. Four of us will each take one part.");

  s = content(1, 0);
  s.addText("What we will cover", { placeholder: "title" });
  const topics = [[fa.FaHospitalUser, "The clinical question, the data, and how each patient becomes a graph"], [fa.FaProjectDiagram, "Graph neural networks, what we tested and dropped, and why accuracy saturates"], [fa.FaBalanceScale, "Learned logic in place of IF-THEN rules, and where our novelty lies"], [fa.FaFileMedicalAlt, "Fusion, month estimates, explanation paths, patient reports, and a live demo"]];
  for (let i = 0; i < 4; i++) {
    const x = 0.6 + i * 3.08;
    card(s, x, 1.55, 2.89, 4.9);
    await icon(s, topics[i][0], x + 0.25, 1.8, 0.7);
    text(s, `Part ${i + 1}`, x + 0.25, 2.7, 2.4, 0.35, { fontSize: 14, bold: true, color: C.accent6 });
    text(s, PARTS[i], x + 0.25, 3.05, 2.4, 1.0, { fontSize: 18, bold: true, color: C.text2 });
    text(s, topics[i][1], x + 0.25, 4.1, 2.4, 1.5, { fontSize: 14 });
    text(s, SPEAKERS[i], x + 0.25, 5.8, 2.4, 0.4, { fontSize: 14, bold: true });
  }
  s.addNotes("Here is the plan. I will set up the clinical problem and the data. Surya will cover the neural models and what we ruled out. Shaurya will explain the symbolic reasoning, which is where our novelty lies. Vishesh will bring it together, show the patient reports, and run the live demo.");

  s = content(1, 0);
  s.addText("The clinical question: will it progress, and when?", { placeholder: "title" });
  const why = [[fa.FaUsers, "A common disease", "About 2.3 million new breast cancer cases worldwide in 2020."], [fa.FaHourglassHalf, "An uneven course", "After treatment, some patients never relapse, some progress in months, and some after a decade."], [fa.FaDna, "Too much unread data", "Sequencing lists hundreds of alterations per tumour, and most mean little on their own."], [fa.FaQuestionCircle, "Scores need reasons", "A clinician can only act on a prediction that can be explained and checked."]];
  for (let i = 0; i < 4; i++) {
    const x = i % 2 ? 6.85 : 0.6, y = i < 2 ? 1.6 : 4.0;
    card(s, x, y, 5.88, 2.1);
    await icon(s, why[i][0], x + 0.25, y + 0.28, 0.66);
    text(s, why[i][1], x + 1.15, y + 0.25, 4.5, 0.4, { fontSize: 20, bold: true, color: C.text2 });
    text(s, why[i][2], x + 1.15, y + 0.75, 4.5, 1.2);
  }
  s.addNotes("Breast cancer is the most commonly diagnosed cancer worldwide. What makes it hard is how uneven the course is after treatment. Doctors rely on stage, lymph-node status, and subtype. Sequencing gives hundreds of extra alterations per tumour, but most of them mean little alone. And even a good score is hard to act on if nobody can say why the model produced it. That is the gap we worked on.");

  s = content(1, 0);
  s.addText("Problem statement", { placeholder: "title" });
  head(s, "Given a patient's record and PrimeKG, the system must", 0.6, 1.5, 7.3);
  const tasks = ["Estimate the risk of progression", "Estimate the month of progression, if it happens", "Explain the estimate with knowledge-graph paths and logic", "Present it as a clinical report, grouped by progression time"];
  tasks.forEach((task, i) => {
    const y = 2.05 + i * 1.08;
    s.addShape(pres.shapes.OVAL, { x: 0.6, y, w: 0.6, h: 0.6, fill: { color: C.text2 }, line: { color: C.text2, width: 0 }, objectName: `Step ${i + 1}` });
    text(s, String(i + 1), 0.6, y, 0.6, 0.6, { align: "center", valign: "middle", color: C.background1, bold: true, fontSize: 18 });
    text(s, task, 1.4, y, 6.4, 0.6, { valign: "middle", fontSize: 18 });
  });
  card(s, 8.3, 1.55, 4.43, 4.75);
  head(s, "Two constraints we never relaxed", 8.55, 1.75, 3.95);
  bullets(s, ["Outcome data must never leak into the inputs", "Every reported prediction comes from a model that did not see that patient"], 8.55, 2.75, 3.95, 3.2, 17, 14);
  s.addNotes("Formally, the system has four jobs: estimate risk, estimate the month of progression, explain the estimate, and present it as a report. We also held two rules throughout. No outcome information may enter the inputs, and every prediction we report must come from a model that never saw that patient.");

  s = content(1, 0);
  s.addText("What a tumour record tells a clinician", { placeholder: "title" });
  const domain = [["Stage: how far it has spread", ["T1 to T4: size of the tumour", "N0 to N3: spread to lymph nodes", "M0 or M1: distant spread", "AJCC stage I to IV sums these up"], fa.FaLayerGroup], ["Subtype: how it behaves", ["Luminal A: slower, hormone-driven", "Luminal B: hormone-driven, faster", "HER2-enriched: too much ERBB2", "Basal-like: aggressive"], fa.FaMicroscope], ["Alterations: what changed in DNA", ["Mutation: a small change, such as TP53", "Amplification: extra copies, such as MYC", "Deletion: lost copies, such as RB1", "Drivers cause growth, passengers do not"], fa.FaDna]];
  for (let i = 0; i < 3; i++) {
    const x = 0.6 + i * 4.11;
    card(s, x, 1.55, 3.91, 4.9);
    await icon(s, domain[i][2], x + 0.25, 1.78, 0.62);
    text(s, domain[i][0], x + 0.25, 2.55, 3.45, 0.8, { fontSize: 19, bold: true, color: C.text2 });
    bullets(s, domain[i][1], x + 0.25, 3.4, 3.45, 2.9, 15, 8);
  }
  s.addNotes("A quick piece of domain background. Stage describes spread, through the TNM system and overall AJCC stage. Subtype, from PAM50, describes how the tumour behaves. Alterations describe what changed in the DNA: mutations, amplifications, and deletions. Our graphs are built from these three kinds of information.");

  s = content(1, 0);
  s.addText("Censoring is the central difficulty", { placeholder: "title" });
  await fit(s, FIG("fig_censoring.png"), 0.6, 1.5, 7.6, 4.9, "Illustration of six patient timelines, some ending in progression and some censored");
  card(s, 8.5, 1.55, 4.23, 4.75);
  head(s, "What it means", 8.75, 1.75, 3.75);
  bullets(s, ["145 patients have a recorded progression", "938 are censored: follow-up ended with no event", "Censored does not mean cured", "Calling them negatives breaks a binary model"], 8.75, 2.3, 3.75, 3.8, 16, 11);
  s.addNotes("This idea drives almost every design decision. Our outcome is progression-free survival. 145 patients have a recorded progression, but 938 are censored, which means follow-up ended with nothing recorded. Patient B here left the study at month 30. We cannot say B will never progress. So every method we use later has to handle this properly.");

  s = content(1, 0);
  s.addText("Data: TCGA PanCancer Atlas and PrimeKG", { placeholder: "title" });
  const stats = [["1,083", "patients with a usable progression record"], ["145", "recorded progressions"], ["126,252", "mutation records"], ["500", "genes in our panel, including 27 known drivers"], ["8.1M", "edges in PrimeKG, from 20 public resources"], ["4,570", "PrimeKG edges kept for our gene panel"]];
  stats.forEach(([value, label], i) => stat(s, value, label, 0.6 + (i % 3) * 4.11, 1.55 + Math.floor(i / 3) * 2.2, 3.91, 2.0));
  text(s, "Files were verified against official checksums. Expression, methylation, and protein data were left out to keep the graphs readable.", 0.6, 6.0, 12.13, 0.7, { fontSize: 15, color: C.accent6 });
  s.addNotes("Our patient data is the TCGA breast cancer PanCancer Atlas study from cBioPortal. We use clinical tables, mutations, and copy-number calls. To bound graph size we use a 500-gene panel ranked by how often genes are altered, with 27 known drivers always included. PrimeKG is a public biomedical knowledge graph with about 8.1 million edges. We filter it to 4,570 edges that touch our panel.");

  s = content(1, 0);
  s.addText("Each patient becomes a graph", { placeholder: "title" });
  await fit(s, REPORT("fig_patient_graph.png"), 0.6, 1.45, 7.0, 5.1, "Patient graph with one patient, one sample, and mutated genes");
  head(s, "Patient to Sample to Gene", 8.0, 1.6, 4.7);
  bullets(s, ["Patient node: age, sex, subtype, stage", "Sample node: tumour type, aneuploidy", "Gene node: symbol and Entrez ID", "Edges keep each mutation record and copy-number value", "Contradictory calls are excluded and logged, not guessed"], 8.0, 2.2, 4.73, 4.2, 16, 10);
  s.addNotes("Each patient is a small directed graph. A patient node connects to a tumour sample, which connects to the altered genes. Edges keep the original evidence. When the source data contradicts itself for a gene, we exclude that record and log it instead of guessing. Every graph is validated before use.");

  s = content(1, 0);
  s.addText("PrimeKG adds biological context to every patient", { placeholder: "title" });
  await fit(s, REPORT("fig_fused_graph.png"), 0.6, 1.45, 6.9, 5.1, "Fused graph linking altered genes to proteins, pathways, and a disease");
  table(s, [["PrimeKG relation", "Edges"], ["Gene in Reactome pathway", { text: "2,410", options: { align: "right" } }], ["Protein interacts with protein", { text: "772", options: { align: "right" } }], ["Gene targeted by drug", { text: "858", options: { align: "right" } }], ["Gene linked to breast disease", { text: "530", options: { align: "right" } }]], 7.9, 1.6, 4.83, [3.73, 1.1], 15, 0.45);
  bullets(s, ["Genes join on Entrez ID, with no fuzzy matching", "Median fused graph: 632 nodes", "Two patients with different genes can now meet at the same pathway"], 7.9, 4.1, 4.83, 2.3, 16, 8);
  s.addNotes("We then fuse each patient graph with PrimeKG, joining on Entrez identifiers so there is no fuzzy name matching. Each altered gene brings in its partner proteins, pathways, drugs, and breast-disease links. The benefit is this: two patients with different mutated genes can now meet at the same pathway, which a gene list can never show. That completes the data side. Surya will now take you through the neural models.");

  // ===== Part 2 =====
  pres.addSection({ title: SEC[1] });
  s = pres.addSlide({ masterName: "SECTION", sectionTitle: SEC[1] });
  s.addText("Part 2: Neural models and what we ruled out", { placeholder: "title" });
  s.addText(SPEAKERS[1], { placeholder: "body" });
  s.addNotes("Thank you, Sehar. In this part I will explain how we evaluated everything, what the graph neural networks achieved, what we tried and dropped, and why we changed direction.");

  s = content(2, 1);
  s.addText("Every result is out-of-fold and leakage-checked", { placeholder: "title" });
  await fit(s, FIG("fig_cross_validation.png"), 0.6, 1.45, 7.5, 2.9, "Five-fold cross-validation grid");
  card(s, 0.6, 4.55, 7.5, 1.95);
  head(s, "Main metric: C-index", 0.85, 4.68, 7);
  await eq(s, "cindex", 0.85, 5.15, 7.0, 0.42, "C equals comparable pairs ranked correctly over all comparable pairs");
  text(s, "0.5 is a coin flip and 1.0 is perfect.", 0.85, 5.75, 7, 0.4, { fontSize: 15 });
  head(s, "How we prevent leakage", 8.45, 1.55, 4.3);
  bullets(s, ["A tested guard rejects outcome columns", "Time and event are targets, never features", "Scaling and rule mining use the training fold only", "58 unit tests cover the pipeline"], 8.45, 2.15, 4.28, 4.2, 16, 11);
  s.addNotes("Everything uses stratified five-fold cross-validation with a fixed seed, so each patient is predicted once by models that never saw them. Our main metric is the concordance index. Among pairs where we know who progressed first, it measures how often the model gave that patient the higher risk. It uses time and handles censoring. To stop leakage, a tested guard blocks outcome columns.");

  s = content(2, 1);
  s.addText("The first label failed, so we built a better one", { placeholder: "title" });
  card(s, 0.6, 1.55, 4.1, 4.9, C.background2);
  head(s, "Raw PFS status", 0.85, 1.75, 3.6);
  text(s, "0.54", 0.85, 2.2, 3.6, 0.9, { fontFace: "+mj-lt", fontSize: 44, bold: true, color: C.accent2 });
  text(s, "AUROC, because censored patients were treated as negatives", 0.85, 3.15, 3.6, 0.9);
  head(s, "36-month landmark", 0.85, 4.1, 3.6);
  text(s, "Progressed by month 36, yes or no. Patients censored earlier are dropped: 498 patients, 89 events.", 0.85, 4.55, 3.6, 1.7, { fontSize: 15 });
  barChart(s, ["12 months (916 patients)", "24 months (629)", "36 months (498)", "48 months (406)", "60 months (344)"], [0.572, 0.674, 0.700, 0.719, 0.654], ["9FB6CC", "9FB6CC", THEME.colors.dk2, "9FB6CC", "9FB6CC"], 5.0, 1.5, 7.73, 4.1, { title: "Clinical AUROC by horizon", min: 0.4, max: 0.8, series: "AUROC", alt: "Clinical AUROC at five horizons, highest at 48 months, 36 months chosen" });
  text(s, "36 months balances events against dropped patients. 48 months is within noise and loses 92 more patients.", 5.0, 5.7, 7.73, 0.75, { fontSize: 15 });
  s.addNotes("Our first attempt treated censored patients as negatives, and a GraphSAGE model reached an AUROC of only 0.54, close to chance. So we moved to a 36-month landmark: did the patient progress by month 36? We drop anyone censored earlier, leaving 498 patients and 89 events. A sweep showed short horizons have too few events, and long ones lose too many patients. Later, the survival models use all 1,083 patients.");

  s = content(2, 1);
  s.addText("Seven graph networks, one shared design", { placeholder: "title" });
  await fit(s, FIG("fig_message_passing.png"), 0.6, 1.4, 12.13, 2.45, "Message passing diagram from pathway to patient");
  card(s, 0.6, 4.0, 5.95, 2.6);
  head(s, "Tested", 0.85, 4.12, 5.4);
  bullets(s, ["GCN, GraphSAGE, GATv2, R-GCN, RGAT, GIN, graph transformer", "3 layers, width 64, dropout 0.3", "A clinical shortcut, so training starts from a sensible score"], 0.85, 4.55, 5.5, 2.0, 15, 6);
  card(s, 6.78, 4.0, 5.95, 2.6);
  head(s, "Three training objectives", 7.03, 4.12, 5.4);
  await eq(s, "cox", 7.03, 4.62, 5.45, 0.6, "Cox hazard and partial likelihood loss");
  text(s, "Cross-entropy on the landmark, a smooth C-index loss, and the Cox loss shown here, which uses censored patients correctly.", 7.03, 5.32, 5.45, 1.2, { fontSize: 15 });
  s.addNotes("A graph neural network updates each node from its neighbours. After three layers, pathway context reaches the patient node, and that node produces the risk. We tested seven architectures with the same depth and readout, so a difference in results comes from message passing alone. We also tested three objectives. The Cox loss matters most, since it uses each patient's follow-up time.");

  s = content(2, 1);
  s.addText("No graph model beat the clinical baseline", { placeholder: "title" });
  barChart(s, ["Clinical logistic regression", "RGAT", "GCN", "GraphSAGE", "GATv2", "R-GCN", "Graph transformer", "GIN", "Gene-list MLP"], [0.703, 0.694, 0.688, 0.679, 0.638, 0.627, 0.613, 0.590, 0.570], [THEME.colors.accent6, THEME.colors.accent1, THEME.colors.accent1, THEME.colors.accent1, THEME.colors.accent1, THEME.colors.accent1, THEME.colors.accent1, THEME.colors.accent1, "9FB6CC"], 0.6, 1.4, 7.9, 5.3, { title: "AUROC at 36 months, five-fold mean", min: 0.4, max: 0.78, series: "AUROC", alt: "Clinical logistic regression 0.703 leads every graph model" });
  card(s, 8.8, 1.55, 3.93, 4.95);
  head(s, "Reading this chart", 9.03, 1.72, 3.5);
  bullets(s, ["Fold spread is 0.05 to 0.15, larger than most gaps", "GIN ignores edge types and ranks low, so edge types do matter", "A C-index loss did not improve ranking"], 9.03, 2.3, 3.5, 4.0, 16, 11);
  s.addNotes("This is the benchmark on the 36-month label. The clinical model, using only age, sex, subtype, stage, and M stage, is at the top. RGAT and GCN come close but do not beat it. The fold spread is larger than most gaps, so we do not read much into this ordering. GIN, which ignores edge types, ranks low, so edge types do carry something. Training directly on the C-index did not help either.");

  s = content(2, 1);
  s.addText("Why the final system does not use graph structure learning", { placeholder: "title" });
  await fit(s, FIG("fig_hops.png"), 0.6, 1.4, 6.3, 3.15, "Hops around a patient and the fixed graph boundary");
  text(s, "GSL learns which edges to keep instead of trusting the input graph.", 0.6, 4.58, 6.3, 0.35, { fontSize: 15, color: C.accent6 });
  table(s, [["AUPRC at 36 months", "Fixed", "GSL v1"], ["GCN", { text: "0.396", options: { align: "right" } }, { text: "0.316", options: { align: "right", bold: true } }], ["GraphSAGE", { text: "0.379", options: { align: "right" } }, { text: "0.383", options: { align: "right" } }], ["GATv2", { text: "0.355", options: { align: "right" } }, { text: "0.366", options: { align: "right" } }]], 0.6, 5.0, 6.3, [3.5, 1.4, 1.4], 14, 0.38);
  head(s, "What happened", 7.25, 1.5, 5.4);
  bullets(s, ["Version 1 shut off the whole second hop: none of 5,745 edges passed 0.5", "Version 2 started from the fixed graph, and its gain stayed within fold noise"], 7.25, 2.0, 5.48, 1.8, 16, 8);
  head(s, "Why we left it out", 7.25, 3.85, 5.4, C.accent2);
  bullets(s, ["No reliable gain over the fixed graph", "Every report path should be a real, stored edge", "R-GCN, RGAT, GIN, and the transformer cannot use edge weights"], 7.25, 4.35, 5.48, 2.2, 16, 8);
  s.addNotes("A natural question is why we did not use graph structure learning. We did try it, twice. It learns which edges matter for each patient. The first version let the scorer see the hop number, and it shut off the entire second hop. GCN's AUPRC fell from 0.396 to 0.316. The second version fixed that by starting from the fixed graph and only gating the third hop. It looked slightly better, but stayed within fold noise. We left it out for three reasons: no reliable gain, our explanations need real stored edges, and it only works with three of our seven models. The hop-prior run's outputs were also not kept, so we cannot recompute those numbers.");

  s = content(2, 1);
  s.addText("Other ideas we tested and set aside", { placeholder: "title" });
  table(s, [["Idea", "What happened", "Decision"], ["RGAT on the full cohort", "Over five minutes per epoch on a laptop CPU", "Dropped as impractical"], ["Other hop depths", "AUROC 0.694, 0.708, 0.705 at 1, 2, 3 hops (3-fold)", "Kept 2 hops"], ["SMOTE and Graph SMOTE", "GNNs at 0.50 to 0.53 on GSE2034", "Dropped"], ["External GSE2034 graphs", "Logistic regression 0.722, GraphSAGE 0.582", "Same lesson, different endpoint"], ["Random survival forest", "C-index 0.554, the lowest of all", "Baseline only"], ["Pretrained embeddings", "Not run. Embeddings train from scratch", "Future work"]], 0.6, 1.5, 12.13, [3.5, 5.6, 3.03], 15, 0.58);
  text(s, "The full list, with reasons, is in our team guide and report.", 0.6, 5.95, 12.13, 0.4, { fontSize: 15, color: C.accent6 });
  s.addNotes("Here are other ideas we tested. RGAT was too slow for the full cohort on our hardware. Hop depth barely changed anything, so we kept two hops, where pathways are the mechanistic second step. Oversampling with SMOTE did not help on the external GSE2034 cohort. On that cohort, a linear model again beat the graph models. The random survival forest was the weakest of all, because 145 events is too few for it.");

  s = content(2, 1);
  s.addText("Every model family stops near 0.67", { placeholder: "title" });
  await fit(s, FIG("fig_saturation.png"), 0.6, 1.4, 8.0, 5.3, "C-index of twelve models with spread and the tie bar at 0.659");
  card(s, 8.85, 1.55, 3.88, 4.95, C.text2);
  text(s, "So we changed the goal", 9.08, 1.78, 3.45, 0.8, { fontSize: 22, bold: true, color: C.background1, fontFace: "+mj-lt" });
  text(s, "With 145 events, stage, nodal status, age, and subtype carry most of the signal.", 9.08, 2.7, 3.45, 1.5, { color: C.background1 });
  text(s, "More model power did not add accuracy. Explanation became the goal, as our supervisor advised.", 9.08, 4.25, 3.45, 2.0, { color: C.background1 });
  s.addNotes("This chart shows the main finding. GNNs, logic, forests, and Cox models all finish in the same range, and the spread lines overlap heavily. With only 145 events, clinical variables carry most of the signal. We accepted that result instead of chasing noise. So, following our supervisor's direction, the question changed from how to raise the score to how to explain it. Shaurya will explain how.");

  // ===== Part 3 =====
  pres.addSection({ title: SEC[2] });
  s = pres.addSlide({ masterName: "SECTION", sectionTitle: SEC[2] });
  s.addText("Part 3: Symbolic reasoning and our novelty", { placeholder: "title" });
  s.addText(SPEAKERS[2], { placeholder: "body" });
  s.addNotes("Thank you, Surya. This part covers how we moved from simple rules to learned logic, and it holds the main novelty of our work.");

  s = content(3, 2);
  s.addText("IF-THEN rules were readable, but limited", { placeholder: "title" });
  card(s, 0.6, 1.55, 12.13, 1.3);
  text(s, "Baseline rule from RuleKit", 0.85, 1.68, 11.6, 0.35, { fontSize: 14, bold: true, color: C.accent6 });
  text(s, "IF nodal stage = N3 AND aneuploidy is between 4 and 26.5 THEN higher risk", 0.85, 2.05, 11.6, 0.6, { fontSize: 20, fontFace: "+mj-lt" });
  stat(s, "7.8", "rules kept per fold", 0.6, 3.1, 2.9, 1.75, C.background2);
  stat(s, "10%", "of rules used the knowledge graph", 3.68, 3.1, 2.9, 1.75, C.background2);
  stat(s, "0", "weights: a rule fires or it does not", 6.76, 3.1, 2.9, 1.75, C.background2);
  card(s, 9.84, 3.1, 2.89, 3.45, C.text2);
  text(s, "Our supervisor's direction", 10.04, 3.27, 2.5, 0.8, { bold: true, color: C.background1, fontSize: 18 });
  text(s, "Try Markov logic, soft logic, and reinforcement learning for the rule part.", 10.04, 4.15, 2.5, 2.2, { color: C.background1 });
  text(s, "We had a knowledge graph, but the rules hardly used it. Few rules, no weights, and little reasoning over PrimeKG.", 0.6, 5.1, 9.0, 1.3, { fontSize: 18 });
  s.addNotes("Our first symbolic layer used RuleKit, a survival rule learner. It gives readable rules like this one. But it kept fewer than eight rules per fold, the rules carry no weights, and only 10 percent used PrimeKG. We had built a knowledge graph, and the rules were hardly reasoning over it. Our supervisor suggested three directions: Markov logic, soft logic, and reinforcement learning. We built all three.");

  s = content(3, 2);
  s.addText("All reasoners read the same 422 first-order facts", { placeholder: "title" });
  s.addChart(pres.charts.BAR, [{ name: "Ground atoms", labels: ["PrimeKG pathway", "Genomic", "Clinical", "PrimeKG drug", "PrimeKG disease"], values: [306, 57, 26, 26, 7] }], { x: 0.6, y: 1.45, w: 6.5, h: 4.3, barDir: "bar", chartColors: [THEME.colors.accent3, THEME.colors.accent1, THEME.colors.accent6, THEME.colors.accent3, THEME.colors.accent3], invertedColors: [THEME.colors.accent3, THEME.colors.accent1, THEME.colors.accent6, THEME.colors.accent3, THEME.colors.accent3], barGapWidthPct: 45, showLegend: false, showValue: true, dataLabelPosition: "outEnd", dataLabelFontSize: 13, dataLabelColor: THEME.colors.dk1, dataLabelFontFace: "+mn-lt", dataLabelFormatCode: "0", catAxisLabelFontSize: 13, catAxisLabelColor: THEME.colors.dk1, catAxisLabelFontFace: "+mn-lt", catAxisOrientation: "maxMin", valAxisHidden: true, valAxisMinVal: 0, valAxisMaxVal: 350, valGridLine: { style: "none" }, catGridLine: { style: "none" }, showTitle: true, title: "Ground atoms by kind", titleFontSize: 15, titleColor: THEME.colors.dk2, titleFontFace: "+mn-lt", altText: "306 pathway, 57 genomic, 26 clinical, 26 drug, and 7 disease atoms" });
  card(s, 7.4, 1.55, 5.33, 2.35);
  head(s, "A relational atom counts", 7.65, 1.68, 4.9);
  await eq(s, "atom", 7.65, 2.2, 4.85, 0.42, "n sub w of p counts altered genes in pathway w");
  text(s, "The number of this patient's altered genes that PrimeKG places in pathway w.", 7.65, 2.8, 4.85, 1.0, { fontSize: 15 });
  bullets(s, ["Kept at 3 to 90% prevalence, never chosen by outcome", "Same facts and same folds, so the comparison is fair"], 7.4, 4.15, 5.33, 2.2, 16, 9);
  text(s, "80% of the facts (339) are knowledge-graph relations.", 0.6, 5.85, 6.5, 0.5, { fontSize: 16, bold: true, color: C.text2 });
  s.addNotes("First we built a shared set of 422 facts called ground atoms. Some are clinical, some genomic, and most are relations over PrimeKG, such as this one, which counts how many of the patient's altered genes sit in a pathway. Selection uses prevalence only, never the outcome, so there is no leakage. Every reasoner reads these same facts on the same folds, which keeps the comparison fair.");

  s = content(3, 2);
  s.addText("Reasoner 1: a survival Markov logic network", { placeholder: "title" });
  card(s, 0.6, 1.55, 12.13, 1.5);
  head(s, "Every rule becomes a weighted formula, and many can apply at once", 0.85, 1.68, 11.6);
  await eq(s, "mln_form", 0.85, 2.25, 9.8, 0.55, "Weighted formula and the patient score");
  card(s, 0.6, 3.25, 12.13, 1.65);
  head(s, "Our change: learn the weights from survival times", 0.85, 3.38, 11.6, C.accent2);
  await eq(s, "mln_obj", 0.85, 3.9, 9.2, 0.8, "Elastic-net penalised Cox partial likelihood");
  bullets(s, ["A standard MLN would treat a patient who has not relapsed yet as a non-progressor", "The Cox likelihood uses each patient's follow-up time correctly", "The L1 penalty sets useless formulas to exactly zero, so it also selects the structure"], 0.6, 5.1, 12.13, 1.6, 16, 5);
  s.addNotes("A Markov logic network attaches a weight to each logic formula, so rules become soft and many can act together. With full evidence, the score for a patient reduces to a weighted count of true formulas. Our key change is how weights are learned. A standard MLN would treat a patient who has not relapsed yet as a non-progressor, which is the censoring mistake. We learn the weights with the Cox partial likelihood instead. The elastic-net penalty also removes useless formulas. So the structure is logic, and the learning objective is survival analysis.");

  s = content(3, 2);
  s.addText("Worked example: why TCGA-BH-A1EV scores high", { placeholder: "title" });
  table(s, [["Formula", "Weight", "Count", "Effect on risk"], ["Her2 subtype and altered gene in Neutrophil degranulation", { text: "+0.079", options: { align: "right" } }, { text: "3", options: { align: "right" } }, { text: "+0.42", options: { align: "right" } }], ["Her2 subtype and altered gene in L1 and Ankyrins pathway", { text: "+0.055", options: { align: "right" } }, { text: "2", options: { align: "right" } }, { text: "+0.40", options: { align: "right" } }], ["N0 stage, false for this patient (T3 N1 M0)", { text: "-0.304", options: { align: "right" } }, { text: "0", options: { align: "right" } }, { text: "+0.29", options: { align: "right" } }], ["Altered gene targeted by Ellagic acid", { text: "+0.082", options: { align: "right" } }, { text: "1", options: { align: "right" } }, { text: "+0.15", options: { align: "right" } }], ["GOLPH3L amplified and altered gene in Complement cascade", { text: "+0.018", options: { align: "right" } }, { text: "1", options: { align: "right" } }, { text: "+0.08", options: { align: "right" } }], [{ text: "All remaining formulas", options: {} }, "", "", { text: "+0.09", options: { align: "right" } }], [{ text: "Logic risk", options: { bold: true, fill: C.background2 } }, { text: "", options: { fill: C.background2 } }, { text: "", options: { fill: C.background2 } }, { text: "+1.43", options: { align: "right", bold: true, fill: C.background2 } }]], 0.6, 1.5, 12.13, [7.53, 1.4, 1.2, 2.0], 15, 0.5);
  text(s, "Effects are relative to the average training patient, so lacking a protective fact also raises risk. Recorded progression: 12.0 months. Weights come from the fold that held this patient out.", 0.6, 5.75, 12.13, 0.8, { fontSize: 15, color: C.accent6 });
  s.addNotes("Here is a real example, from the model that held this patient out. Each row is a formula. Two Her2 formulas with pathway relations contribute most. The third row is interesting: N0 is a protective formula, and this patient does not have N0, so losing that protection raises risk relative to the average. Adding the rows gives a logic risk of 1.43. This is what we mean by reasoning: each part of the score is visible.");

  s = content(3, 2);
  s.addText("Reasoner 2: soft logic with collective reasoning", { placeholder: "title" });
  card(s, 0.6, 1.55, 5.95, 2.1);
  head(s, "Rules become penalties", 0.85, 1.68, 5.4);
  await eq(s, "psl_hinge", 0.85, 2.2, 4.6, 0.42, "Hinge penalty");
  text(s, "Truth values lie between 0 and 1. A rule pays a penalty for how far it is from true.", 0.85, 2.75, 5.45, 0.8, { fontSize: 15 });
  card(s, 6.78, 1.55, 5.95, 2.1);
  head(s, "Patients influence each other", 7.03, 1.68, 5.4);
  await eq(s, "psl_collective", 7.03, 2.2, 5.45, 0.42, "Similar patients share risk");
  text(s, "Each patient links to their ten most similar training patients.", 7.03, 2.75, 5.45, 0.8, { fontSize: 15 });
  bullets(s, ["Censoring is handled with Kaplan-Meier soft targets for neighbours", "Weights are trained through 40 unrolled inference steps"], 0.6, 3.9, 7.0, 2.4, 16, 9);
  stat(s, "≈ 0.01", "learned weight of the collective rule", 7.95, 3.9, 2.3, 2.3, C.background2, C.accent2);
  text(s, "A real finding: in this cohort, similar patients add nothing beyond a patient's own facts.", 10.4, 3.95, 2.33, 2.4, { fontSize: 15 });
  s.addNotes("The second reasoner is probabilistic soft logic. Truth values are soft, and a rule pays a penalty for how far it is from being satisfied. Its special power is collective reasoning: similar patients can pass risk to each other. We expected this to help, but the learned weight of the collective rule was about 0.01 in every fold. In this cohort, similar patients add no outcome information beyond a patient's own facts. That is a finding, and we report it as one.");

  s = content(3, 2);
  s.addText("Reasoner 3: a reinforcement-learning rule writer", { placeholder: "title" });
  await fit(s, FIG("fig_rl_loop.png"), 0.6, 1.4, 12.13, 3.55, "RL loop from GRU policy to reward, validation filter, and MLN");
  card(s, 0.6, 5.05, 12.13, 0.98);
  await eq(s, "rl_reward", 0.85, 5.2, 8.0, 0.68, "Stability reward");
  text(s, "A rule earns reward only if it holds on two separate halves of patients.", 9.0, 5.17, 3.55, 0.8, { fontSize: 15, valign: "middle" });
  text(s, "Alone it overfits: 20,000 to 27,000 rules tried per fold, and a C-index of 0.563.", 0.6, 6.2, 12.13, 0.5, { fontSize: 17, bold: true, color: C.accent2 });
  s.addNotes("The third reasoner is a small recurrent network that writes a rule, one fact at a time. It is trained with REINFORCE. We reward stability: a rule must separate fast from slow progressors in the same direction on two separate halves of patients, and we score the weaker half. It must then pass a held-out 30 percent. Even so, alone it overfits, since it tries over 20,000 rules per fold and scores 0.563. It searches well but judges badly. That led to the next idea.");

  s = content(3, 2);
  s.addText("Our engine: RL proposes, Markov logic decides", { placeholder: "title" });
  card(s, 0.6, 1.55, 5.95, 4.9);
  await icon(s, fa.FaRobot, 0.85, 1.8, 0.66, C.accent4);
  text(s, "RL does the structure search", 1.68, 1.85, 4.7, 0.6, { fontSize: 20, bold: true, color: C.text2, valign: "middle" });
  text(s, "It finds multi-step knowledge-graph rules that pairwise mining cannot reach, and sends up to 150 replicated proposals.", 0.85, 2.7, 5.45, 1.3);
  await icon(s, fa.FaBalanceScale, 0.85, 4.15, 0.66, C.text2);
  text(s, "The survival likelihood does the judging", 1.68, 4.05, 4.7, 0.85, { fontSize: 20, bold: true, color: C.text2, valign: "middle" });
  text(s, "Every proposal competes for weight. Bad ones get exactly zero, so RL's weakness stops mattering.", 0.85, 5.0, 5.45, 1.3);
  s.addChart(pres.charts.DOUGHNUT, [{ name: "Formulas", labels: ["Mined clinical and PrimeKG pairs", "Single atoms", "RL-proposed multi-step"], values: [81, 51, 44] }], { x: 6.9, y: 1.45, w: 5.83, h: 4.7, holeSize: 58, chartColors: [THEME.colors.accent1, THEME.colors.accent6, THEME.colors.accent4], showLegend: true, legendPos: "b", legendFontSize: 14, legendColor: THEME.colors.dk1, legendFontFace: "+mn-lt", showPercent: false, showValue: true, dataLabelColor: "FFFFFF", dataLabelFontSize: 14, dataLabelFontFace: "+mn-lt", dataLabelFormatCode: "0", showTitle: true, title: "176 formulas kept over five folds", titleFontSize: 15, titleColor: THEME.colors.dk2, titleFontFace: "+mn-lt", altText: "81 mined, 51 single atom, 44 RL formulas" });
  s.addNotes("So we combined them. RL proposes candidate formulas, including multi-step knowledge-graph rules. The survival MLN then learns all weights. Bad proposals get exactly zero. This pairing is the core of our novelty: RL for structure learning, and the survival likelihood for weight learning. Across the five folds the engine kept 176 formulas, and 44 of them came from RL.");

  s = content(3, 2);
  s.addText("Accuracy is a tie; the reasoning is not", { placeholder: "title" });
  const labels = ["Survival MLN", "RL-structured MLN (engine)", "RuleKit (baseline)", "Survival PSL", "RL rules alone"];
  const engineColors = (values) => values.map((_, i) => (i === 1 ? THEME.colors.dk2 : i === 2 ? THEME.colors.accent6 : "9FB6CC"));
  barChart(s, labels, [0.630, 0.628, 0.619, 0.604, 0.563], engineColors(labels), 0.6, 1.4, 6.0, 3.95, { title: "C-index, five-fold mean", min: 0.4, max: 0.72, series: "C-index", alt: "MLN 0.630, RL-MLN 0.628, RuleKit 0.619, PSL 0.604, RL 0.563" });
  barChart(s, labels, [63, 69, 10, 63, 91], engineColors(labels), 6.73, 1.4, 6.0, 3.95, { title: "Formulas using PrimeKG (%)", min: 0, max: 115, format: "0", series: "Share", alt: "Share of formulas using PrimeKG, engine 69 percent, RuleKit 10 percent" });
  card(s, 0.6, 5.45, 12.13, 1.2);
  text(s, "Tie rule: within one standard error of the best (bar 0.605), MLN, the engine, and RuleKit are tied, and we pick the richest. RL alone uses PrimeKG most, but it is not tied on accuracy.", 0.85, 5.55, 11.65, 1.0, { fontSize: 16, valign: "middle" });
  s.addNotes("Here is the honest comparison. On the left, Markov logic leads, but the gap to RuleKit, about 0.01, is smaller than the fold swing of about 0.05, so we do not claim better accuracy. On the right is the real difference: 69 percent of the engine's formulas use the knowledge graph, against 10 percent for RuleKit. RL alone is higher, but it fails on accuracy. Among the tied models we pick the richest reasoning, which is the RL-structured MLN.");

  s = content(3, 2);
  s.addText("What the learned formulas look like", { placeholder: "title" });
  const examples = [["f_n0", "Clinical, protective", "-0.30", "408 patients", 3.2], ["f_n3", "Mined pair, raises risk", "+0.15", "43 patients", 7.3], ["f_vegfa", "RL-proposed, protective", "-0.14", "93 patients", 7.3]];
  for (let i = 0; i < 3; i++) {
    const y = 1.5 + i * 1.28;
    card(s, 0.6, y, 7.8, 1.13);
    text(s, examples[i][1], 0.8, y + 0.09, 3.9, 0.3, { fontSize: 14, bold: true, color: C.text2 });
    text(s, `weight ${examples[i][2]}, ${examples[i][3]}`, 4.5, y + 0.09, 3.75, 0.3, { fontSize: 14, align: "right" });
    await eq(s, examples[i][0], 0.8, y + 0.5, examples[i][4], 0.38, "Logic formula");
  }
  text(s, "Formulas are weighted per standard deviation, and each one comes with its training support.", 0.6, 5.5, 7.8, 0.8, { fontSize: 15, color: C.accent6 });
  await fit(s, REPORT("fig_rule_km.png"), 8.7, 1.45, 4.03, 3.6, "Kaplan-Meier curve of patients for whom the RL formula holds");
  text(s, "Patients for whom the RL formula holds, against the whole cohort. For illustration, not a separate test.", 8.7, 5.15, 4.03, 1.1, { fontSize: 14, color: C.accent6 });
  s.addNotes("These are real learned formulas. Clinical facts carry the largest weights, as we would expect. No lymph-node spread is protective. The second combines N3 stage with a knowledge-graph relation. The third was proposed by RL and joins two pathways, which RuleKit could not have expressed in this form. On the right is the survival curve of patients for whom it holds. These are associations, not causal claims.");

  s = content(3, 2);
  s.addText("Our novelty", { placeholder: "title" });
  const novelty = [[fa.FaWeightHanging, "Weighted logic", "About 35 weighted first-order formulas per fold, against 8 crisp rules"], [fa.FaHeartbeat, "Survival-aware Markov logic", "Weights from the Cox likelihood, so censoring is handled correctly"], [fa.FaRobot, "RL as structure learner", "A replication reward that guards against lucky rules"], [fa.FaProjectDiagram, "Multi-step PrimeKG reasoning", "69% of formulas use the knowledge graph, against 10%"], [fa.FaRoute, "Rules inside explanations", "Paths scored as 0.6 neural plus 0.4 rule"], [fa.FaClipboardCheck, "An honest comparison", "All three suggested methods tested, negative results included"]];
  for (let i = 0; i < 6; i++) {
    const x = 0.6 + (i % 3) * 4.11, y = 1.55 + Math.floor(i / 3) * 2.5;
    card(s, x, y, 3.91, 2.3);
    await icon(s, novelty[i][0], x + 0.22, y + 0.22, 0.58);
    text(s, novelty[i][1], x + 0.95, y + 0.22, 2.8, 0.7, { fontSize: 17, bold: true, color: C.text2, valign: "middle" });
    text(s, novelty[i][2], x + 0.22, y + 1.02, 3.47, 1.15, { fontSize: 15 });
  }
  s.addNotes("To sum up the novelty. We replaced crisp rules with weighted first-order logic. We made Markov logic survival-aware with the Cox likelihood. We used reinforcement learning as its structure learner, with a reward that guards against luck. The formulas reason over PrimeKG, and they now drive part of every explanation path. And we tested all three methods fairly, including the ones that did not work. Vishesh will now show how this reaches the patient.");

  // ===== Part 4 =====
  pres.addSection({ title: SEC[3] });
  s = pres.addSlide({ masterName: "SECTION", sectionTitle: SEC[3] });
  s.addText("Part 4: From model to patient", { placeholder: "title" });
  s.addText(SPEAKERS[3], { placeholder: "body" });
  s.addNotes("Thank you, Shaurya. I will show how these pieces become one prediction, how we estimate the month of progression, how each path is scored, and what the final report looks like. Then I will run the live demo.");

  s = content(4, 3);
  s.addText("Fusion: one Cox model, three inputs", { placeholder: "title" });
  await fit(s, FIG("fig_fusion.png"), 0.6, 1.4, 12.13, 3.45, "Clinical, GNN, and logic inputs enter one Cox model that drives risk, month, and paths");
  card(s, 0.6, 5.0, 5.95, 1.65);
  head(s, "Tie rule (one standard error)", 0.85, 5.1, 5.4);
  await eq(s, "one_se", 0.85, 5.58, 5.45, 0.9, "Bar equals best minus standard error, 0.659");
  card(s, 6.78, 5.0, 5.95, 1.65);
  head(s, "What fusion taught us", 7.03, 5.1, 5.4);
  text(s, "GNN and logic get hazard ratios of 0.94 to 1.09 (p > 0.28). They add explanation, not accuracy.", 7.03, 5.55, 5.45, 1.0, { fontSize: 15 });
  s.addNotes("The final prediction comes from one Cox model with three inputs: the 19 clinical fields, the GNN risk, and the logic risk. Both risks are out-of-fold, so this is proper stacking. The tie rule gives us a bar of 0.659. In every fold, once the clinical block is present, the GNN and logic terms have hazard ratios close to 1 and are not significant. So they add explanation, not accuracy. We say this openly in the reports too.");

  s = content(4, 3);
  s.addText("The chosen model is tied with clinical Cox and can explain itself", { placeholder: "title" });
  table(s, [["Model", "Harrell's C", "IBS", "Explains itself?"], ["Clinical Cox (19 fields)", "0.675 ± 0.035", "0.134", "No"], [{ text: "Neuro-symbolic Cox (chosen)", options: { bold: true, fill: C.background2 } }, { text: "0.664 ± 0.021", options: { bold: true, fill: C.background2 } }, { text: "0.134", options: { bold: true, fill: C.background2 } }, { text: "Yes: paths and formulas", options: { bold: true, fill: C.background2 } }], ["GNN + logic, no clinical", "0.642 ± 0.056", "0.139", "Partly"], ["RL-structured MLN alone", "0.628 ± 0.055", "--", "Formulas only"], ["Best GNN (GraphSAGE)", "0.622 ± 0.067", "--", "Paths only"], ["Random survival forest", "0.554 ± 0.023", "0.145", "No"]], 0.6, 1.5, 12.13, [5.0, 2.4, 1.2, 3.53], 15, 0.5);
  stat(s, "0.011", "gap to clinical Cox, below one standard error of 0.016", 0.6, 5.2, 5.95, 1.4, C.background2);
  stat(s, "No measurable cost", "in C-index or Brier score for the explanation", 6.78, 5.2, 5.95, 1.4, C.background2);
  s.addNotes("Here are the main results on all 1,083 patients. Clinical Cox reaches 0.675 and our neuro-symbolic Cox reaches 0.664. The gap of 0.011 is smaller than one standard error, 0.016, so they are statistically tied. Their integrated Brier scores match as well. Among tied models we choose the one that can give reasons. We pay nothing measurable for the explanation.");

  s = content(4, 3);
  s.addText("Estimating the month of progression", { placeholder: "title" });
  table(s, [["Estimator (145 held-out progressors)", "Median error", "Within 12 mo", "Window"], ["Constant guess", "16.0", "37%", "19%"], ["Cox conditional median, τ = 60", "18.2", "33%", "18%"], ["Cox conditional median, τ = 120", "24.6", "23%", "39%"], ["Log-normal AFT", "12.9", "48%", "34%"], [{ text: "Log-logistic AFT (chosen)", options: { bold: true, fill: C.background2 } }, { text: "12.9", options: { bold: true, fill: C.background2 } }, { text: "48%", options: { bold: true, fill: C.background2 } }, { text: "33%", options: { bold: true, fill: C.background2 } }]], 0.6, 1.5, 7.7, [3.9, 1.3, 1.3, 1.2], 14, 0.5);
  await eq(s, "aft", 0.6, 4.75, 5.2, 0.4, "AFT model");
  bullets(s, ["Most patients never progress, so we estimate time given progression", "Fitted on training-fold progressors only", "Estimates bunch at 15 to 45 months, so very early and very late cases are hard"], 0.6, 5.25, 7.7, 1.5, 15, 3);
  await fit(s, REPORT("fig_timing.png"), 8.55, 1.45, 4.18, 4.3, "Estimated against recorded month");
  text(s, "Chance window accuracy is about 25%. The estimate is coarse, and reports always print the recorded month beside it.", 8.55, 5.8, 4.18, 0.95, { fontSize: 14, color: C.accent6 });
  s.addNotes("Most patients never progress, so a plain survival median is usually undefined. We estimate the month given that progression happens. A log-logistic AFT model, fitted only on training-fold progressors, was best. It is within 12 months for 48 percent of progressors, against 37 percent for a constant guess. It is coarse, especially for very early or very late cases. So every report prints the recorded month beside the estimate.");

  s = content(4, 3);
  s.addText("Explanation paths combine neural and rule evidence", { placeholder: "title" });
  card(s, 0.6, 1.5, 12.13, 0.95);
  await eq(s, "path", 0.85, 1.62, 8.6, 0.72, "Path score formula");
  text(s, "Weights follow our supervisor's design.", 9.65, 1.62, 2.9, 0.72, { fontSize: 15, valign: "middle" });
  await fit(s, REPORT("fig_paths_example.png"), 0.6, 2.6, 7.7, 4.15, "Top paths for patient TCGA-BH-A1EV with score split");
  bullets(s, ["Neural: how much the GNN relied on the path's edges", "Rule: weights of formulas true for this patient that involve the path", "Prominent: top 20% of the cohort, score at least 0.523", "A report shows min(5, prominent) paths, with at least 3"], 8.55, 2.7, 4.18, 4.0, 15, 9);
  s.addNotes("Each path is a real walk in the patient's graph, from an altered gene to a pathway, drug, or disease. Its score has two parts. Neural says how much the GNN relied on those edges. Rule is the summed weight of the learned formulas that hold for this patient and involve this path. The weights 0.6 and 0.4 follow our supervisor's design. In this example, the top path runs from the amplified gene NFASC to Neurofascin interactions. Paths show what the model relied on, and they are not causal claims.");

  s = content(4, 3);
  s.addText("A report for every patient, grouped by progression time", { placeholder: "title" });
  await fit(s, REPORT("fig_report_p1.png"), 0.6, 1.4, 3.75, 5.4, "Patient report first page");
  await fit(s, REPORT("fig_report_p2.png"), 4.45, 1.4, 3.75, 5.4, "Patient report second page with paths and formulas");
  table(s, [["Group", "Recorded", "Patients"], ["A", "up to 12 months", { text: "32", options: { align: "right" } }], ["B", "12 to 24 months", { text: "30", options: { align: "right" } }], ["C", "24 to 36 months", { text: "27", options: { align: "right" } }], ["D", "over 36 months", { text: "56", options: { align: "right" } }], ["N", "no progression", { text: "938", options: { align: "right" } }]], 8.45, 1.5, 4.28, [0.9, 2.28, 1.1], 14, 0.4);
  bullets(s, ["1,083 PDF reports", "Record, risk, month, curve, paths, formulas, profile, and similar patients", "Drug links are marked as research context"], 8.45, 4.15, 4.28, 2.6, 15, 7);
  s.addNotes("The system writes a two to three page PDF for all 1,083 patients, grouped by when they progressed. The first page gives the clinical record, risk, month estimate, and survival curve. The second gives top paths and the formulas that fire. Patients with no recorded progression are group N, and their reports state that censoring is not proof of remission. Drug links are marked as research context, not treatment advice.");

  s = content(4, 3);
  s.addText("Live demo: the model running in the browser", { placeholder: "title" });
  await fit(s, FIG("fig_demo.png"), 0.6, 1.4, 7.4, 5.35, "Screenshot of the BRCA Progression Reasoner demo");
  head(s, "Two ways in", 8.3, 1.5, 4.4);
  bullets(s, ["TCGA patient: any of the 1,083 real cases, scored by the fold that held them out", "New patient: start from a synthetic example and change a few fields or genes"], 8.3, 2.0, 4.43, 1.95, 15, 7);
  head(s, "Checked against the pipeline", 8.3, 4.0, 4.4);
  bullets(s, ["Cox risk matches for all 1,083 patients", "New patients: the GNN term stays at its average, which moves the median patient 2 percentile points"], 8.3, 4.5, 4.43, 1.7, 15, 7);
  text(s, [{ text: "Open the demo", options: { hyperlink: { url: DEMO_URL, tooltip: "Open the live demo" } } }], 8.3, 6.3, 4.43, 0.4, { fontSize: 16, bold: true, color: C.accent1 });
  s.addNotes("Now the demo. Open the link, or demo/index.html from the repository if the internet is unavailable. Step one: show group A, TCGA-BH-A1EV, at the 98th percentile, and point out that its 23-month estimate misses the recorded window, which shows we do not hide errors. Step two: read one path and one formula. Step three: press Change this patient, set N stage to N0, and watch risk and formulas change live. Step four: open New patient and select the Basal-like example. To the question of how data is entered: real cases are already loaded, and new cases start from a synthetic example, so only a few fields need changing.");

  s = content(4, 3);
  s.addText("Limitations we state openly", { placeholder: "title" });
  const limits = [["Accuracy ceiling", "C-index saturates near 0.67 with only 145 events."], ["Coarse timing", "Within a year for about half of progressors. Late cases are hard."], ["Reasoning, not accuracy", "PSL's collective rule found no signal, and RL alone overfits."], ["Not causal", "Paths show what the model relied on. They are not proof of mechanism."], ["Not treatment advice", "PrimeKG drug links include enzymes and carriers."], ["One cohort", "One seed, and no external validation of the reports yet."]];
  limits.forEach(([title, body], i) => {
    const x = i % 2 ? 6.78 : 0.6, y = 1.55 + Math.floor(i / 2) * 1.68;
    card(s, x, y, 5.95, 1.5);
    text(s, title, x + 0.25, y + 0.15, 5.45, 0.4, { fontSize: 18, bold: true, color: C.text2 });
    text(s, body, x + 0.25, y + 0.62, 5.45, 0.8);
  });
  s.addNotes("We want to be direct about the limits. Accuracy saturates near 0.67. Timing is coarse. The logic adds reasoning, not accuracy. Paths are not causal, and drug links are not treatment advice. We used one cohort and one seed, and the reports have not been validated externally. This is a research prototype.");

  s = content(4, 3);
  s.addText("Conclusion and future work", { placeholder: "title" });
  card(s, 0.6, 1.55, 6.95, 5.0, C.text2);
  text(s, "What we delivered", 0.9, 1.75, 6.35, 0.5, { fontSize: 22, bold: true, color: C.background1, fontFace: "+mj-lt" });
  s.addText([
    "A leakage-checked pipeline from raw TCGA tables to patient graphs fused with PrimeKG",
    "A survival Markov logic engine with RL-proposed, knowledge-graph-grounded formulas",
    "A fused model tied with the best clinical model, at 0.664 against 0.675",
    "1,083 explained reports and a live demo",
  ].map((item, i) => ({ text: item, options: { bullet: { indent: 16 }, breakLine: i < 3, paraSpaceAfter: 14 } })), { x: 0.9, y: 2.45, w: 6.35, h: 3.9, fontSize: 19, color: C.background1, margin: 0, valign: "top", isTextBox: true });
  head(s, "Future work", 7.85, 1.65, 4.85);
  bullets(s, ["Let RL walk PrimeKG relation paths, trained jointly with MLN weights", "Link patients and run full MLN inference", "Add expression and methylation data", "Validate on an external cohort such as METABRIC", "Ask oncologists to rate the reports"], 7.85, 2.25, 4.88, 4.2, 16, 10);
  s.addNotes("To conclude: we built a complete pipeline, and turned a saturated prediction problem into an explained one. On this cohort, knowledge-graph reasoning adds explanation rather than accuracy, and the system is designed around that finding. Next, we would let the RL agent walk PrimeKG relation paths, run full joint inference, add more data types, validate externally, and ask oncologists to assess the reports.");

  s = pres.addSlide({ masterName: "SECTION", sectionTitle: SEC[3] });
  s.addText("Thank you", { placeholder: "title" });
  s.addText([{ text: "We welcome your questions.", options: { breakLine: true } }, { text: "Shaurya Pratap Singh, Surya Pratap Singh, Vishesh Vigh, Sehar Majid Abdi", options: { breakLine: true } }, { text: "Supervisor: Dr. Shashwati Banerjea", options: {} }], { placeholder: "body" });
  s.addNotes("Thank you. We welcome your questions. Any of us can take them, and backup slides are ready.");

  // ===== Backup =====
  pres.addSection({ title: SEC[4] });
  s = content("B", 4);
  s.addText("Backup: all models on the full follow-up", { placeholder: "title" });
  table(s, [["Model", "Harrell's C", "Uno's C", "Model", "Harrell's C", "Uno's C"], ["Clinical Cox (19 fields)", "0.675 ± 0.035", "0.616", "GraphSAGE, C-index", "0.622 ± 0.067", "0.573"], ["Clinical Cox (extended)", "0.667 ± 0.042", "0.657", "RuleKit rules", "0.619 ± 0.048", "0.623"], ["Neuro-symbolic Cox", "0.664 ± 0.021", "0.616", "GCN, Cox loss", "0.612 ± 0.044", "0.549"], ["Neuro-symbolic Cox (ext.)", "0.662 ± 0.042", "0.655", "Survival PSL", "0.604 ± 0.059", "0.571"], ["GNN + logic, no clinical", "0.642 ± 0.056", "0.606", "GraphSAGE, Cox loss", "0.603 ± 0.062", "0.556"], ["Survival MLN", "0.630 ± 0.055", "0.617", "R-GCN, Cox loss", "0.586 ± 0.041", "0.590"], ["RL-structured MLN", "0.628 ± 0.055", "0.615", "RL rules", "0.563 ± 0.046", "0.570"], ["", "", "", "Random survival forest", "0.554 ± 0.023", "0.559"]], 0.6, 1.5, 12.13, [3.0, 1.9, 1.0, 3.3, 1.93, 1.0], 14, 0.5);
  s.addNotes("Backup slide. Use this if the panel asks for exact numbers. All values are five-fold means on the same folds, with 1,083 patients and 145 events.");

  s = content("B", 4);
  s.addText("Backup: 36-month results under two objectives", { placeholder: "title" });
  table(s, [["Model", "AUROC (cross-entropy)", "AUPRC", "AUROC (C-index loss)", "AUPRC"], ["Clinical logistic regression", "0.703", "0.463", "0.703", "0.463"], ["RGAT", "0.694", "0.412", "0.678", "0.389"], ["GCN", "0.688", "0.415", "0.658", "0.379"], ["GraphSAGE", "0.679", "0.406", "0.660", "0.393"], ["GATv2", "0.638", "0.366", "0.640", "0.366"], ["R-GCN", "0.627", "0.352", "0.602", "0.343"], ["Graph transformer", "0.613", "0.354", "0.627", "0.376"], ["GIN", "0.590", "0.320", "0.558", "0.263"], ["MLP", "0.570", "0.274", "0.566", "0.263"]], 0.6, 1.5, 12.13, [4.13, 2.5, 1.5, 2.5, 1.5], 14, 0.46);
  text(s, "498 patients, 89 events. A random ranking has an AUPRC of 0.179.", 0.6, 6.3, 12.13, 0.4, { fontSize: 14, color: C.accent6 });
  s.addNotes("Backup slide with the full 36-month benchmark. Note that the fold standard deviations are 0.05 to 0.15.");

  await pres.writeFile({ fileName: OUTPUT });
  await applyTheme(OUTPUT, THEME);
  console.log("wrote", OUTPUT);
}

main().catch((error) => { console.error(error); process.exit(1); });
