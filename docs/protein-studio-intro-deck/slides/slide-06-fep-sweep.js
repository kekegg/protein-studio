// Section 04 - FEP sweep 与 NaN 三层防护
// Content - timeline/process: horizontal flow with 5 steps + 3 retry layers callout

const { addPageNumber, addFooter, addSlideTitle, addSectionMarker } = require('./_common.js');

function createSlide(pres, theme) {
  const slide = pres.addSlide();
  slide.background = { color: theme.bg };

  addSlideTitle(slide, theme, '04  -  FEP 计算引擎 & NaN 三层防护',
    'OpenMM solvation + multi-lambda sweep,核心 fep.py (3214 L),三层防护规避生产期 NaN');
  addSectionMarker(slide, theme, '04', 'FEP ENGINE');

  // Horizontal flow: 5 steps
  const steps = [
    { name: 'PDB prep',      file: 'scripts/prep_pdb_for_fep.py',  detail: 'pdbfixer + cleanup' },
    { name: 'Solvate',       file: 'build_amber_solvated_system',   detail: 'TIP3P  +  padding retry' },
    { name: 'Minimise',      file: '_protein_minimise',            detail: 'LBFGS  +  drop loop' },
    { name: 'Pre-equil',     file: 'fep_reference.py',             detail: 'NVT  +  NPT' },
    { name: 'Lambda sweep',  file: 'fep.py sweep',                 detail: 'multi-lambda MD' }
  ];

  const flowY = 1.65;
  const flowH = 1.50;
  const boxW = 1.65;
  const boxGap = 0.20;
  const flowStartX = 0.40;

  steps.forEach((s, i) => {
    const x = flowStartX + i * (boxW + boxGap);
    // Box
    slide.addShape(pres.shapes.RECTANGLE, {
      x: x, y: flowY, w: boxW, h: flowH,
      fill: { color: theme.light },
      line: { color: theme.accent, width: 1.5 },
      rectRadius: 0.06
    });
    // Step number badge (top-left corner)
    slide.addShape(pres.shapes.OVAL, {
      x: x + 0.08, y: flowY + 0.08, w: 0.40, h: 0.40,
      fill: { color: theme.accent },
      line: { type: 'none' }
    });
    slide.addText(String(i + 1), {
      x: x + 0.08, y: flowY + 0.08, w: 0.40, h: 0.40,
      fontFace: 'Georgia',
      fontSize: 14,
      bold: true,
      color: 'FFFFFF',
      align: 'center',
      valign: 'middle',
      margin: 0
    });
    // Step name
    slide.addText(s.name, {
      x: x + 0.55, y: flowY + 0.10, w: boxW - 0.60, h: 0.40,
      fontFace: 'Microsoft YaHei',
      fontSize: 14,
      bold: true,
      color: theme.primary,
      align: 'left',
      valign: 'middle',
      margin: 0
    });
    // File
    slide.addText(s.file, {
      x: x + 0.18, y: flowY + 0.62, w: boxW - 0.30, h: 0.28,
      fontFace: 'Consolas',
      fontSize: 9,
      color: theme.accent,
      align: 'left',
      valign: 'middle',
      margin: 0
    });
    // Detail
    slide.addText(s.detail, {
      x: x + 0.18, y: flowY + 0.92, w: boxW - 0.30, h: 0.50,
      fontFace: 'Calibri',
      fontSize: 10,
      color: theme.secondary,
      align: 'left',
      valign: 'top',
      margin: 0
    });

    // Arrow between boxes
    if (i < steps.length - 1) {
      const ax = x + boxW;
      const ay = flowY + flowH / 2 - 0.07;
      slide.addShape(pres.shapes.RIGHT_TRIANGLE, {
        x: ax + 0.03, y: ay, w: 0.14, h: 0.16,
        fill: { color: theme.secondary },
        line: { type: 'none' },
        rotate: 30
      });
    }
  });

  // NaN retry layers section (bottom)
  slide.addText('NaN 三层防护 (added this round)', {
    x: 0.40, y: 3.45, w: 9.20, h: 0.40,
    fontFace: 'Microsoft YaHei',
    fontSize: 16,
    bold: true,
    color: theme.primary,
    margin: 0
  });

  const layers = [
    { n: 'L1', t: 'Padding retry', d: '0.8 -> 1.1 -> 1.4 nm,逐级放大 water box' },
    { n: 'L2', t: 'Drop loop', d: '去掉空间冲突的随机水分子,再 minimize' },
    { n: 'L3', t: 'Protein minimise', d: '先固定蛋白质仅松弛水分子,再放开全体系' }
  ];
  layers.forEach((L, i) => {
    const lx = 0.40 + i * 3.10;
    const ly = 3.95;
    const lw = 2.85;
    const lh = 1.10;
    // Outer card
    slide.addShape(pres.shapes.RECTANGLE, {
      x: lx, y: ly, w: lw, h: lh,
      fill: { color: 'FFFFFF' },
      line: { color: theme.accent, width: 1 },
      rectRadius: 0.06
    });
    // Left accent block with L# badge
    slide.addShape(pres.shapes.RECTANGLE, {
      x: lx, y: ly, w: 0.55, h: lh,
      fill: { color: theme.accent },
      line: { type: 'none' },
      rectRadius: 0.06
    });
    slide.addText(L.n, {
      x: lx, y: ly, w: 0.55, h: lh,
      fontFace: 'Georgia',
      fontSize: 22,
      bold: true,
      color: 'FFFFFF',
      align: 'center',
      valign: 'middle',
      margin: 0
    });
    // Title
    slide.addText(L.t, {
      x: lx + 0.70, y: ly + 0.15, w: lw - 0.80, h: 0.32,
      fontFace: 'Microsoft YaHei',
      fontSize: 13,
      bold: true,
      color: theme.primary,
      align: 'left',
      valign: 'middle',
      margin: 0
    });
    // Desc
    slide.addText(L.d, {
      x: lx + 0.70, y: ly + 0.50, w: lw - 0.80, h: 0.55,
      fontFace: 'Microsoft YaHei',
      fontSize: 10,
      color: theme.secondary,
      align: 'left',
      valign: 'top',
      margin: 0
    });
  });

  addFooter(slide, theme);
  addPageNumber(slide, theme, 6, 10);
}

module.exports = { createSlide };
