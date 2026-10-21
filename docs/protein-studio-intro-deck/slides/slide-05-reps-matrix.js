// Section 03 - 7 reps x 7 schemes 表达矩阵
// Content - mixed media: matrix grid on left, SS-aware detail card on right

const { addPageNumber, addFooter, addSlideTitle, addSectionMarker } = require('./_common.js');

function createSlide(pres, theme) {
  const slide = pres.addSlide();
  slide.background = { color: theme.bg };

  addSlideTitle(slide, theme, '03  -  表达矩阵: 7 reps x 7 schemes',
    '任意表达 x 任意着色方案 = 49 种可瞬间切换的展示组合,代表性模块 representations.py (1938 L)');
  addSectionMarker(slide, theme, '03', 'RENDERING');

  // Matrix: 7 reps (rows) x 7 schemes (cols)
  // rep names + scheme names
  const reps = ['cartoon', 'ribbon', 'ball_stick', 'line', 'surface', 'esp_surface', 'atoms'];
  const schemes = ['by_chain', 'by_ss', 'pLDDT', 'element', 'hydro.', 'bfactor', 'uniform'];

  const matrixX = 0.40;
  const matrixY = 1.50;
  const cellW = 0.65;
  const cellH = 0.40;
  const labelW = 0.95;
  const labelH = 0.40;

  // Header row: scheme names
  schemes.forEach((s, j) => {
    const x = matrixX + labelW + j * cellW;
    slide.addText(s, {
      x: x, y: matrixY, w: cellW, h: labelH,
      fontFace: 'Consolas',
      fontSize: 9,
      bold: true,
      color: theme.primary,
      align: 'center',
      valign: 'middle',
      margin: 0
    });
  });
  // "rep" label for first column
  slide.addText('rep / scheme', {
    x: matrixX, y: matrixY, w: labelW, h: labelH,
    fontFace: 'Consolas',
    fontSize: 9,
    bold: true,
    color: theme.secondary,
    align: 'left',
    valign: 'middle',
    margin: 0
  });

  // Rep rows
  reps.forEach((r, i) => {
    const y = matrixY + labelH + i * cellH;
    // Rep name in left col
    slide.addText(r, {
      x: matrixX, y: y, w: labelW, h: cellH,
      fontFace: 'Consolas',
      fontSize: 10,
      bold: true,
      color: theme.primary,
      align: 'left',
      valign: 'middle',
      margin: 0
    });
    // 7 cells per row
    for (let j = 0; j < 7; j++) {
      const cx = matrixX + labelW + j * cellW;
      // alternate cell colors lightly
      const isSupported = true;
      const fillColor = isSupported ? theme.light : 'FFFFFF';
      slide.addShape(pres.shapes.RECTANGLE, {
        x: cx, y: y, w: cellW - 0.04, h: cellH - 0.04,
        fill: { color: fillColor },
        line: { color: theme.light, width: 0.5 },
        rectRadius: 0.02
      });
      // Subtle indicator dot to mark supported combos (visually balanced)
      // Don't crowd - leave just the cell filled
    }
  });

  // Legend below matrix
  slide.addText('所有 49 组合均可用;部分 (esp x uniform) 视觉上退化为另一种组合', {
    x: matrixX, y: matrixY + labelH + 7 * cellH + 0.10, w: 7.0, h: 0.30,
    fontFace: 'Microsoft YaHei',
    fontSize: 10,
    color: theme.secondary,
    align: 'left',
    margin: 0
  });

  // Right: SS-aware ribbon highlight card
  const cx = 7.50, cy = 1.50, cw = 2.10, ch = 3.30;
  slide.addShape(pres.shapes.RECTANGLE, {
    x: cx, y: cy, w: cw, h: ch,
    fill: { color: theme.primary },
    line: { type: 'none' },
    rectRadius: 0.10
  });
  slide.addText('HIGHLIGHT', {
    x: cx + 0.20, y: cy + 0.20, w: cw - 0.40, h: 0.30,
    fontFace: 'Consolas',
    fontSize: 10,
    bold: true,
    color: theme.light,
    charSpacing: 4,
    margin: 0
  });
  slide.addText('SS-aware ribbon', {
    x: cx + 0.20, y: cy + 0.50, w: cw - 0.40, h: 0.40,
    fontFace: 'Georgia',
    fontSize: 18,
    bold: true,
    color: 'FFFFFF',
    margin: 0
  });
  slide.addText('自研,不靠 PyMOL', {
    x: cx + 0.20, y: cy + 0.92, w: cw - 0.40, h: 0.28,
    fontFace: 'Microsoft YaHei',
    fontSize: 11,
    color: theme.light,
    margin: 0
  });

  // Feature bullets
  const feats = [
    'alpha-helix: 圆柱',
    'beta-sheet: 箭头扁带',
    'loop: 平滑 tube',
    '二级结构自动推断',
    'PyMOL 同级视觉质量'
  ];
  feats.forEach((f, i) => {
    slide.addText(
      [
        { text: '- ', options: { color: theme.accent, fontFace: 'Consolas', bold: true } },
        { text: f, options: { color: 'FFFFFF', fontFace: 'Microsoft YaHei' } }
      ],
      {
        x: cx + 0.20, y: cy + 1.30 + i * 0.36, w: cw - 0.30, h: 0.32,
        fontSize: 11,
        align: 'left',
        valign: 'middle',
        margin: 0
      }
    );
  });

  addFooter(slide, theme);
  addPageNumber(slide, theme, 5, 10);
}

module.exports = { createSlide };
