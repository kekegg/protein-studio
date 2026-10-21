// Cover - Asymmetric left/right layout
// Left: title block. Right: visual motif (7x7 matrix grid).
// No accent line under title. Pure block design.

function createSlide(pres, theme) {
  const slide = pres.addSlide();
  slide.background = { color: theme.bg };

  // Left vertical navy strip - full height accent
  slide.addShape(pres.shapes.RECTANGLE, {
    x: 0, y: 0, w: 0.35, h: 5.625,
    fill: { color: theme.accent },
    line: { type: 'none' }
  });

  // Tiny eyebrow label
  slide.addText('TECH REVIEW  /  2026.10', {
    x: 0.7, y: 0.7, w: 6.0, h: 0.30,
    fontFace: 'Consolas',
    fontSize: 12,
    color: theme.accent,
    bold: true,
    charSpacing: 4,
    margin: 0
  });

  // Big project name
  slide.addText('Protein-Studio', {
    x: 0.7, y: 1.10, w: 8.6, h: 1.10,
    fontFace: 'Georgia',
    fontSize: 60,
    bold: true,
    color: theme.primary,
    align: 'left',
    valign: 'middle',
    margin: 0
  });

  // Chinese subtitle - what the project does in one line
  slide.addText('AI-native 分子可视化 + 计算工作台', {
    x: 0.7, y: 2.35, w: 8.6, h: 0.55,
    fontFace: 'Microsoft YaHei',
    fontSize: 26,
    color: theme.accent,
    align: 'left',
    valign: 'middle',
    margin: 0
  });

  // Brief positioning line
  slide.addText('PySide6  +  PyVista  +  OpenMM  +  MCP   |   7 reps x 7 schemes  +  FEP  +  Docking', {
    x: 0.7, y: 3.05, w: 8.6, h: 0.40,
    fontFace: 'Consolas',
    fontSize: 13,
    color: theme.secondary,
    align: 'left',
    valign: 'middle',
    margin: 0
  });

  // Meta line: presenter, audience, date
  slide.addText(
    [
      { text: 'DrXie', options: { bold: true, color: theme.primary } },
      { text: '    |    ', options: { color: theme.secondary } },
      { text: 'AI-native drug discovery tooling', options: { color: theme.secondary } },
      { text: '    |    ', options: { color: theme.secondary } },
      { text: 'Internal Engineering Review', options: { color: theme.accent, bold: true } }
    ],
    {
      x: 0.7, y: 4.55, w: 8.6, h: 0.35,
      fontFace: 'Calibri',
      fontSize: 13,
      align: 'left',
      valign: 'middle',
      margin: 0
    }
  );

  // Bottom info bar with three blocks (presentation meta)
  const barY = 5.05;
  const barH = 0.40;
  const items = [
    { label: 'AUDIENCE', value: 'Engineering Review' },
    { label: 'DATE', value: '2026-10-11' },
    { label: 'FORMAT', value: 'Internal Presentation' }
  ];
  items.forEach((it, i) => {
    const bx = 0.7 + i * 2.7;
    slide.addText(
      [
        { text: it.label + '   ', options: { color: theme.secondary, fontSize: 9, charSpacing: 3 } },
        { text: it.value, options: { color: theme.primary, fontSize: 12, bold: true } }
      ],
      {
        x: bx, y: barY, w: 2.5, h: barH,
        fontFace: 'Calibri',
        align: 'left',
        valign: 'middle',
        margin: 0
      }
    );
  });
}

module.exports = { createSlide };
