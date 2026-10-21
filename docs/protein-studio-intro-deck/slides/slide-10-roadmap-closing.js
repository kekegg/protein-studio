// Section 08 - 路线图 & Q&A 收尾
// Closing - split recap: 路线图 next steps + Q&A thank-you

const { addPageNumber, addFooter, addSlideTitle, addSectionMarker } = require('./_common.js');

function createSlide(pres, theme) {
  const slide = pres.addSlide();
  slide.background = { color: theme.bg };

  addSlideTitle(slide, theme, '路线图 & 讨论',
    '下个迭代的 4 项工作 + 一个明确的 ask,留给评审讨论');
  addSectionMarker(slide, theme, '08', 'ROADMAP');

  // Left: 4-step roadmap timeline
  const rmX = 0.40, rmY = 1.45, rmW = 6.10;
  slide.addText('ROADMAP  -  接下来 6 周', {
    x: rmX, y: rmY, w: rmW, h: 0.30,
    fontFace: 'Consolas',
    fontSize: 11,
    bold: true,
    color: theme.accent,
    charSpacing: 4,
    margin: 0
  });

  const phases = [
    { w: 'W1-2', t: '收口 FEP NaN', d: '本地 prep_pdb_for_fep.py 串入 GUI,默认参数上线; 收口 last flaky visual test' },
    { w: 'W3-4', t: 'MCP tools 扩展', d: '新增 trajectory/snapshot/diff 工具; 引 MCP sampling 启发推理动作' },
    { w: 'W5',   t: '打包 PoC',       d: 'PyInstaller / conda-pack 试配; 体积目标 < 250MB' },
    { w: 'W6',   t: 'Agent loop 联调', d: '与 Lingsu Agent 端到端跑通: load -> analyse -> modify -> visual -> analyse' }
  ];

  const lineX1 = rmX + 0.15;
  const lineX2 = rmX + rmW - 0.15;
  const lineY = rmY + 0.85;
  slide.addShape(pres.shapes.LINE, {
    x: lineX1, y: lineY, w: lineX2 - lineX1, h: 0,
    line: { color: theme.accent, width: 2 }
  });

  phases.forEach((p, i) => {
    const px = lineX1 + (lineX2 - lineX1) * (i / (phases.length - 1));
    // Milestone dot
    slide.addShape(pres.shapes.OVAL, {
      x: px - 0.12, y: lineY - 0.12, w: 0.24, h: 0.24,
      fill: { color: theme.accent },
      line: { color: 'FFFFFF', width: 2 }
    });
    // Week label above
    slide.addText(p.w, {
      x: px - 0.5, y: lineY - 0.55, w: 1.0, h: 0.30,
      fontFace: 'Consolas',
      fontSize: 10,
      bold: true,
      color: theme.accent,
      align: 'center',
      valign: 'middle',
      margin: 0
    });
    // Title + detail below
    slide.addText(p.t, {
      x: px - 0.85, y: lineY + 0.20, w: 1.7, h: 0.30,
      fontFace: 'Microsoft YaHei',
      fontSize: 13,
      bold: true,
      color: theme.primary,
      align: 'center',
      valign: 'middle',
      margin: 0
    });
    slide.addText(p.d, {
      x: px - 1.0, y: lineY + 0.50, w: 2.0, h: 0.95,
      fontFace: 'Calibri',
      fontSize: 10,
      color: theme.secondary,
      align: 'center',
      valign: 'top',
      margin: 0
    });
  });

  // Right: Q&A / ask panel
  const rx = 6.70, ry = 1.45, rw = 2.90, rh = 3.60;
  slide.addShape(pres.shapes.RECTANGLE, {
    x: rx, y: ry, w: rw, h: rh,
    fill: { color: theme.accent },
    line: { type: 'none' },
    rectRadius: 0.10
  });

  slide.addText('FOR REVIEW', {
    x: rx + 0.25, y: ry + 0.25, w: rw - 0.50, h: 0.30,
    fontFace: 'Consolas',
    fontSize: 10,
    bold: true,
    color: 'C5D2E8',
    charSpacing: 4,
    margin: 0
  });

  slide.addText('Ask', {
    x: rx + 0.25, y: ry + 0.55, w: rw - 0.50, h: 0.40,
    fontFace: 'Georgia',
    fontSize: 22,
    bold: true,
    color: 'FFFFFF',
    margin: 0
  });

  slide.addShape(pres.shapes.RECTANGLE, {
    x: rx + 0.25, y: ry + 1.05, w: 0.60, h: 0.05,
    fill: { color: 'C5D2E8' },
    line: { type: 'none' }
  });

  const asks = [
    'MCP tools schema 需要 review 吗?',
    'PyInstaller PoC 还是 conda-pack?',
    'compute 任务是否要沙箱化?',
    '与 Lingsu 集成的 contract 在哪层定义?'
  ];
  asks.forEach((a, i) => {
    slide.addText(
      [
        { text: '?  ', options: { color: 'C5D2E8', fontFace: 'Consolas', bold: true } },
        { text: a, options: { color: 'FFFFFF', fontFace: 'Microsoft YaHei' } }
      ],
      {
        x: rx + 0.25, y: ry + 1.25 + i * 0.50, w: rw - 0.50, h: 0.45,
        fontSize: 11,
        align: 'left',
        valign: 'top',
        margin: 0
      }
    );
  });

  // Bottom contact / thank-you strip
  slide.addText(
    [
      { text: 'Thanks  ', options: { fontFace: 'Georgia', fontSize: 18, bold: true, color: theme.accent } },
      { text: '  -  Q&A welcome', options: { fontFace: 'Calibri', fontSize: 13, color: theme.secondary } }
    ],
    {
      x: 0.40, y: 4.80, w: 6.10, h: 0.30,
      align: 'left',
      valign: 'middle',
      margin: 0
    }
  );

  addFooter(slide, theme);
  addPageNumber(slide, theme, 10, 10);
}

module.exports = { createSlide };
