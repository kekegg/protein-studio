// Section 01 - 项目定位 & 现状
// Stats grid (mixed media): 4 big stat cards + "why us" bullet column

const { addPageNumber, addFooter, addSlideTitle, addSectionMarker } = require('./_common.js');

function createSlide(pres, theme) {
  const slide = pres.addSlide();
  slide.background = { color: theme.bg };

  addSlideTitle(slide, theme, '01  -  我们在解决什么',
    'AI-native 分子可视化 + 计算工作台 - 给 drug discovery 团队的 agent-controllable 桌面工具链');
  addSectionMarker(slide, theme, '01', 'POSITIONING');

  // Top row: 3 big stat cards
  const cards = [
    { v: '7 x 7', l: 'rep x scheme', sub: '表达矩阵全维度覆盖' },
    { v: '15,868', l: 'src LOC', sub: '15 个核心模块' },
    { v: '4', l: '计算引擎', sub: 'OpenMM / Vina / EquiScore / pdbfixer' }
  ];
  cards.forEach((c, i) => {
    const cx = 0.4 + i * 3.10;
    slide.addShape(pres.shapes.RECTANGLE, {
      x: cx, y: 1.55, w: 2.85, h: 1.55,
      fill: { color: theme.light },
      line: { type: 'none' },
      rectRadius: 0.08
    });
    // Top accent bar
    slide.addShape(pres.shapes.RECTANGLE, {
      x: cx, y: 1.55, w: 2.85, h: 0.10,
      fill: { color: theme.accent },
      line: { type: 'none' },
      rectRadius: 0.05
    });
    slide.addText(c.v, {
      x: cx + 0.20, y: 1.70, w: 2.55, h: 0.55,
      fontFace: 'Georgia',
      fontSize: 32,
      bold: true,
      color: theme.accent,
      align: 'left',
      valign: 'middle',
      margin: 0
    });
    slide.addText(c.l, {
      x: cx + 0.20, y: 2.30, w: 2.55, h: 0.30,
      fontFace: 'Microsoft YaHei',
      fontSize: 14,
      bold: true,
      color: theme.primary,
      align: 'left',
      valign: 'middle',
      margin: 0
    });
    slide.addText(c.sub, {
      x: cx + 0.20, y: 2.60, w: 2.55, h: 0.40,
      fontFace: 'Calibri',
      fontSize: 11,
      color: theme.secondary,
      align: 'left',
      valign: 'top',
      margin: 0
    });
  });

  // Bottom-left: 痛点 column
  slide.addText('现有工具的痛点', {
    x: 0.4, y: 3.35, w: 4.5, h: 0.35,
    fontFace: 'Microsoft YaHei',
    fontSize: 14,
    bold: true,
    color: theme.primary,
    margin: 0
  });
  const pains = [
    'PyMOL:  无法被 AI agent 脚本化驱动,CLI 缺失关键动作',
    '3Dmol.js:  仅浏览器,接 compute 引擎要在两端桥接',
    'Chimera / VMD:  UI 老化,agent 集成需 patch',
    'PyMol + scripts:  工程成本高,跨平台分发差'
  ];
  pains.forEach((p, i) => {
    slide.addText(
      [
        { text: p.split(':')[0] + ':  ', options: { bold: true, color: theme.primary, fontFace: 'Consolas' } },
        { text: p.split(':').slice(1).join(':').trim(), options: { color: theme.secondary, fontFace: 'Microsoft YaHei' } }
      ],
      {
        x: 0.4, y: 3.70 + i * 0.32, w: 5.0, h: 0.30,
        fontSize: 11,
        align: 'left',
        valign: 'middle',
        margin: 0
      }
    );
  });

  // Bottom-right: 我们的定位 column
  slide.addText('我们的定位', {
    x: 5.6, y: 3.35, w: 4.0, h: 0.35,
    fontFace: 'Microsoft YaHei',
    fontSize: 14,
    bold: true,
    color: theme.primary,
    margin: 0
  });
  const pos = [
    '桌面级可视化 (PySide6)  +  计算引擎 (OpenMM) 同进程',
    'MCP server 显式暴露所有 GUI / 计算动作供 agent 调用',
    '可视化精度对标 PyMOL, 计算精度对接 AmberTools 等基线'
  ];
  pos.forEach((p, i) => {
    slide.addText(
      [
        { text: 'v  ', options: { bold: true, color: theme.accent, fontFace: 'Consolas' } },
        { text: p, options: { color: theme.primary, fontFace: 'Microsoft YaHei' } }
      ],
      {
        x: 5.6, y: 3.70 + i * 0.42, w: 4.0, h: 0.40,
        fontSize: 11,
        align: 'left',
        valign: 'top',
        margin: 0
      }
    );
  });

  addFooter(slide, theme);
  addPageNumber(slide, theme, 3, 10);
}

module.exports = { createSlide };
