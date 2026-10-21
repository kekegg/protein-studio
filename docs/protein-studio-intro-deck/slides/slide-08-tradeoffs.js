// Section 06 - 工程取舍
// Content - comparison cards: 3 alternatives side by side

const { addPageNumber, addFooter, addSlideTitle, addSectionMarker } = require('./_common.js');

function createSlide(pres, theme) {
  const slide = pres.addSlide();
  slide.background = { color: theme.bg };

  addSlideTitle(slide, theme, '为什么自研,而不是 fork PyMOL / wrap 3Dmol',
    '三条路径对比 - 我们选的不是代码最短的那条,是 agent-controllable + 计算闭环最长那条');
  addSectionMarker(slide, theme, '06', 'TRADEOFFS');

  const options = [
    {
      name: 'Fork PyMOL',
      sub: '复刻 C/Python 混合源',
      pros: ['可视化质量齐平', '学界/业界熟悉'],
      cons: ['C 桥 Py 维护难', '几乎无法 agent 化', '打包跨平台痛苦', '无 compute 闭环'],
      verdict: 'REJECTED',
      verdictColor: theme.secondary
    },
    {
      name: '3Dmol.js wrap',
      sub: 'Web viewer + Python backend',
      pros: ['轻量,启动快', '前端栈天然可控'],
      cons: ['IPC 需维护', 'compute 要建第二服务', 'GPU 资源调度分裂', 'snapshot/recording 跨层'],
      verdict: 'PARTIAL',
      verdictColor: '5A6478'
    },
    {
      name: '自研 (PySide6)',
      sub: '我们的选择',
      pros: ['UI + compute 同一进程', 'MCP 原生可接入', 'SS-aware ribbon 自主', '单 zip 分发'],
      cons: ['前期投入大', '需自己解决打包', '需自己追 Visual 化学正确性'],
      verdict: 'CHOSEN',
      verdictColor: theme.accent
    }
  ];

  const cardY = 1.45;
  const cardW = 3.10;
  const cardH = 3.55;
  const cardStartX = 0.40;
  const cardGap = 0.10;

  options.forEach((opt, i) => {
    const x = cardStartX + i * (cardW + cardGap);
    const isChosen = (i === 2);

    // Outer card
    slide.addShape(pres.shapes.RECTANGLE, {
      x: x, y: cardY, w: cardW, h: cardH,
      fill: { color: isChosen ? theme.accent : theme.light },
      line: isChosen ? { color: theme.accent, width: 0 } : { color: 'D0D5DD', width: 0.5 },
      rectRadius: 0.08
    });

    // Name
    slide.addText(opt.name, {
      x: x + 0.20, y: cardY + 0.18, w: cardW - 0.40, h: 0.40,
      fontFace: 'Georgia',
      fontSize: 20,
      bold: true,
      color: isChosen ? 'FFFFFF' : theme.primary,
      align: 'left',
      valign: 'middle',
      margin: 0
    });
    // Sub
    slide.addText(opt.sub, {
      x: x + 0.20, y: cardY + 0.58, w: cardW - 0.40, h: 0.26,
      fontFace: 'Microsoft YaHei',
      fontSize: 11,
      color: isChosen ? 'C5D2E8' : theme.secondary,
      align: 'left',
      valign: 'top',
      margin: 0
    });

    // Pros label
    const prosLabelY = cardY + 0.98;
    slide.addText('+ PROS', {
      x: x + 0.20, y: prosLabelY, w: cardW - 0.40, h: 0.26,
      fontFace: 'Consolas',
      fontSize: 9,
      bold: true,
      color: isChosen ? 'C5D2E8' : '5A7E48',
      charSpacing: 3,
      margin: 0
    });
    opt.pros.forEach((p, j) => {
      slide.addText(
        [
          { text: '+ ', options: { color: isChosen ? 'C5D2E8' : '5A7E48', fontFace: 'Consolas', bold: true } },
          { text: p, options: { color: isChosen ? 'FFFFFF' : theme.primary, fontFace: 'Microsoft YaHei' } }
        ],
        {
          x: x + 0.20, y: prosLabelY + 0.30 + j * 0.28, w: cardW - 0.40, h: 0.26,
          fontSize: 10,
          align: 'left',
          valign: 'middle',
          margin: 0
        }
      );
    });

    // Cons label
    const consLabelY = prosLabelY + 0.30 + opt.pros.length * 0.28 + 0.10;
    slide.addText('- CONS', {
      x: x + 0.20, y: consLabelY, w: cardW - 0.40, h: 0.26,
      fontFace: 'Consolas',
      fontSize: 9,
      bold: true,
      color: isChosen ? 'A8BAD3' : 'B04949',
      charSpacing: 3,
      margin: 0
    });
    opt.cons.forEach((c, j) => {
      slide.addText(
        [
          { text: '- ', options: { color: isChosen ? 'A8BAD3' : 'B04949', fontFace: 'Consolas', bold: true } },
          { text: c, options: { color: isChosen ? 'C5D2E8' : theme.secondary, fontFace: 'Microsoft YaHei' } }
        ],
        {
          x: x + 0.20, y: consLabelY + 0.30 + j * 0.28, w: cardW - 0.40, h: 0.26,
          fontSize: 10,
          align: 'left',
          valign: 'middle',
          margin: 0
        }
      );
    });

    // Verdict bottom
    slide.addText(opt.verdict, {
      x: x + 0.20, y: cardY + cardH - 0.42, w: cardW - 0.40, h: 0.30,
      fontFace: 'Consolas',
      fontSize: 11,
      bold: true,
      color: isChosen ? 'FFFFFF' : opt.verdictColor,
      align: 'center',
      valign: 'middle',
      charSpacing: 4,
      margin: 0
    });
  });

  addFooter(slide, theme);
  addPageNumber(slide, theme, 8, 10);
}

module.exports = { createSlide };
