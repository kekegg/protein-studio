// Protein-Studio internal tech review deck
// Compile: node compile.js
// Output: ./output/protein-studio-tech-review.pptx

const pptxgen = require('pptxgenjs');
const pres = new pptxgen();

pres.layout = 'LAYOUT_16x9';
pres.author = 'DrXie';
pres.title = 'Protein-Studio Internal Tech Review';
pres.subject = 'Protein-Studio project introduction and engineering decisions';

const theme = {
  primary:   '1A1F2E',   // deep navy text
  secondary: '5A6478',   // muted gray-blue
  accent:    '1B4FA0',   // signature navy blue
  light:     'E8EEF7',   // pale blue panel
  bg:        'FFFFFF'    // clean white
};

const slides = [
  '01-cover',
  '02-toc',
  '03-positioning',
  '04-architecture',
  '05-reps-matrix',
  '06-fep-sweep',
  '07-mcp-control',
  '08-tradeoffs',
  '09-tests-status',
  '10-roadmap-closing'
];

slides.forEach((name) => {
  require(`./slide-${name}.js`).createSlide(pres, theme);
});

pres.writeFile({ fileName: './output/protein-studio-tech-review.pptx' })
  .then((fileName) => console.log('Wrote', fileName));
