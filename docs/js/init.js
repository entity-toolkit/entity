document.addEventListener("DOMContentLoaded", function () {
  renderMathInElement(document.body, {
    // customised options
    // • auto-render specific keys, e.g.:
    delimiters: [
      { left: '$$', right: '$$', display: true },
      { left: '$', right: '$', display: false },
      { left: '\\(', right: '\\)', display: false },
      { left: '\\[', right: '\\]', display: true }
    ],
    // • rendering keys, e.g.:
    throwOnError: false
  });
});

/* ----------------------------- MathJax support ---------------------------- */
// - https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js
// window.MathJax = {
//   loader: {
//     load: [
//       '[tex]/boldsymbol',
//       '[tex]/mathtools',
//       '[tex]/ams',
//     ]
//   },

//   tex: {
//     inlineMath: [["\\(", "\\)"]],
//     displayMath: [["\\[", "\\]"]],
//     processEscapes: true,
//     processEnvironments: true,
//     packages: {
//       '[+]': ['boldsymbol'],
//       '[+]': ['mathtools'],
//       '[+]': ['ams'],
//     }
//   },

//   options: {
//     ignoreHtmlClass: ".*|",
//     processHtmlClass: "arithmatex"
//   }
// };

// document$.subscribe(MathJax.typesetPromise)