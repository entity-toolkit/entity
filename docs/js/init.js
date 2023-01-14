<<<<<<< HEAD
document.addEventListener("DOMContentLoaded", function () {
  renderMathInElement(document.body, {
    // customised options
    // • auto-render specific keys, e.g.:
=======
document.addEventListener("DOMContentLoaded", () => {
  renderMathInElement(document.body, {
>>>>>>> refs/remotes/origin/wiki
    delimiters: [
      { left: '$$', right: '$$', display: true },
      { left: '$', right: '$', display: false },
      { left: '\\(', right: '\\)', display: false },
      { left: '\\[', right: '\\]', display: true }
    ],
<<<<<<< HEAD
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
=======
    throwOnError: false
  });

  window.addEventListener('load', () => {
    // index.md
    let _ = document.getElementById('contributors');
    if (_) {
      let ul = _.nextElementSibling;
      if (ul) {
        ul.children.forEach(li => {
          const tags_str = />:(.*)\}/.exec(li.innerHTML)[1];
          const tags = tags_str.split(',').map(c => c.trim());
          li.innerHTML = li.innerHTML.replace(tags_str, tags.map(t => `<span class="tag ${t.toLowerCase().replace(' ', '_')}">${t}</span>`).join(''));
        });
      }
    }
  });
});
>>>>>>> refs/remotes/origin/wiki
