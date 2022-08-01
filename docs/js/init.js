window.MathJax = {
  loader: {
    load: [
      '[tex]/boldsymbol',
      '[tex]/mathtools',
      '[tex]/ams',
      '[tex]/mathtools',
    ]
  },

  tex: {
    inlineMath: [["\\(", "\\)"]],
    displayMath: [["\\[", "\\]"]],
    processEscapes: true,
    processEnvironments: true,
    packages: {
      '[+]': ['boldsymbol'],
      '[+]': ['mathtools'],
      '[+]': ['ams'],
      '[+]': ['mathtools'],
    }
  },

  options: {
    ignoreHtmlClass: ".*|",
    processHtmlClass: "arithmatex"
  }
};

document$.subscribe(MathJax.typesetPromise)