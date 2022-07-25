window.MathJax = {
  loader: {
    load: [
      '[tex]/boldsymbol',
      '[tex]/mathtools'
    ]
  },

  tex: {
    inlineMath: [["\\(", "\\)"]],
    displayMath: [["\\[", "\\]"]],
    processEscapes: true,
    processEnvironments: true,
    packages: {
      '[+]': ['boldsymbol'],
      '[+]': ['mathtools']
    }
  },

  options: {
    ignoreHtmlClass: ".*|",
    processHtmlClass: "arithmatex"
  }
};

document$.subscribe(MathJax.typesetPromise)

const ifDocumentContains = (id, callback) => {
  const element = document.getElementById(id)
  if (element) {
    callback(element)
  }
}