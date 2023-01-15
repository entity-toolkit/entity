document.addEventListener("DOMContentLoaded", () => {
  renderMathInElement(document.body, {
    delimiters: [
      { left: '$$', right: '$$', display: true },
      { left: '$', right: '$', display: false },
      { left: '\\(', right: '\\)', display: false },
      { left: '\\[', right: '\\]', display: true }
    ],
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
