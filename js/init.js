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
    [document.getElementById('contributors'), document.getElementById('core-developers')].forEach(el => {
      if (el) {
        let ul = el.nextElementSibling;
        if (ul) {
          ul.children.forEach(li => {
            const tags_str = />:(.*)\}/.exec(li.innerHTML)[1];
            const tags = tags_str.split(',').map(c => c.trim());
            li.innerHTML = li.innerHTML.replace(tags_str, tags.map(t => `<span class="tag ${t.toLowerCase().replace(' ', '_')}">${t}</span>`).join(''));
          });
        }
      }
    })
  });

  // const cards = document.getElementsByClassName('nt-card');
  // const collapsables = document.getElementsByClassName('abstract');
  // cards.forEach((card, i) => {
  //   card.addEventListener('click', () => {
  //     cards.forEach(c => c.classList.remove('active'));
  //     card.classList.add('active');
  //     collapsables.forEach(c => c.removeAttribute('open'));
  //     collapsables[i].setAttribute('open', true);
  //   });
  // });
});
