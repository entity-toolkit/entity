---
hide:
  - footer
---

`entity` documentation is automatically generated using the `mkdocs` framework and the [`Material for mkdocs`](https://squidfunk.github.io/mkdocs-material/) theme. When you commit/push to the `wiki` branch the static website is automatically compiled and pushed to the `gh-pages` branch of the main repository.

!!! hint

    Documentations are created using `markdown` syntax which is then automatically parsed and converted into `html`. As such, any `html`/`css`/`js` code you write in the documentation will be automatically rendered in the documentation. To add global external `css` or `js` files, add them to the `mkdocs.yml` file (`extra_javascript` and `extra_css`). If a script is meant to run on just one page, add it with a dedicated `<script></script>` tag.

## Workflow

1. Pull the `wiki` branch of the main repository (it is recommended to do this in a separate directory from the main code).
  ```shell
  git clone -b wiki git@github.com:haykh/entity.git entity-wiki
  cd entity-wiki
  ```

2. Create an isolated python virtual environment and activate it.
  ```shell
  python -m venv .venv
  source .venv/bin/activate
  ```

1. Install the dependencies (everything is installed locally in the `.venv` directory).
  ```shell
  pip install -r requirements.txt
  ```

1. Start the reactive server that will generate the website and will dynamically update any changes made to the documentation.
  ```shell
  mkdocs serve
  ```
  To access the documentation simply open the [`http://127.0.0.1:8000/`](http://127.0.0.1:8000/) in your browser.

1. When satisfied with all the changes made simply push them to the `wiki` branch.
  ```shell
  git add .
  git commit -m "<reasonable comment>"
  git push origin wiki
  ```
  Shortly after that `github-actions` will generate the website and push it to the `gh-pages` branch of the main repository, which will be accessible from the web.