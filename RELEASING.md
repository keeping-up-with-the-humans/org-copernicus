# Releasing copernicus

This checklist takes a version from the working tree to a published release. The steps that publish (the upload to a package index, the push to the public repository) are done by hand by a maintainer; nothing here runs them. The documentation site is the exception: it is published by the `docs` workflow on every push to `main` of the public repository.

## 1. Before you start

Work from a clean checkout with the development environment:

```bash
python -m venv .venv
.venv/bin/pip install -e ".[dev,docs]"
```

Run the full suite once. It includes the tests marked `embed`, which need the `embed` extra (`pip install -e ".[embed]"`) and the MiniLM weights cached or downloadable (`COPERNICUS_TEST_DOWNLOAD=1`):

```bash
.venv/bin/python -m pytest -q
```

The continuous-integration workflow in `.github/workflows/ci.yml` runs a smaller selection. The `test` job runs `pytest -q -m "not embed and not slow"` on Python 3.10 and 3.12, together with `ruff` and `mypy src`; the 3.12 leg also installs the `docs` extra and runs `mkdocs build --strict`, and the documentation build test skips itself on 3.10. The `slow` job runs `pytest -q -m "slow and not embed"` on Python 3.12. Neither job installs `torch`, so the tests marked `embed` never run there, and the documentation pages whose blocks are tagged `run embed` are skipped with a reason that names the page. Run the full suite locally before every release.

The workflow in `.github/workflows/docs.yml` builds the site with `mkdocs build --strict` on Python 3.12 with the `docs` extra and deploys it through GitHub Pages on every push to `main`; it runs no test.

## 2. Bump the version

The version appears in four places, and the tests tie them together. Change all four to the same number:

| File | What to change |
|------|----------------|
| `src/copernicus/_version.py` | `__version__`; `pyproject.toml` reads its version from this file |
| `CITATION.cff` | the `version` field |
| `CHANGELOG.md` | a new `## <version>` heading above the previous one, with the changes of the release |
| `README.md` and `docs/citing.md` | the `Version <version>` in the citation line |

`tests/test_package.py`, `tests/test_docs_assets.py` and `tests/test_docs_build.py` pin the number `0.1.0` in a few assertions and read the first heading of the changelog; update them in the same commit as the bump. The docs build test reads the third line of `CHANGELOG.md` as the heading of the current release, so the newest release always comes first.

## 3. Check the documentation build

Building the documentation needs Python 3.11 or later; the library itself runs on 3.10. The continuous-integration workflow builds it on 3.12 only.

```bash
NO_MKDOCS_2_WARNING=1 .venv/bin/mkdocs build --strict -d /tmp/site
```

The build must finish without a warning. This build is a check: the site that is published is built by the `docs` workflow on the runner, from the pushed export (section 4), after the placeholders are replaced.

## 4. Anonymity export

The accompanying methods paper is under review, so the public repository must hold neither author names nor history that carries them. The public repository is a fresh repository created from `git archive` of `library/`. It is never a clone of the research repository, whose history holds names, and never a `git subtree` or a filtered copy of that history. The accounts that host the repository, run its continuous integration and publish to the package index must not carry a person's name either.

1. Check the name on the package index first, so that a name problem never forces a rebuild and a second export commit. The name on the package index is `org-copernicus`, and the import name stays `copernicus`. Before the first upload, check that `org-copernicus` is still free on the index, or already owned by the project's publishing account; the simple index at `https://pypi.org/simple/org-copernicus/` answers 404 while the name is free. The name `copernicus` itself belongs to a different project and is never used. If `org-copernicus` is taken, choose another name and change every place that names it: run `grep -rnE "org-copernicus|org_copernicus" library` in the research tree and edit each hit, namely `name` in `pyproject.toml`, the install lines of the README and `docs/installation.md`, the glob names in the tests, the hint strings in `src/copernicus/embeddings/encoder.py` and `src/copernicus/llm/vllm_backend.py`, the docstring of `src/copernicus/embeddings/__init__.py`, the dist-info names in section 5 and the CHANGELOG; then run this section from its start.

2. Before the export, run the anonymity guard from the root of the research repository, on a clean checkout of the commit to export, with the library's development environment and the site that section 3 built:

   ```bash
   COPERNICUS_GUARD_SITE=/tmp/site HF_HUB_OFFLINE=1 library/.venv/bin/python -m pytest -q -p no:asyncio library_guard
   ```

   The guard sits beside `library/`, outside the exported tree. It checks every file of `library/`, the archive of `HEAD:library`, and the built site: the folder named by `COPERNICUS_GUARD_SITE`, and `library/site` as well when it exists. The export must contain no `library_guard` and no file named in its lists. Then count the archive's paths that mention the guard:

   ```bash
   git archive HEAD:library | tar -t | grep -c guard
   ```

   The count must be 0.

3. From the root of the research repository, export the tree without history:

   ```bash
   git archive --format=tar --prefix=copernicus/ HEAD:library | tar -x -C /tmp
   ```

4. In `/tmp/copernicus`, review the files and replace the placeholders by the real values:

   - `pyproject.toml`: add `[project.urls]` with `Documentation` and `Source`, the URLs of the documentation site and of the repository. The research tree carries no URL table, because the package index refuses a project URL that is not a URL; the `authors` entry changes once the authors may be named;
   - `CITATION.cff`: the `authors` entries, and add `repository-code` and `url`;
   - `README.md`: the author line of the citation block (and the same line in `docs/citing.md`), and a link to the repository and to the documentation site;
   - `mkdocs.yml`: add `site_url` and `repo_url`.

5. Update the tests that pin the placeholders: `test_pyproject_carries_no_project_url_before_the_export`, `test_citation_file_is_valid_and_withholds_the_authors` and `test_readme_cites_the_paper_as_under_review` in `tests/test_docs_assets.py`. Run the suite in the export; `tests/test_no_banned_tokens.py` checks every file for e-mail addresses, user paths, secrets, scheduler flags and planning labels, there as here.

6. Make the first commit under a neutral identity and in UTC, so that neither a name, an e-mail address nor a time zone enters the history:

   ```bash
   cd /tmp/copernicus
   git init
   git add -A
   NOW="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
   TZ=UTC GIT_AUTHOR_DATE="$NOW" GIT_COMMITTER_DATE="$NOW" \
     git -c user.name="Author names withheld" -c user.email="withheld@example.invalid" \
     commit -m "copernicus <version>"
   git log --format="%an <%ae> %ad | %cn <%ce> %cd" --date=iso-strict
   ```

   The last command must show `Author names withheld <withheld@example.invalid>` twice and times that end in `Z` or `+00:00`. Make every later commit of the export the same way.

The workflow in `.github/workflows/ci.yml` runs only once `library/` is the root of a repository, which is the case in the export and not in the research repository, where GitHub does not read workflows below the root. Push the export, open the Actions tab, and confirm that both jobs of `ci` and the `docs` run are green before tagging.

## 5. Build and inspect the distributions

Build `dist/` in the export, after the placeholders are replaced, never in the research tree: the research tree's `pyproject.toml` carries no URL, and its folder can hold files that the export does not.

```bash
cd /tmp/copernicus
rm -rf dist
python -m build
```

This writes the wheel and the source distribution to `dist/`, with the development environment of the export (section 1). Check the wheel before anything is uploaded:

```bash
unzip -l dist/org_copernicus-*.whl
```

The wheel must contain:

- `copernicus/py.typed`, the marker that tells type checkers the package is typed;
- every file below `copernicus/assets/` (the toy corpus and cue words under `examples/`, the rubric under `rubrics/`), the schema `copernicus/rubric/schema.json`, and the three prompt templates under `copernicus/coders/prompts/`;
- `org_copernicus-<version>.dist-info/entry_points.txt` with the line `copernicus = copernicus.cli.app:app`;
- the `LICENSE` file in `org_copernicus-<version>.dist-info/licenses/`, and `License-Expression: MIT` plus `License-File: LICENSE` in `METADATA`.

It must not contain `tests/` or `docs/`. `tests/test_docs_assets.py::test_built_wheel_holds_the_marker_the_assets_and_the_console_script` builds a wheel and checks the same list; a release that passes it has the assets and the marker.

List the source distribution too. It holds the tracked files only, because `[tool.hatch.build.targets.sdist]` in `pyproject.toml` includes the package, the tests, the docs, the examples, the workflow and the top-level files and nothing else, so tables or notes that a run leaves in the folder never ship; `tests/test_docs_assets.py::test_a_stray_file_in_the_root_folder_stays_out_of_the_source_distribution` checks it. Confirm that `LICENSE`, `README.md` and `CHANGELOG.md` are in it and that no other file is:

```bash
tar -tzf dist/org_copernicus-*.tar.gz
```

Install the wheel in a fresh environment and check that the package imports and the console script starts. This does not run the quick start of the README, which `tests/test_docs_assets.py` runs from the working tree:

```bash
python -m venv /tmp/release-check
/tmp/release-check/bin/pip install dist/org_copernicus-*.whl
/tmp/release-check/bin/python -c "import copernicus; print(copernicus.__version__)"
/tmp/release-check/bin/copernicus --help
```

Check the build of the site in the export as well, with `NO_MKDOCS_2_WARNING=1 mkdocs build --strict -d /tmp/site-export` once `mkdocs.yml` carries `site_url` and `repo_url`. The runner builds the site that is published, from the same files, so this build is a check and deploys nothing.

## 6. Tag and publish

Publishing is not part of this checklist; a maintainer does it after the steps above.

1. Tag the commit of the export: `git tag v<version>` and push the tag. A tag made this way carries no identity; an annotated tag would record the tagger.
2. Upload `dist/*` to the package index under the project's publishing account, never a personal one, with a token that is stored outside the repository (never in a workflow file or in the working tree). The name was checked in section 4, before the export.
3. The documentation site needs no step of its own: the push to `main` deploys the site through the `docs` workflow. The Pages source "GitHub Actions" was set once through the repository's API, and nothing is repeated per release.
4. Add a heading for the next version to `CHANGELOG.md` only when its first change lands.
