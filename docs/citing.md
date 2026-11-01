# Citing

Please cite the accompanying methods paper (under review) and the software:

> Author names withheld. copernicus: rubric-based construct coding with LLM methods. Version 0.1.0. Software accompanying a methods paper under review.

The entry for the paper is a placeholder until the review ends: "The accompanying methods paper", by the withheld authors, noted as under review. The citation metadata of the software is in `CITATION.cff`, in the Citation File Format that reference managers and code hosting sites read. This page includes the file, so the two cannot differ. The source is at <https://github.com/keeping-up-with-the-humans/org-copernicus> and the documentation at <https://keeping-up-with-the-humans.github.io/org-copernicus/>.

```yaml
--8<-- "CITATION.cff"
```

## Notes on the version

The version in the citation is the version of the package that produced your results. `copernicus.__version__` gives it, and the run store records it in the column `library_version` of every score. Cite that version, and keep the provenance files that the library writes beside your results.
