# API reference

The reference is generated from the docstrings of the package. There is one page for each public subpackage. A page lists the names that the subpackage exports in its `__all__`, with the signature, the arguments, the return value and the errors of each.

You can import most names from the top-level package, as the tutorials do: `import copernicus as cop`, then `cop.code_texts`. The table lists the top-level names, grouped by the subpackage that defines them.

| Subpackage | Top-level names | Page |
|------------|-----------------|------|
| `copernicus.rubric` | `load_rubric`, `Rubric` | [rubric](rubric.md) |
| `copernicus.data` | `TextTable` | [data](data.md) |
| `copernicus.llm` | `OpenAICompatibleBackend`, `ModelCapabilities`, `VLLMBackend`, `FakeBackend`, `ScriptedResponder`, `cue_responder` | [llm](llm.md) |
| `copernicus.coders` | `TwoStageConfig`, `OneStageConfig`, `CodingSettings`, `CodingSummary`, `code_texts`, `export_scores`, `export_construct_scores`, `export_text_scores`, `export_evidence`, `export_provenance`, `merge_stores` | [coders](coders.md) |
| `copernicus.embeddings` | `Encoder`, `EncoderSpec`, `ENCODERS`, `EmbeddingSet` | [embeddings](embeddings.md) |
| `copernicus.references` | `GeneratorSettings`, `generate_references`, `ReferenceSet` | [references](references.md) |
| `copernicus.similarity` | `ReferenceEmbeddings`, `embed_references`, `SimilaritySettings`, `score_similarity`, `run_grid`, `PAPER_GRID` | [similarity](similarity.md) |
| `copernicus.hybrid` | `score_hybrid` | [hybrid](hybrid.md) |
| `copernicus.validity` | `validity_panel`, `PanelSettings`, `ValidityPanel`, `association`, `score_errors`, `cronbach_alpha`, `reliability_ceiling`, `cluster_bootstrap_ci` | [validity](validity.md) |
| `copernicus.baseline` | `BaselineSettings`, `run_ridge_baseline`, `curves` | [baseline](baseline.md) |
| `copernicus.corrections` | `CorrectionData`, `draw_human_sample`, `plan_sample_size`, `get_correction`, `list_corrections`, `cross_fit_predictions`, `pseudo_values`, `mean_ppi`, `group_difference_ppi`, `ols_outcome_ppi`, `ols_predictor_ppi`, `moment_subtraction`, `simex`, `oriv`, `repeated_study`, `run_battery`, `summarize_battery`, `BatterySettings` | [corrections](corrections.md) |
| `copernicus.errors` | `errors` (the module) | [errors](errors.md) |

`copernicus.__version__` holds the version of the package. The helper `copernicus.assets` gives the toy corpus (`toy_corpus_path`) and the cue words of the toy rubric (`load_toy_cues`). `copernicus.rubric.toy_rubric_path` gives the toy rubric.

## The command line

Every step is also a command: `copernicus rubric`, `code`, `store`, `references`, `embed`, `similarity`, `hybrid`, `validity`, `baseline` and `corrections`. Run `copernicus --help` for the list and `copernicus <group> --help` for the commands of a group. The tutorials show the commands that match their Python calls.
