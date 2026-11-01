# Examples

## `end_to_end.py`

One script that runs every step of the library on the toy corpus that ships with the package: 54 short feedback texts written by 18 people, with a hand-assigned gold value. It is the same path as [Getting started](../docs/getting-started.md), in a script that you can run, read and change. `end_to_end.py` holds `run`, which puts the steps in order, the command line (`main`), the list of CSV files (`CSV_FILES`) and the settings of the battery (`BATTERY_*`). `end_to_end_steps.py` beside it holds the function of each step (`build_reference_set`, `code_both_ways`, `validate`, `baseline_records`, `correct_with_a_sample` and the small helpers) with the constants they read: the gold column, the number of runs, the sample size and its seed (`SAMPLE_UNITS`, `DRAW_SEED`), the baseline settings and the hand-written reference texts. Keep the two files in one folder.

The script does the following, in this order:

1. Loads the toy rubric and the toy corpus.
2. Codes the texts with the two-stage LLM coder (two runs) and the one-stage LLM coder (one run), and exports the scores.
3. Embeds the texts with MiniLM, builds a reference set of hand-written feedback sentences (two for each pole of the construct and of each attribute) and embeds it.
4. Scores the texts with the similarity coder and the hybrid coder.
5. Builds the validity panel of the two-stage construct score, the one-stage score, the similarity construct score and the hybrid score against the gold value, and prints the report of each.
6. Fits the NLP-ML baseline at a training size of 20 texts and at the full size.
7. Draws a human sample of 18 texts (whole groups), attaches the gold values as the human values, and estimates the mean and the long-minus-short difference with the prediction-powered estimators. The draw seed (39, `DRAW_SEED` in `end_to_end_steps.py`) gives one illustrative draw: another seed can leave too few labelled training rows in a fold or fewer than two labelled groups on one side of the difference, and the library refuses such a draw loudly instead of returning a number.
8. Cross-fits a linear calibration of the score, and compares three corrections (linear calibration, ridge stacking over the two coders' scores, and their ensemble) in a small battery.

It writes the run stores, the embeddings, nine CSV files, `validity_report.txt` and `summary.json` to the folder you name.

### Run it

Install the library with the `embed` extra, then run the script from the package's root folder:

```bash
pip install ".[embed]"
python examples/end_to_end.py OUT_DIR
```

`OUT_DIR` is created when it does not exist. The first run downloads MiniLM once (about 90 MB); the run itself takes about ten seconds on a laptop CPU. The script needs no GPU and no API key: the default backend is the fake backend, which answers from a table of cue words. Its scores are not those of a real model; they let you run every step and read every output.

### Use a real model

The script codes with the backend that `run` receives, and builds the fake backend only when it gets none. To code with a real model, replace the one line in `run` that builds the fake backend (`backend = cop.FakeBackend(...)`) with:

```python
backend = cop.OpenAICompatibleBackend(model="gpt-4.1-mini")  # reads OPENAI_API_KEY
```

or build the backend yourself and pass it in with `run(Path("OUT_DIR"), backend=backend)`. The rest of the pipeline is unchanged. A real model makes metered calls to a provider that charges for them: estimate the number of calls and the cost first (see the [FAQ](../docs/faq.md#how-much-will-a-run-cost)). The toy corpus is small, so even a real model's numbers say little about your own texts.

## `smoke_two_stage.py`

A budgeted check that the two-stage LLM coder works against a real OpenAI-compatible model. The script estimates the tokens and the cost of the run before it makes a call, adds the cost of each run to a running total in `OUT_DIR/spend.json` (the backend kind, numbers and timestamps only), and refuses to start when the estimate plus the running total reaches the ceiling you give it with `--ceiling`. The prices (`--price-input`, `--price-output`, in US dollars per million tokens) and the ceiling have no defaults, because prices change. It reads the key from the `OPENAI_API_KEY` environment variable and never prints or writes it. It then codes the first `--texts` texts of the toy corpus, writes the run store and the exports, reports the share of attribute calls that did not fail in the extraction stage, the share of returned passages that were verified, the count of each status and of scores equal to 0, and builds the validity panel against the gold value. Run it first with `--dry-run`, which prints the estimate and stops, then with `--texts 12 --runs 1` as a small trial; `--fake` replaces the model with the fake backend, so the script can be tried without a key, a network or a cost: its estimate counts characters / 4 instead of loading tiktoken's encoding, and its ledger entry records the backend `fake` with a cost of 0. A call that raises returns no token counts, so it is not metered, and the ledger can fall short of the provider's bill by such calls.

The script imports `smoke_support.py`, which holds the estimate, the ledger, the metered backend and the acceptance metrics, so keep the two files in one folder.

```bash
python examples/smoke_two_stage.py --out OUT_DIR --price-input 0.40 --price-output 1.60 --ceiling 5 --dry-run
```
