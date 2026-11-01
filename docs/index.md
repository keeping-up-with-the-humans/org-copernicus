# copernicus

copernicus codes constructs from text against a rubric. A rubric names a construct, splits it into attributes, and gives each attribute markers that a reader can look for in a text. The library scores your texts against the rubric, checks the scores against human gold values, and corrects the bias that remains when you use the scores in a mean, a group difference or a regression.

It reproduces the configurations of the accompanying methods paper (under review) and ships the prompts of that paper.

## What you can do with it

| Step | What the library offers | Tutorial |
|------|-------------------------|----------|
| Write the rubric | A model and a JSON schema for constructs, attributes, markers and five described levels | [The rubric](tutorials/rubric.md) |
| Code the texts | The two-stage LLM coder, the one-stage LLM coder, the similarity coder and the hybrid coder | [The two-stage coder](tutorials/two-stage-coder.md), [the similarity coder](tutorials/similarity-coder.md), [the hybrid coder](tutorials/hybrid-coder.md) |
| Set a reference point | The NLP-ML baseline, a ridge regression on embeddings | [The NLP-ML baseline](tutorials/nlp-ml-baseline.md) |
| Check the scores | The validity panel: rank validity, four score errors, reliability ceilings, bootstrap intervals | [The validity panel](tutorials/validity-panel.md) |
| Correct the scores | A human sample of whole groups, score corrections, and estimates of means, differences and coefficients with intervals | [The human sample](tutorials/human-sample.md), [correcting scores](tutorials/correcting-scores.md), [correcting estimates](tutorials/correcting-estimates.md), [sample size](tutorials/sample-size.md) |

```text
texts + rubric
  |-- two-stage / one-stage LLM coder ------------> run store --> scores, evidence
  |-- encoder + reference texts ------------------> similarity coder --> scores
  |-- verified passages + reference texts --------> hybrid coder --> scores
  v
scores --> validity panel (against gold values) <-- NLP-ML baseline
  v
human sample of whole groups --> corrections --> estimates with intervals
```

## Nothing here needs a GPU or an API key

Every tutorial runs on a laptop CPU. The LLM coders run against a fake backend that answers from a table of cue words, so no model is called and nothing is paid for. The tutorials that need embeddings use the small encoder MiniLM, which runs on a CPU. Each tutorial that codes texts shows, as a snippet that you do not run, the one line that switches to a real model through `OpenAICompatibleBackend`.

The scores that the fake backend produces are not scores of a real model. They let you run every step and read every output. They tell you nothing about how well a real model codes your texts.

## Where to start

1. [Install the library](installation.md).
2. Run [Getting started](getting-started.md). It takes the toy corpus through the whole pipeline. Reading it takes about ten minutes; the code runs in about ten seconds on a laptop CPU once MiniLM is cached.
3. Read the tutorial of the step you need. Each one lists what you need, runs on the toy corpus, explains the output and names the usual pitfalls.
4. Use the [decision guide](decision-guide.md) to choose between the coders and the corrections, and the [methods pages](methods/index.md) for the reasoning behind them.
5. Look up a function in the [API reference](api/index.md). The [FAQ](faq.md) answers the questions that come up on costs, resuming, reproducibility and sample size.

## How to cite

Please cite the accompanying methods paper (under review) and the software. See [Citing](citing.md).
