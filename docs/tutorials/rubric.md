# The rubric

Every coder in the library reads the same rubric file. A rubric names the construct you want to code, splits it into attributes, and gives each attribute markers: features of a text that a reader can point to. Each marker has five described levels. This tutorial shows how a rubric is built, how the library checks it, and how to write your own.

## What you need

- The base install: `pip install .` in the package's root folder (see [Installation](../installation.md)). No encoder, no GPU, no API key.
- Nothing else. The toy rubric ships with the package. No model is called anywhere in this tutorial.

## The structure of a rubric

| Part | Fields | Role |
|------|--------|------|
| Rubric | `rubric_id`, `concept`, `concept_definition`, `literature_sources`, `attributes`, `definitions` | The construct, its definition, and a glossary of terms the prompts use |
| Attribute | `attribute_id`, `name`, `description`, `literature_sources` (optional), `markers` | One dimension of the construct, scored on its own |
| Marker | `marker_id`, `name`, `description`, `sub_rubric` | One observable feature of a text that evidences the attribute |
| Level | `1` to `5`, each with a `label` and a `description` | What a text at that level looks like for the marker |

`literature_sources` lists the works a rubric rests on. Give them for the rubric as a whole, for each attribute, or both. The rubric's list is required, though it may be empty; an attribute may leave its list out, which is the same as an empty list. No prompt shows the sources: they document the rubric, and they are part of its content hash.

The two-stage LLM coder asks the model for passages that show each marker, then scores the attribute from 1 to 5. The one-stage LLM coder scores the attribute in a single call. Both read the marker descriptions and the level descriptions from the rubric. The reference texts of the similarity coder are written from the same text.

## Steps

### Step 1: load the toy rubric

<!-- run -->
```python
import copy
import json

import copernicus as cop

rubric = cop.load_rubric(cop.rubric.loader.toy_rubric_path())
print(rubric.concept)
print([(a.attribute_id, len(a.markers)) for a in rubric.attributes])
marker = rubric.attributes[0].markers[0]
print(marker.marker_id, "|", marker.name)
print(marker.level(1).label, "|", marker.level(5).description)
```

<!-- output -->
```text
Constructiveness of written feedback
[('specificity', 3), ('tone', 3), ('reasoning', 3)]
spec_examples | Concrete examples
Clearly low | Every main point is illustrated with a precise, identifiable example from the work.
```

`load_rubric` reads the JSON file, checks it against the schema and the naming rules, and returns an immutable `Rubric`. The rubric has a content hash. A run store records it, so a store refuses to continue with a changed rubric.

<!-- run -->
```python
print(rubric.content_hash())
```

<!-- output -->
```text
4bdee2efb58b0003
```

### Step 2: build a small rubric in code

A rubric is plain data. Write it as a dictionary, or as a JSON file with the same layout, and build it with `rubric_from_dict`. The block below defines a rubric with one attribute and two markers.

<!-- run -->
```python
def levels() -> dict:
    return {
        str(n): {"label": f"Level {n}", "description": f"What level {n} of the marker looks like."}
        for n in range(1, 6)
    }


politeness = {
    "rubric_id": "politeness_v1",
    "concept": "Politeness of a message",
    "concept_definition": "Politeness is the degree to which a message shows respect for its reader.",
    "literature_sources": [],
    "attributes": [
        {
            "attribute_id": "courtesy",
            "name": "Courtesy",
            "description": "Courteous wording.",
            "literature_sources": [],
            "markers": [
                {
                    "marker_id": "greeting",
                    "name": "Greeting",
                    "description": "The message greets the reader.",
                    "sub_rubric": levels(),
                },
                {
                    "marker_id": "thanks",
                    "name": "Thanks",
                    "description": "The message thanks the reader.",
                    "sub_rubric": levels(),
                },
            ],
        }
    ],
    "definitions": {"message": "A short written text sent to one reader."},
}
small = cop.rubric.rubric_from_dict(politeness)
print(small.rubric_id, small.marker_ids("courtesy"))
```

<!-- output -->
```text
politeness_v1 ['greeting', 'thanks']
```

### Step 3: read the error messages

A rubric that breaks a rule is refused with a `RubricValidationError`. The message holds one line per violation, in the form `<json path>: <message>`.

<!-- run -->
```python
broken = copy.deepcopy(politeness)
broken["attributes"][0]["markers"][1]["marker_id"] = "Greeting"
del broken["attributes"][0]["markers"][0]["sub_rubric"]["5"]
try:
    cop.rubric.rubric_from_dict(broken)
except cop.errors.RubricValidationError as error:
    print(error)
```

<!-- output -->
```text
attributes/0/markers/0/sub_rubric: '5' is a required property
attributes/0/markers/0/name: marker name 'Greeting' equals the id of another marker of attribute 'courtesy'; an extraction key could name either marker
attributes/0/markers/1/marker_id: id 'Greeting' collides with another attribute or marker id ('greeting'); ids are compared without regard to case
```

An id cannot take the name of a target the coders generate next to the attributes: `construct`, the whole construct, or a name that ends in `__rollup`, the suffix of the roll-up `<rubric_id>__rollup`.

<!-- run -->
```python
renamed = copy.deepcopy(politeness)
renamed["attributes"][0]["attribute_id"] = "construct"
try:
    cop.rubric.rubric_from_dict(renamed)
except cop.errors.RubricValidationError as error:
    print(error)
```

<!-- output -->
```text
attributes/0/attribute_id: id 'construct' is the name of the whole-construct target
```

### Step 4: write the rubric to a file and check it

<!-- run -->
```python
with open("politeness.json", "w", encoding="utf-8") as handle:
    json.dump(small.to_dict(), handle, indent=2)
print(cop.load_rubric("politeness.json").content_hash() == small.content_hash())
```

<!-- output -->
```text
True
```

On the command line, `copernicus rubric validate` runs the same checks:

```bash
copernicus rubric validate politeness.json
```

It prints `valid: politeness_v1 (1 attributes, 2 markers)` when the file passes. When the file does not pass, it prints `error:` followed by one line for each violation and exits with code 1.

## What the output means

- **A rubric that loads is a rubric every coder can use.** The loader checks the layout (every required field present, every marker with five levels) and the names.
- **The naming rules exist because ids become column names and target names.** An attribute or marker id matches `[A-Za-z_][A-Za-z0-9_]*`, differs from the names the run store and the exports use for their own columns (for example `text_id` or `construct_score`), is not `construct` and does not end in `__rollup` (the names of the whole-construct target and of the roll-up `<rubric_id>__rollup`), and differs from every other id, all when case is ignored. A marker name is unique within its attribute and differs from the ids of the attribute's other markers. A rubric built in code, or changed with `dataclasses.replace`, checks its marker names and its ids against the same rules when it is constructed and refuses with the same messages.
- **The content hash changes when any part of the rubric changes.** Reordering the glossary changes it too, because the prompts list the glossary in order.

## Pitfalls

- **Edit a rubric after coding and the store refuses to continue.** The run store records the rubric hash. Code with the final rubric, or start a new store.
- **Write levels that a reader can tell apart.** The scoring stage picks the level whose description fits the verified passages best. Levels that differ only in an adverb give noisy scores.
- **Describe markers as features of the text.** A marker such as "the feedback is good" cannot be pointed to. A marker such as "the feedback names the exact part of the work" can.
- **Keep the glossary short.** Every term in `definitions` enters the prompts of every call.
- **Do not reuse an id for two markers, even in different attributes.** Ids are unique across the whole rubric.

## Next

[The two-stage coder](two-stage-coder.md) codes texts with this rubric.
