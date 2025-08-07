# Datasets

This folder houses dataset loaders, preprocessing utilities, and metadata for
all corpora used in the project.

Planned datasets
----------------
1. **OpenWebText** – large-scale web crawl that approximates the content used
   for OpenAI's WebText. Serves as a general-purpose pre-training corpus.
2. **ARC** – the AI2 Reasoning Challenge, a multiple-choice science exam used
   to probe reasoning and world knowledge.
3. **GSM8K** – a grade-school math word-problem dataset that tests arithmetic
   and multi-step reasoning.

Each dataset will get its own Python module under `datasets/` once we begin
implementing loaders. For now, refer to the notes in `proposal.txt` for
motivation and design considerations.
