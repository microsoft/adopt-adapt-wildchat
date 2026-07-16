# Code for "Adopt ≠ Adapt"
This repository accompanies the paper
> Rebecca M. M. Hicke and Kiran Tomlinson. Adopt ≠ Adapt: Longitudinal Analyses of LLM Conversations in the Wild. arXiv, 2026. https://arxiv.org/abs/2605.29018

## Contents
This repository contains code for running LLM classifiers on WildChat data and making all WildChat plots from the paper. 
In addition, the repository contains classifier outputs from our run of the WildChat pipeline. 
Bing Copilot data analyses are excluded as the dataset is private.

The repository contains the following files:
- `classify.py`: Runs the classification pipeline over WildChat data.
- `worker_pool.py`: Async OpenAI runner for LLM calls.
- `compute_sentence_lengths.py`: Computes the sentence-length data used in the analysis.
- `wildchat_metrics.py`: Loads classification results, computes metrics, and caches them.
- `wildchat_plots.py`: Generates figures and tables from WildChat classification results.
- `requirements.txt`: Lists the Python dependencies used to run the code.
- `wildchat-results/wildchat-4.8m-results/`: Contains the released sentence-length data and domain, intent, and task classifier outputs under `no-content/`. The classifier outputs include keys and labels but not conversation text.


## Libraries
Tested with:
- `python==3.12.10`
- `openai==2.3.0`
- `pydantic==2.9.2`
- `tiktoken==0.8.0`
- `pandas==2.2.3`
- `polars==1.39.3`
- `numpy==2.1.3`
- `scipy==1.14.1`
- `statsmodels==0.14.4`
- `matplotlib==3.9.2`
- `pyarrow==21.0.0`
- `spacy==3.8.14` with `en_core_web_sm==3.8.0`
- `tqdm==4.67.1`

Install the dependencies and spaCy model with:

```
python -m pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

## Reproducing
First, download WildChat-4.8M from HuggingFace: https://huggingface.co/datasets/allenai/WildChat-4.8M. 
Place the downloaded files under `wildchat-data/wildchat-4.8m/`; for example,
the first shard should be `wildchat-data/wildchat-4.8m/train-00000-of-00086.parquet`.

To rerun the classification pipeline:
1. Set the environment variable `OPENAI_API_KEY` to your API key.
2. Run `python classify.py`. Expect this to take some time, so running in `screen`/`tmux`/etc is a good idea. 
Results will be saved in `wildchat-results/wildchat-4.8m-results/`. Each result
file contains the columns needed to align it with WildChat plus the classifier
outputs; conversation text is not copied into the result files. The cost of
running the classification pipeline with GPT-4o-mini on WildChat-4.8M (at
$0.15 / 1M input tokens, $0.60 / 1M output tokens) is ~ $3500.
3. Run `python compute_sentence_lengths.py`.

To rerun the plotting code, run `python wildchat_plots.py` to generate the plots and tables under
`outputs/`.

The plotting code accepts both the direct outputs from `classify.py` and the
released results whose conversation text has been removed and whose files are
stored under `wildchat-results/wildchat-4.8m-results/no-content/`.

## Additional information

This repository is published by Microsoft. For information about how Microsoft processes personal data, please see the Microsoft Privacy Statement: https://go.microsoft.com/fwlink/?LinkId=521839