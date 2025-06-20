# An Empirical Study of the Consistency between Protocols for Evaluating Explanations of Predicted Links in Knowledge Graphs

## Create auxiliary files

Run:

```bash
python -m src.crossE_explanation.generate_files.map_and_filter <dataset>
python -m src.crossE_explanation.generate_files.generate_dicts --dataset <dataset> --model <model>
python -m src.crossE_explanation.generate_files.generate_pickles --dataset <dataset> --model <model
```

For the semantic similarity

```bash
python -m src.crossE_explanation.generate_files.semantic_data <dataset>
python -m src.crossE_explanation.generate_files.ran_dom_dict --dataset <dataset>
python -m src.crossE_explanation.generate_files.classes_dict --dataset <dataset>
```
## Compute Similarity

```bash
python -m src.crossE_explanation.batch_similarity --data kge_models/<model>_<dataset>.pt --save_dir pickles/<model>_<dataset>/<distance>/ --distance <distance> --dataset <dataset> --complex
```

## Generate Explanations

```bash
python -m src.crossE_explanation.explanation --data pickles/<model>_<dataset>/ --predictions_perc 100 --distance <distance> --pretty_print True --model <model> --dataset <dataset>
```

## Evaluate Explanations

```bash
python -m src.evaluation.re-training --dataset <dataset> --model <model> --method kelpie --mode necessary
python -m src.evaluation.re-training_metrics --dataset <dataset> --model <model> --method kelpie --mode necessary
```
