import wandb
import copy
import click
import pandas as pd
import numpy as np
from collections import defaultdict
from pathlib import Path

from .. import DATASETS, METHODS, MODELS, MODES
from .. import CRIAGE, DATA_POISONING, IMAGINE, KELPIE, KELPIEPP
from .. import NECESSARY, SUFFICIENT
from .. import LP_CONFIGS_PATH, DATA_PATH
from ..dataset import MANY_TO_ONE, ONE_TO_ONE
from ..explanation_builders.summarization import SUMMARIZATIONS
from ..dataset import Dataset
from ..link_prediction.evaluation import Evaluator
from ..utils import format_paths
from ..utils import read_json, write_json
from ..utils import init_model, load_model
from ..utils import init_optimizer
from ..utils import set_seeds
import time


def load_mappings_from_files(dataset_path):
    """Carica direttamente le mappature dai file"""
    entity_to_id = {}
    with open(dataset_path / "entity2id.txt", 'r', encoding='utf-8-sig') as f:
        for line in f:
            parts = line.strip().split('\t') if '\t' in line else line.split()
            if len(parts) >= 2:
                entity_to_id[parts[0]] = int(parts[1])

    relation_to_id = {}
    with open(dataset_path / "relation2id.txt", 'r', encoding='utf-8-sig') as f:
        for line in f:
            parts = line.strip().split('\t') if '\t' in line else line.split()
            if len(parts) >= 2:
                relation_to_id[parts[0]] = int(parts[1])

    return entity_to_id, relation_to_id


def verify_and_correct_mappings(dataset_path, dataset_obj):
    """Verifica e corregge le mappature"""
    file_entity_to_id, file_relation_to_id = load_mappings_from_files(dataset_path)

    try:
        # Crea nuovo dataset
        corrected_dataset = Dataset(dataset=dataset_obj.dataset_name)

        # Forza le mappature corrette (se possibile)
        if hasattr(corrected_dataset, '_entity_to_id'):
            corrected_dataset._entity_to_id = file_entity_to_id
            corrected_dataset._relation_to_id = file_relation_to_id
            corrected_dataset._id_to_entity = {v: k for k, v in file_entity_to_id.items()}
            corrected_dataset._id_to_relation = {v: k for k, v in file_relation_to_id.items()}
            return corrected_dataset
        else:
            return dataset_obj

    except Exception:
        return dataset_obj


def format_result(results, new_results, pred):
    result = results[pred]
    new_result = new_results[pred]

    score = result["score"]["tail"]
    rank = result["rank"]["tail"]
    new_score = new_result["score"]["tail"]
    new_rank = new_result["rank"]["tail"]

    return {
        "score": str(score),
        "rank": str(rank),
        "new_score": str(new_score),
        "new_rank": str(new_rank),
    }


def get_results(model, triples):
    evaluator = Evaluator(model=model)
    evaluator.evaluate(np.array(triples))
    results = evaluator.results
    results = {c: result for c, result in zip(triples, results)}
    return results


@click.command()
@click.option("--dataset", type=click.Choice(DATASETS))
@click.option("--model", type=click.Choice(MODELS))
@click.option("--method", type=click.Choice(METHODS), default=KELPIE)
@click.option("--mode", type=click.Choice(MODES))
@click.option("--summarization", type=click.Choice(SUMMARIZATIONS))
def main(
        dataset,
        model,
        method,
        mode,
        summarization,
):
    set_seeds(42)
    start_time = time.time()  

    paths = format_paths(method, mode, model, dataset, summarization)
    dataset_path = DATA_PATH / dataset

    dataset_obj = Dataset(dataset=dataset)
    dataset_obj = verify_and_correct_mappings(dataset_path, dataset_obj)

    lp_config_path = LP_CONFIGS_PATH / f"{model}_{dataset}.json"
    lp_config = read_json(lp_config_path)
    lp_config["training"]["epochs"] = lp_config["training"]["trained_epochs"]
    config = read_json(paths["configs"])
    outputs = read_json(paths["exps"])

    wandb.init(project="dixti", config=config)

    model = load_model(lp_config, dataset_obj)
    model.eval()

    preds = []

    if method in [CRIAGE, DATA_POISONING, KELPIE, KELPIEPP]:
        if mode == NECESSARY:
            removals = []
            pred_to_explanation = defaultdict(list)

            for output in outputs:
                pred = dataset_obj.ids_triple(output["pred"])
                preds.append(pred)
                explanation = output["explanation"]

                explanation_ids = []
                for triple in explanation:
                    conv_triple = dataset_obj.ids_triple(triple)
                    explanation_ids.append(conv_triple)

                pred_to_explanation[pred] = explanation_ids
                removals += explanation_ids

            unique_removals = list({tuple(t) for t in removals})
            new_dataset = copy.deepcopy(dataset_obj)
            existing_removals = [t for t in unique_removals if t in new_dataset.training_triples]

            for triple in existing_removals:
                try:
                    new_dataset.remove_training_triple(triple)
                except ValueError:
                    continue

            results = get_results(model, preds)

            new_model = init_model(lp_config, new_dataset)
            optimizer = init_optimizer(lp_config, new_model)
            optimizer.train(training_triples=new_dataset.training_triples)

            new_model.eval()
            new_results = get_results(new_model, preds)

            evaluations = []
            for pred in preds:
                result = format_result(results, new_results, pred)
                explanation = [dataset_obj.labels_triple(triple) for triple in pred_to_explanation[pred]]
                evaluation = {
                    "pred": dataset_obj.labels_triple(pred),
                    "explanation": explanation,
                    "result": result
                }
                evaluations.append(evaluation)

    write_json(evaluations, paths["evals"])
    eval_df = pd.DataFrame.from_records(evaluations)
    eval_df["pred"] = eval_df["pred"].map(" ".join)

    if mode == SUFFICIENT:
        for i in range(10):
            if f"conversion_{i}.additions" in eval_df.columns:
                eval_df[f"conversion_{i}.additions"] = eval_df[f"conversion_{i}.additions"].fillna("")
                eval_df[f"conversion_{i}.additions"] = eval_df[f"conversion_{i}.additions"].map(
                    lambda e: [] if e == "" else e)
                eval_df[f"conversion_{i}.additions"] = eval_df[f"conversion_{i}.additions"].map(
                    lambda e: [" ".join(t) for t in e])
                eval_df[f"conversion_{i}.additions"] = eval_df[f"conversion_{i}.additions"].map("\n".join)

    eval_df["explanation"] = eval_df["explanation"].apply(lambda e: [" ".join(t) for t in e])
    eval_df["explanation"] = eval_df["explanation"].map("\n".join)

    table = wandb.Table(dataframe=eval_df)
    wandb.log({"evaluations": table})

    end_time = time.time()
    print(f"Tempo totale di esecuzione: {end_time - start_time:.2f} secondi")


if __name__ == "__main__":
    main()