import argparse
import os
import shutil
from src.dataset import Dataset

def save_dataset_mappings(dataset_name: str):
    """
    Carica un dataset e salva i mapping entity_to_id e relation_to_id in file separati.
    """
    print(f"Caricamento dataset {dataset_name}...")
    dataset = Dataset(dataset_name)

    dataset.save_mappings_to_separate_txt(dataset_name)

    print(f"File creati:")
    print("- entity2id.txt")
    print("- relation2id.txt")

    return dataset

def backup_original_files(dataset_name: str):
    """
    Sposta i file train.txt, valid.txt e test.txt originali nella cartella original_files/.
    """
    data_dir = f"data/{dataset_name}"
    backup_dir = os.path.join(data_dir, "original_files")
    os.makedirs(backup_dir, exist_ok=True)

    for split in ["train", "valid", "test"]:
        file_path = os.path.join(data_dir, f"{split}.txt")
        if os.path.exists(file_path):
            shutil.copy(file_path, os.path.join(backup_dir, f"{split}.txt"))
            print(f"Backup creato per {split}.txt")
        else:
            print(f"Attenzione: {split}.txt non trovato in {data_dir}")

def overwrite_with_filtered(dataset: Dataset, dataset_name: str):
    """
    Sovrascrive i file di triple (train, valid, test) con versioni filtrate (ottenute con il caricamento con Dataset).
    """
    id2entity = {v: k for k, v in dataset.entity_to_id.items()}
    id2relation = {v: k for k, v in dataset.relation_to_id.items()}
    data_dir = f"data/{dataset_name}"

    def save_triples(triples, file_path):
        with open(file_path, "w", encoding="utf-8") as f:
            for h, r, t in triples:
                f.write(f"{id2entity[h]}\t{id2relation[r]}\t{id2entity[t]}\n")

    save_triples(dataset.training_triples, os.path.join(data_dir, "train.txt"))
    save_triples(dataset.validation_triples, os.path.join(data_dir, "valid.txt"))
    save_triples(dataset.testing_triples, os.path.join(data_dir, "test.txt"))

    print("Triple filtrate sovrascritte nei file:")
    print(f"- {data_dir}/train.txt")
    print(f"- {data_dir}/valid.txt")
    print(f"- {data_dir}/test.txt")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Processa un dataset e filtra le triple.")
    parser.add_argument("dataset_name", type=str, help="Nome del dataset nella cartella data/")

    args = parser.parse_args()
    dataset_name = args.dataset_name

    dataset = save_dataset_mappings(dataset_name)
    backup_original_files(dataset_name)
    overwrite_with_filtered(dataset, dataset_name)
