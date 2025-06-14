import pandas as pd
import pickle
import argparse
from src.dataset import Dataset

def parse_classlike_field(value):
    """Pulisce e restituisce una lista di valori da stringhe tipo {dbo:A, dbo:B}"""
    if pd.isna(value):
        return []
    cleaned = value.strip('{}').replace(' ', '')
    return cleaned.split(',') if cleaned else []

def main():
    # Parser degli argomenti
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True, help="Nome del dataset (es. 'DB50K')")
    args = parser.parse_args()

    # Inizializza la classe Dataset
    dataset = Dataset(args.dataset)

    # Carica il file entities.csv
    entities_df = pd.read_csv(f'data/{args.dataset}/reasoned/entities.csv')

    # Ottieni entity_to_id dalla classe Dataset
    entity2id = dataset.entity_to_id

    # Carica class2id.txt
    class2id = {}
    with open(f'data/{args.dataset}/class2id.txt', 'r', encoding='utf-8') as f:
        next(f)  # Salta l'header se presente
        for line in f:
            class_name, class_id = line.strip().split('\t')
            class2id[class_name] = int(class_id)

    # Costruisce il dizionario entità → classi
    ent2class_dict = {}

    for _, row in entities_df.iterrows():
        entity_name = row['entity']
        classes = parse_classlike_field(row['classes'])

        if entity_name in entity2id:
            entity_id = entity2id[entity_name]
            class_ids = [class2id[c] for c in classes if c in class2id]
            if class_ids:
                ent2class_dict[entity_id] = class_ids
            else:
                ent2class_dict[entity_id] = []

    # Salvataggio su file .pkl
    with open(f'data/{args.dataset}/entity2class_dict.pkl', 'wb') as f:
        pickle.dump(ent2class_dict, f)

    print("Dizionario entity2class_dict.pkl creato con successo!")

if __name__ == "__main__":
    main()