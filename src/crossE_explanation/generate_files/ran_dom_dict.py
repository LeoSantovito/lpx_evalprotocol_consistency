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

    dataset_path = f'data/{args.dataset}'

    # Inizializza il dataset
    dataset = Dataset(args.dataset)

    # Ottieni il mapping relazione-id dal dataset
    relation2id = dataset.relation_to_id

    # Carica class2id.txt
    class2id = {}
    with open(f'{dataset_path}/class2id.txt', 'r', encoding='utf-8') as f:
        next(f)  # Salta l'header
        for line in f:
            class_name, class_id = line.strip().split('\t')
            class2id[class_name] = int(class_id)

    # Carica il file relations.csv
    relations_df = pd.read_csv(f'{dataset_path}/reasoned/relations.csv')

    # Costruisce i dizionari relazione → classi di dominio e range
    rs_domain_dict = {}
    rs_range_dict = {}

    for _, row in relations_df.iterrows():
        relation_name = row['label']

        if relation_name in relation2id:
            relation_id = relation2id[relation_name]

            # Domini
            domains = parse_classlike_field(row['domainSet'])
            domain_ids = [class2id[d] for d in domains if d in class2id]
            if domain_ids:
                rs_domain_dict[relation_id] = domain_ids

            # Range
            ranges = parse_classlike_field(row['rangeSet'])
            range_ids = [class2id[r] for r in ranges if r in class2id]
            if range_ids:
                rs_range_dict[relation_id] = range_ids

    # Salvataggio su file .pkl
    with open(f'{dataset_path}/rs_domain2id_dict.pkl', 'wb') as f:
        pickle.dump(rs_domain_dict, f)

    with open(f'{dataset_path}/rs_range2id_dict.pkl', 'wb') as f:
        pickle.dump(rs_range_dict, f)

    print("Dizionari rs_domain2id_dict.pkl e rs_range2id_dict.pkl creati con successo!")


if __name__ == "__main__":
    main()