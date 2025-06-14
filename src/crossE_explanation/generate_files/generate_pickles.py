import pandas as pd
import numpy as np
import pickle
import argparse
import os
from collections import defaultdict
from src.dataset import Dataset

# Configurazione argomenti
parser = argparse.ArgumentParser(description='Genera file pickle per le predizioni.')
parser.add_argument('--dataset', type=str, required=True, help='Nome del dataset da processare')
parser.add_argument('--model', type=str, required=True, help='Nome del modello usato')
args = parser.parse_args()

# Caricamento dataset
dataset_obj = Dataset(dataset=args.dataset)

# Caricamento triple di test
test_triples = []
file_path = f"./data/{args.dataset}/test.txt"
with open(file_path, "r") as f:
    for line in f:
        s, p, o = line.strip().split('\t')
        test_triples.append((s, p, o))

# Caricamento predizioni
file_path = f"./selected_preds/{args.model}_{args.dataset}.csv"
transe_data = pd.read_csv(file_path, sep='\t', header=None, names=['s', 'p', 'o'])

# Preparazione predizioni
predictions = defaultdict(list)
for _, row in transe_data.iterrows():
    key = (row['s'], row['p'])
    predictions[key].append(row['o'])

# Costruzione strutture finali (solo per triple con predizioni)
test_triples_ids = []
test_predicted_tails = []

for s, p, o in test_triples:
    key = (s, p)
    if key not in predictions:
        continue  # salta triple senza predizioni

    preds = predictions[key]

    if o not in preds:
        continue

    try:
        h = dataset_obj.entity_to_id[s]
        r = dataset_obj.relation_to_id[p]
        t = dataset_obj.entity_to_id[o]
    except KeyError as e:
        print(f"Errore con la tripla {s}, {p}, {o}: {str(e)} mancante nel dataset")
        continue

    tail_ids = [dataset_obj.entity_to_id[tail] for tail in preds
                if tail != o and tail in dataset_obj.entity_to_id]

    if o in dataset_obj.entity_to_id:
        tail_ids.insert(0, dataset_obj.entity_to_id[o])

    if tail_ids:
        test_triples_ids.append([h, t, r])
        test_predicted_tails.append(np.array(tail_ids))

# Creazione directory se non esiste
model_pickle_dir = f"pickles/{args.model}_{args.dataset}"
os.makedirs(model_pickle_dir, exist_ok=True)

# Salvataggio
with open(f"{model_pickle_dir}/test_triples.pkl", "wb") as f:
    pickle.dump(test_triples_ids, f)

with open(f"{model_pickle_dir}/test_predicted_tails.pkl", "wb") as f:
    pickle.dump(test_predicted_tails, f)

print(f"File test_triples.pkl e test_predicted_tails.pkl generati in {model_pickle_dir}/")
