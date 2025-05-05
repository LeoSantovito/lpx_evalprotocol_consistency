import pandas as pd
import numpy as np
import pickle
import argparse
import os
from collections import defaultdict

# Configurazione argomenti
parser = argparse.ArgumentParser(description='Genera file pickle per le predizioni.')
parser.add_argument('--dataset', type=str, required=True, help='Nome del dataset da processare')
parser.add_argument('--model', type=str, required=True, help='Nome del modello usato')
args = parser.parse_args()

# Caricamento mapping
entity2id = {}
file_path = f"./data/{args.dataset}/entity2id.txt"
with open(file_path, "r") as f:
    for line in f.readlines():
        name, eid = line.strip().split('\t')
        entity2id[name] = int(eid)

relation2id = {}
file_path = f"./data/{args.dataset}/relation2id.txt"
with open(file_path, "r") as f:
    for line in f.readlines():
        name, rid = line.strip().split('\t')
        relation2id[name] = int(rid)

# Caricamento triple di test
test_triples = []
file_path = f"./data/{args.dataset}/test.txt"
with open(file_path, "r") as f:
    for line in f.readlines():
        s, p, o = line.strip().split('\t')
        test_triples.append((s, p, o))

# Caricamento predizioni
file_path = f"./preds/{args.model}_{args.dataset}.csv"
transe_data = pd.read_csv(file_path, sep=';')

# Preparazione predizioni
predictions = defaultdict(list)
for _, row in transe_data.iterrows():
    key = (row['s'], row['p'])
    predictions[key].append((row['o'], row['o_rank']))

# Costruzione strutture finali
test_triples_ids = []
test_predicted_tails = []

# Modifica questa parte del codice:
for s, p, o in test_triples:
    try:
        h = entity2id[s]
        r = relation2id[p]
        t = entity2id[o]
        test_triples_ids.append([h, t, r])

        preds = predictions.get((s, p), [])
        sorted_preds = sorted(preds, key=lambda x: x[1])
        tail_ids = [entity2id[tail] for tail, _ in sorted_preds if tail != o and tail in entity2id]

        if o in entity2id:
            tail_ids.insert(0, entity2id[o])
        else:
            print(f"Attenzione: entità {o} mancante in entity2id.txt")
            continue

        test_predicted_tails.append(np.array(tail_ids))
    except KeyError as e:
        print(f"Errore con la tripla {s}, {p}, {o}: {str(e)} mancante")
        continue
        
# Creazione directory modello se non esiste
model_pickle_dir = f"pickles/{args.model}_{args.dataset}"
os.makedirs(model_pickle_dir, exist_ok=True)

# Salvataggio
with open(f"{model_pickle_dir}/test_triples.pkl", "wb") as f:
    pickle.dump(test_triples_ids, f)

with open(f"{model_pickle_dir}/test_predicted_tails.pkl", "wb") as f:
    pickle.dump(test_predicted_tails, f)

print(f"File generati correttamente in {model_pickle_dir}/")