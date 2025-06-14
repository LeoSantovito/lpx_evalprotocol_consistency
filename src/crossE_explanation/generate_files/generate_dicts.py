import pickle
import argparse
import os
from collections import defaultdict
from src.dataset import Dataset

def default_dict_factory():
    return defaultdict(set)

# Configurazione argomenti
parser = argparse.ArgumentParser(description='Genera dizionari per il dataset specificato.')
parser.add_argument('--dataset', type=str, required=True, help='Nome del dataset da processare')
parser.add_argument('--model', type=str, required=True, help='Nome del modello usato')
args = parser.parse_args()

# Inizializzazione dataset
dataset_obj = Dataset(dataset=args.dataset)

# Caricamento triple di training
train_triples = []
file_path = f"data/{args.dataset}/train.txt"
with open(file_path, "r") as f:
    for line in f.readlines():
        s, p, o = line.strip().split('\t')
        train_triples.append((s, p, o))

# Costruzione dizionari
train_hr_t = defaultdict(default_dict_factory)  # (head, relation) -> {tails}
train_tr_h = defaultdict(default_dict_factory)  # (tail, relation) -> {heads}

for s, p, o in train_triples:
    try:
        h = dataset_obj.entity_to_id[s]
        r = dataset_obj.relation_to_id[p]
        t = dataset_obj.entity_to_id[o]
        train_hr_t[h][r].add(t)
        train_tr_h[t][r].add(h)
    except KeyError as e:
        print(f"Errore con la tripla {s}, {p}, {o}: {str(e)} mancante nel dataset")
        continue

# Creazione directory se non esiste
model_pickle_dir = f"pickles/{args.model}_{args.dataset}"
os.makedirs(model_pickle_dir, exist_ok=True)

# Salvataggio in formato pickle
with open(f"{model_pickle_dir}/train_hr_t.pkl", "wb") as f:
    pickle.dump(dict(train_hr_t), f)

with open(f"{model_pickle_dir}/train_tr_h.pkl", "wb") as f:
    pickle.dump(dict(train_tr_h), f)

print(f"File train_hr_t.pkl e train_tr_h.pkl generati in {model_pickle_dir}/")