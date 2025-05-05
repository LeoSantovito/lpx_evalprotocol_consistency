import pickle
import argparse
import os
from collections import defaultdict

def default_dict_factory():
    return defaultdict(set)

# Configurazione argomenti
parser = argparse.ArgumentParser(description='Genera dizionari per il dataset specificato.')
parser.add_argument('--dataset', type=str, required=True, help='Nome del dataset da processare')
parser.add_argument('--model', type=str, required=True, help='Nome del modello usato')
args = parser.parse_args()

# Caricamento mapping
data_path = "../../../data/"
entity2id = {}
file_path = f"{data_path}{args.dataset}/entity2id.txt"
with open(file_path, "r") as f:
    for line in f.readlines():
        name, eid = line.strip().split('\t')
        entity2id[name] = int(eid)

relation2id = {}
file_path = f"{data_path}{args.dataset}/relation2id.txt"
with open(file_path, "r") as f:
    for line in f.readlines():
        name, rid = line.strip().split('\t')
        relation2id[name] = int(rid)

# Caricamento triple di training
train_triples = []
file_path = f"{data_path}{args.dataset}/train.txt"
with open(file_path, "r") as f:
    for line in f.readlines():
        s, p, o = line.strip().split('\t')
        train_triples.append((s, p, o))

# Costruzione dizionari
train_hr_t = defaultdict(default_dict_factory)  # (head, relation) -> {tails}
train_tr_h = defaultdict(default_dict_factory)  # (tail, relation) -> {heads}

for s, p, o in train_triples:
    h = entity2id[s]
    r = relation2id[p]
    t = entity2id[o]
    train_hr_t[h][r].add(t)
    train_tr_h[t][r].add(h)

# Creazione directory modello se non esiste
model_pickle_dir = f"pickles/{args.model}_{args.dataset}"
os.makedirs(model_pickle_dir, exist_ok=True)

# Salvataggio in formato pickle
with open(f"{model_pickle_dir}/train_hr_t.pkl", "wb") as f:
    pickle.dump(dict(train_hr_t), f)

with open(f"{model_pickle_dir}/train_tr_h.pkl", "wb") as f:
    pickle.dump(dict(train_tr_h), f)

print(f"File generati correttamente in {model_pickle_dir}/")