import pandas as pd
import pickle
import ast
import argparse

def main():
    # Configura il parser degli argomenti
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True, help="Nome del dataset (es. 'DB50K')")
    args = parser.parse_args()

    # Carica il file entities.csv (entità, classi)
    entities_df = pd.read_csv(f'data/{args.dataset}/entities.csv')

    # Carica entity2id.txt (nome -> id entità)
    entity2id = {}
    with open(f'data/{args.dataset}/entity2id.txt', 'r', encoding='utf-8') as f:
        for line in f:
            entity_name, entity_id = line.strip().split('\t')
            entity2id[entity_name] = int(entity_id)

    # Carica class2id.txt (nome -> id classe)
    class2id = {}
    with open(f'data/{args.dataset}/class2id.txt', 'r', encoding='utf-8') as f:
        for line in f:
            class_name, class_id = line.strip().split('\t')
            class2id[class_name] = int(class_id)

    # Crea il dizionario ent2class_dict
    ent2class_dict = {}

    for _, row in entities_df.iterrows():
        entity_name = row['entity']
        # Converte la stringa del set in un vero oggetto set
        classes = ast.literal_eval(row['classes'])

        if entity_name in entity2id:
            entity_id = entity2id[entity_name]
            # Aggiungi tutte le classi a cui l'entità appartiene
            class_ids = [class2id[class_name] for class_name in classes if class_name in class2id]

            # Se l'entità ha più classi, associamo la lista degli ID delle classi
            ent2class_dict[entity_id] = class_ids

    # Salva il dizionario come file .pkl
    with open(f'data/{args.dataset}/entity2class_dict.pkl', 'wb') as f:
        pickle.dump(ent2class_dict, f)

    print("Dizionario entity2class_dict.pkl creato con successo!")

if __name__ == "__main__":
    main()