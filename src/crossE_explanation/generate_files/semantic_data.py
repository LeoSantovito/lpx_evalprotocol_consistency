import csv
import os
import sys


def extract_classes_from_set(class_set):
    """Estrae le classi da una stringa che rappresenta un set (formato {classe1, classe2})"""
    if not class_set or class_set.strip() == "":
        return set()
    # Rimuovi parentesi graffe e split su virgole
    return {c.strip() for c in class_set.strip('{}').split(',') if c.strip()}


def generate_class_mapping(relations_path, entities_path):
    classes = set()

    # Processa il file relations.csv
    with open(relations_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Estrai classi da domainSet e rangeSet
            domain_classes = extract_classes_from_set(row['domainSet'])
            range_classes = extract_classes_from_set(row['rangeSet'])
            classes.update(domain_classes)
            classes.update(range_classes)

    # Processa il file entities.csv
    with open(entities_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Estrai classi da classSet
            entity_classes = extract_classes_from_set(row['classes'])
            classes.update(entity_classes)

    # Ordina le classi e assegna ID
    sorted_classes = sorted(classes)
    return {cls: idx for idx, cls in enumerate(sorted_classes, start=1)}


def write_class2id_file(class2id, output_path):
    """Scrive il mapping classe-ID su file"""
    with open(output_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f, delimiter='\t')
        writer.writerow(['class', 'id'])
        for cls, id_ in class2id.items():
            writer.writerow([cls, id_])
    print(f"Creato file {output_path} con {len(class2id)} classi uniche.")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python script.py <dataset_name>")
        print("Esempio: python script.py my_dataset")
        sys.exit(1)

    dataset_name = sys.argv[1]
    data_dir = os.path.join('data', dataset_name)

    # Verifica che la cartella esista
    if not os.path.exists(data_dir):
        print(f"Errore: la cartella {data_dir} non esiste")
        sys.exit(1)

    # Path dei file di input
    relations_path = os.path.join(data_dir, 'reasoned/relations.csv')
    entities_path = os.path.join(data_dir, 'reasoned/entities.csv')

    # Verifica che i file esistano
    if not os.path.isfile(relations_path):
        print(f"Errore: file {relations_path} non trovato")
        sys.exit(1)
    if not os.path.isfile(entities_path):
        print(f"Errore: file {entities_path} non trovato")
        sys.exit(1)

    # Genera il mapping
    class2id = generate_class_mapping(relations_path, entities_path)

    # Path del file di output (nella stessa cartella del dataset)
    output_path = os.path.join(data_dir, 'class2id.txt')

    # Scrivi il file di output
    write_class2id_file(class2id, output_path)