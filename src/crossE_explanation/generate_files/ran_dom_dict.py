import csv
import pickle
import argparse

def load_id_file(filepath):
    result = {}
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            name, id_ = line.strip().split('\t')
            result[name] = int(id_)
    return result

def main():
    # Configura il parser degli argomenti
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True, help="Nome del dataset (es. 'DB50K')")
    args = parser.parse_args()

    # Carica i dizionari id
    relation2id = load_id_file(f'data/{args.dataset}/relation2id.txt')
    domain2id = load_id_file(f'data/{args.dataset}/domain2id.txt')
    range2id = load_id_file(f'data/{args.dataset}/range2id.txt')

    # Dizionari finali
    rs_domain2id_dict = {}
    rs_range2id_dict = {}

    # Lettura CSV
    with open(f'data/{args.dataset}/relations.csv', 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            relation_name = row['relation']
            domains_str = row['domains']
            ranges_str = row['ranges']

            relation_id = relation2id.get(relation_name)
            if relation_id is None:
                continue


            try:
                domains_set = eval(domains_str) if domains_str.strip() else set()
                ranges_set = eval(ranges_str) if ranges_str.strip() else set()
            except Exception as e:
                print(f"Errore parsing per relazione {relation_name}: {e}")
                continue

            # Mappa a ID, con controllo
            try:
                domain_ids = {domain2id[domain] for domain in domains_set if domain in domain2id}
                range_ids = {range2id[range_] for range_ in ranges_set if range_ in range2id}
            except KeyError as e:
                print(f"Errore KeyError con {e} in {relation_name}")
                continue

            rs_domain2id_dict[relation_id] = list(domain_ids)
            rs_range2id_dict[relation_id] = list(range_ids)

    # Salvataggio pickle
    with open(f'data/{args.dataset}/rs_domain2id_dict.pkl', 'wb') as f:
        pickle.dump(rs_domain2id_dict, f)

    with open(f'data/{args.dataset}/rs_range2id_dict.pkl', 'wb') as f:
        pickle.dump(rs_range2id_dict, f)

    print("Dizionari salvati con successo nei file rs_domain2id_dict.pkl e rs_range2id_dict.pkl.")

if __name__ == "__main__":
    main()