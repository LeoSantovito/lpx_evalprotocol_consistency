import pandas as pd
import re
import argparse

def main():
    # Configura il parser degli argomenti
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True, help="Nome del dataset (es. 'DB50K')")
    args = parser.parse_args()

    # File di input
    entities_file = f'data/{args.dataset}/entities.csv'
    relations_file = f'data/{args.dataset}/relations.csv'

    # Funzione per pulire e filtrare nomi indesiderati
    def pulisci_nome(nome):
        nome = re.sub(r"[\"\'{}\[\]]", "", nome).strip()
        if nome.lower() in ["set()"]:  # ignora vuoti o "set()"
            return None
        return nome

    # Estrai valori singoli e puliti da una colonna CSV
    def estrai_valori_puliti(colonna):
        valori = set()
        for val in colonna.dropna():
            for v in str(val).split(','):
                nome_pulito = pulisci_nome(v)
                if nome_pulito:
                    valori.add(nome_pulito)
        return sorted(valori)

    # Caricamento dati
    entities_df = pd.read_csv(entities_file)
    relations_df = pd.read_csv(relations_file)

    # Estrazione e pulizia
    classi = estrai_valori_puliti(entities_df['classes'])
    domini = estrai_valori_puliti(relations_df['domains'])
    range = estrai_valori_puliti(relations_df['ranges'])

    # Salvataggio
    def salva_mappa(nome_file, lista):
        with open(nome_file, 'w', encoding='utf-8') as f:
            for idx, nome in enumerate(lista):
                f.write(f"{nome}\t{idx}\n")

    # Scrivi i file
    salva_mappa(f"data/{args.dataset}/class2id.txt", classi)
    salva_mappa(f"data/{args.dataset}/domain2id.txt", domini)
    salva_mappa(f"data/{args.dataset}/range2id.txt", range)

    print("File generati correttamente.")

if __name__ == "__main__":
    main()