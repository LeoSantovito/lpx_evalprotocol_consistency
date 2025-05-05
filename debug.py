from src.dataset import Dataset

# 1. Inizializzazione del dataset
print("1. Caricamento del dataset 'Countries'...")
countries_dataset = Dataset("DB50K")

# 2. Informazioni base del dataset
print("\n2. Informazioni base del dataset:")
print(f"- Numero di entità: {countries_dataset.num_entities}")
print(f"- Numero di relazioni: {countries_dataset.num_relations}")
print(f"- Numero di triple di training: {len(countries_dataset.training_triples)}")
print(f"- Numero di triple di test: {len(countries_dataset.testing_triples)}")
print(f"- Numero di triple di validazione: {len(countries_dataset.validation_triples)}")

# 3. Esempio di conversione ID-label
print("\n3. Conversione di alcune triple da ID a label:")
sample_triple = countries_dataset.training_triples[0]  # Prendo la prima tripla di training
print(f"- Tripla in ID: {sample_triple}")
print(f"- Tripla in label: {countries_dataset.labels_triple(sample_triple)}")
print(f"- Formattazione stampabile: {countries_dataset.printable_triple(sample_triple)}")

# 4. Esempio di accesso alle relazioni di un'entità
print("\n4. Accesso alle relazioni di un'entità:")
sample_entity_id = countries_dataset.training_triples[0][0]  # Prendo la prima entità della prima tripla
print(f"- ID entità: {sample_entity_id}")
print(f"- Nome entità: {countries_dataset.id_to_entity[sample_entity_id]}")
print(f"- Triple collegate (training): {len(countries_dataset.entity_to_training_triples[sample_entity_id])}")

# 5. Esempio di sottografo
print("\n5. Creazione di un sottografo per un'entità:")
subgraph = countries_dataset.get_subgraph(sample_entity_id)
print(f"- Numero di nodi nel sottografo: {subgraph.number_of_nodes()}")
print(f"- Numero di archi nel sottografo: {subgraph.number_of_edges()}")
print("- Primi 3 archi del sottografo:")
for i, edge in enumerate(subgraph.edges(data=True)):
    if i >= 3:
        break
    print(f"  {edge}")

# 6. Esempio di classificazione relazioni
print("\n6. Classificazione delle relazioni:")
sample_relation_id = countries_dataset.training_triples[0][1]  # Prendo la prima relazione della prima tripla
print(f"- ID relazione: {sample_relation_id}")
print(f"- Nome relazione: {countries_dataset.id_to_relation[sample_relation_id]}")
print(f"- Tipo relazione: {countries_dataset.relation_to_type.get(sample_relation_id, 'non classificata')}")

