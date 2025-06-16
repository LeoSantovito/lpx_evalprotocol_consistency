# Tesi

**Lavoro di tirocinio interno svolto per la tesi**

Di seguito una descrizione di come ottenere spiegazioni usando CrossE/SemanticCrossE e di come valutare tali spiegazioni secondo il protocollo di Kelpie.
La struttura del progetto rispetta quella di LP-DIXIT e i file di embedding ( \<modello>\_\<dataset>.pt) di predizioni (\<modello>\_\<dataset>.csv) e di configurazione (nelle cartelle configs e lp_configs) sono stati presi dai risultati già disponibili, ottenuti con LP-DIXIT.

## 1 - Creazione file di mapping e applicazione filtro al dataset

Con il comando:

```bash
python -m src.crossE_explanation.generate_files.map_and_filter <dataset>
```

il dataset viene filtrato in modo che le triple di test contenti entità e/o relazioni mai osservate tra le triple di train vengano rimosse.
I nuovi file `train.txt`, `test.txt` e `valid.txt` (filtrati) sovrascrivono gli originali, che però vengono prima salvati nella cartella `original_files`.

---

## 2 - Creazione file ausiliari

A questo punto si può procedere a generare i file necessari al funzionamento del modulo di spiegazione di CrossE/SemanticCrossE: tali file venivano generati in fase di embedding e LP, ma non sono previsti in LP-DIXIT, è necessario quindi ottenerli a partire dalle informazioni disponibili, per una corretta integrazione:

```bash
python -m src.crossE_explanation.generate_files.generate_dicts \
    --dataset <dataset> \
    --model <modello>
```

A partire dal file `train.txt` (contenente le triple di train in forma soggetto-verbo-oggetto, con i nomi di entità e relazioni), vengono creati i file `train_hr_t.pkl` e `train_tr_h.pkl`, contenenti rispettivamente:

* un dizionario che mappa ogni coppia testa-relazione a tutte le code osservate tra le triple di train;
* un dizionario che mappa a tutte le coppie coda-relazione le teste associate.

Successivamente:

```bash
python -m src.crossE_explanation.generate_files.generate_pickles \
    --dataset <dataset> \
    --model <modello>
```

A partire da `test.txt` (triple di test, stessa forma di `train.txt`) vengono ottenuti:

* `test_triples.pkl`, contenente appunto le triple di test;
* `test_predicted_tails.pkl`, contenente le code predette per le triple di test.

Per poter utilizzare la **distanza semantica** nel calcolo delle similarità, è necessaria la generazione di altri file per la corretta integrazione del modulo di spiegazione di CrossE/SemanticCrossE in LP-DIXIT:

```bash
python -m src.crossE_explanation.generate_files.semantic_data <dataset>
```

Sulla base di due file: `entities.csv` e `relations.csv`, contenenti rispettivamente una tabella in cui ogni entità è associata alle classi di cui fa parte, e una tabella in cui a ogni relazione sono associati domini e range; viene creato il file `class2id.txt` che contiene tutti i nomi delle classi osservate (una sola volta), associate a un ID numerico.

A questo punto, con:

```bash
python -m src.crossE_explanation.generate_files.ran_dom_dict --dataset <dataset>
python -m src.crossE_explanation.generate_files.classes_dict --dataset <dataset>
```

sono stati ottenuti i dizionari:

* `rs_domain2id_dict.pkl` (mappa relazioni ai domini, tutto rappresentato con ID),
* `rs_range2id.pkl` (mappa relazioni ai range, tutto rappresentato con ID),
* `entity2class_dict` (mappa entità alle classi a cui appartengono, sempre ID).

Tali file possono essere ottenuti a partire dai file di mapping di entità, relazioni e classi e dai CSV `entities.csv` e `relations.csv`.

---

## 3 - Calcolo similarità

Una volta generati tutti i file, è possibile procedere con il processo di generazione delle spiegazioni. Prima però bisogna calcolare la similarità tra entità e relazioni, attraverso il comando:

```bash
python -m src.crossE_explanation.batch_similarity \
    --data kge_models/<modello>_<dataset>.pt \
    --save_dir pickles/<modello>_<dataset>/<distanza>/ \
    --distance <distanza> \
    --dataset <dataset> \
    --complex   # da aggiungere in caso di embedding complessi
```

È importante che la distanza specificata in `--distance` e quella nel path `--save_dir` sia la stessa, e che non vengano modificate parti che non siano tra `< >`.
La distanza può essere:

* `cosine`
* `euclidian`
* `semantic`

Nel caso di modelli come `ComplEx`, in cui gli embedding sono di tipo complesso ma rappresentati come reali (parte reale e parte immaginaria concatenate), è necessario aggiungere `--complex`.

In questo modo vengono salvati nella directory specificata in `--save_dir` i file:

* `sim_ent.pkl`
* `sim_rel.pkl`
* `sim_inv_rek.pkl`

contenenti le similarità.

---

## 4 - Generazione spiegazioni con CrossE/SemanticCrossE

A questo punto si dispone di quanto necessario per avviare la generazione di spiegazioni con il comando:

```bash
python -m src.crossE_explanation.explanation \
    --data pickles/<modello>_<dataset>/ \
    --predictions_perc 100 \
    --distance <distanza> \
    --pretty_print True \
    --model <modello> \
    --dataset <dataset>
```

Viene così creato un file `.json` contenente le predizioni per cui è stata trovata spiegazione, con spiegazione annessa.
Viene anche fatta una valutazione delle prestazioni in termini di **recall** e **supporto medio** per ogni tipologia di spiegazione (protocollo di valutazione di CrossE/SemanticCrossE).

Lo script prende in input i file generati in precedenza; il file di predizioni e le configurazioni per LP e spiegazione sono quelle già disponibili in LP-DIXIT.

---

## 5 - Valutazione secondo il protocollo di Kelpie

Per avviare il retraining con rimozione delle triple di spiegazione e salvare i risultati bisogna lanciare il comando:

```bash
python -m src.evaluation.re-training \
    --dataset <dataset> \
    --model <modello> \
    --method kelpie \
    --mode necessary
```

Successivamente, con:

```bash
python -m src.evaluation.re-training_metrics \
    --dataset <dataset> \
    --model <modello> \
    --method kelpie \
    --mode necessary
```

viene preso in input il file con i risultati della valutazione, e calcolate e salvate le metriche **ΔMRR** e **ΔHit\@1**.

---

