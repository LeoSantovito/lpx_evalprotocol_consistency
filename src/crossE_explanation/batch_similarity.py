"""
Compute in batch the similarity between each entity and the similarity between each relation, in order to use these data
to speed up the explanation process
"""
import argparse
import multiprocessing
from pathlib import Path
import pickle
import numpy as np
from global_logger import Log
from numpy import dot
from numpy.linalg import norm
import torch
import time


class semantic_data:
    """
    Static objects class to host semantic data
    """
    entity2class_dict = {}
    rs_domain2id_dict = {}
    rs_range2id_dict = {}


def load_data(model_path: str, entity2id_path: str, relation2id_path: str):
    """
    Load the data needed for the similarity evaluation; the files needed are model.pt, entity2id.txt, relation2id.txt
    :param model_path: path to the model file
    :param entity2id_path: path to the entity2id file
    :param relation2id_path: path to the relation2id file
    :return: data loaded in the form of list of lists; E.g ent_emb = [[emb1 numbs], [emb2 numbs], ...]
    """
    # Carica il modello
    model = torch.load(model_path)
    entity_emb = model['entity_embeddings'].cpu().numpy()
    # Calcola il punto di divisione
    num_relations = model['relation_embeddings'].shape[0]
    split_point = num_relations // 2
    # Estrai gli embeddings
    rel_emb = model['relation_embeddings'][:split_point].cpu().numpy()  # Prima metà → relazioni normali
    inv_rel_emb = model['relation_embeddings'][split_point:].cpu().numpy()  # Seconda metà → relazioni inverse

    # Caricamento entity2id mapping
    entity2id = {}
    with open(entity2id_path, 'r') as f:
        for line in f:
            entity, id = line.strip().split('\t')
            entity2id[entity] = int(id)

    # Caricamento relation2id mapping
    relation2id = {}
    with open(relation2id_path, 'r') as f:
        for line in f:
            relation, id = line.strip().split('\t')
            relation2id[relation] = int(id)

    entity2class_dict = None
    rs_domain2id_dict = None
    rs_range2id_dict = None
    if args.distance_type == 'semantic':
        # entità e classi
        file_path = f"{args.semantic_dir}entity2class_dict.pkl"
        with open(file_path, 'rb') as f:
            entity2class_dict = pickle.load(f)
        # relazioni e domini
        file_path = f"{args.semantic_dir}rs_domain2id_dict.pkl"
        with open(file_path, 'rb') as f:
            rs_domain2id_dict = pickle.load(f)
        # relazioni e ranges
        file_path = f"{args.semantic_dir}rs_range2id_dict.pkl"
        with open(file_path, 'rb') as f:
            rs_range2id_dict = pickle.load(f)

    return entity_emb, rel_emb, inv_rel_emb, entity2class_dict, rs_domain2id_dict, rs_range2id_dict


def __euclidean(a, b, id_a=None, id_b=None, obj_type=None, classes=None, domains=None, ranges=None):
    """
    Computes euclidian distance between two lists; the other paramas are not used, is only useful to simplify the code for semantic evaluations
    :return: distance
    """
    return np.linalg.norm(a - b)


def __cosine(a, b, id_a=None, id_b=None, obj_type=None, classes=None, domains=None, ranges=None):
    """
    Computes the cosine similarity between a and b; the other paramas are not used, is only useful to simplify the code for semantic evaluations
    :param a: embedding a
    :param b: embedding b
    :return: cosine similarity between two embeddings a and b
    """
    cos_sim = dot(a, b) / (norm(a) * norm(b))
    return cos_sim


def __jaccard(set1, set2):
    """
    Computes jaccard index, given two sets
    :param set1:
    :param set2:
    :return: similarity between the two sets in terms of jaccard index
    """

    if len(set1) == 0 or len(set2) == 0:  # uno dei due ha owl:thing
        return 0
    else:
        numeratore = len(set1.intersection(set2))
        denominatore = len(set1.union(set2))  # se è pari a 0 significa che nessuno dei due ha nulla
        return numeratore / denominatore


def __semantic(emb_a, emb_b, id, other_id, obj_type, classes, domains, ranges):
    """
    Compute a semantic distance based on jaccard index...
    """
    log = Log.get_logger(name="general")
    if obj_type == 'ent':
        # Converti le liste in set
        ent_sim = __jaccard(set(classes[id]), set(classes[other_id]))
        
        if ent_sim > 0:
            log.debug(f"{id} ({classes[id]}),{other_id} ({classes[other_id]}), {ent_sim}")

        return (0.2 * ent_sim) + (__cosine(emb_a, emb_b) * 0.8)

    elif obj_type == 'rel':
        # Converti le liste in set
        dom_jaccard = __jaccard(set(domains[id]), set(domains[other_id]))
        range_jaccard = __jaccard(set(ranges[id]), set(ranges[other_id]))

        rel_sim = dom_jaccard + range_jaccard
        if rel_sim > 0:
            log.debug(f"{id} (domain: {domains[id]}) (range: {ranges[id]}),{other_id} (domain: {domains[other_id]}) (range: {ranges[other_id]}), {rel_sim}")

        return (0.2 * rel_sim) + (__cosine(emb_a, emb_b) * 0.8)
    else:
        raise ValueError(f"obj_type can be either 'ent' or 'rel', not '{obj_type}'")


def __top_sim_emb(emb, emb_id, embedding_matrix, distance_type, obj_type, classes=None, domains=None, ranges=None,
                  first_n=15):
    """
    Compute the distance/similarity between an embedding of the triple (head, relation, or tail)
    and all the other embeddings of the same kind of objects set for wich embeddings are provided; if the mode is semantic,
    the similarity will be a mix between semantic informations and euclidian distances
    :param top_k: entities/relationships most similar to return
    :param emb_id: id of the object of the KG, useful to exlude it from the comparison
    :param emb: relationship of the test triple to compare with the other relationships
    :param embedding_matrix: embeddings of the entities/relationships in the KG
    :param obj_type: either 'rel' or 'ent', useful only for the semantic distance
    :param first_n: number of top similar embeddings ids to keep, it helps to reduce the size required to store files; also
    the explnation experiments does not require all the ids, but only the first (at most 15) will be used
    :return: list of ids of the top_k most similar objects to emb
    """
    if distance_type == 'euclidian':
        distance_function = __euclidean
    elif distance_type == 'cosine':
        distance_function = __cosine
    elif distance_type == 'semantic':
        distance_function = __semantic

    distances = {}  # dizionario {id: distanza dall'emb, id: distanza,...}
    for i in range(0, len(embedding_matrix)):
        if i != emb_id:
            other_rel = embedding_matrix[i]
            dst = distance_function(other_rel, emb, emb_id, i, obj_type, classes, domains, ranges)

            distances[i] = dst

    if distance_type == 'cosine' or distance_type == 'semantic':
        sorted_dict = {k: v for k, v in sorted(distances.items(), key=lambda item: item[1], reverse=True)}  # descending
    else:  # euclidean
        sorted_dict = {k: v for k, v in sorted(distances.items(), key=lambda item: item[1])}  # ascending
    ids = list(sorted_dict.keys())
    return ids[:first_n]


def compute_sim_dictionary(embeddings: list, dict_multiproc, proc_name, distance_type, obj_type, classes=None,
                           domains=None, ranges=None):
    """
    Compute the similarity between each element in the embedding list
    :param obj_type: either 'rel' or 'ent', useful only for the semantic distance
    :param embeddings: embeddings list in the form [[emb1], [emb2], ...]
    :return: dictionary in the form {emb_id: [sim_emb_id1, sim_emb_id2]} ranked by similarity
    """
    similarity_dictionary = {}
    for emb_id in range(0, len(embeddings)):
        similarity_dictionary[emb_id] = __top_sim_emb(embeddings[emb_id], emb_id, embeddings, distance_type, obj_type,
                                                      classes, domains, ranges)
    dict_multiproc[proc_name] = similarity_dictionary


def save_data(sim_dict, save_path, filename):
    """
    Saves the dictionary containing emb ids and the lists of similar embeddings ids
    :param sim_dict:
    :param save_path:
    :return:
    """
    Path(save_path).mkdir(parents=True, exist_ok=True)
    file_path = f"{save_path}{filename}"
    with open(file_path, 'wb') as f:
        pickle.dump(sim_dict, f)


if __name__ == '__main__':
    start_time = time.time()
    parser = argparse.ArgumentParser(description='Batch similarity.')
    parser.add_argument('--data', dest='data_dir', type=str,
                        help="Data folder containing the output of the training phase (pickle_files")

    parser.add_argument('--save_dir', dest='save_dir', type=str,
                        help='directory to save in the similarity data; default will be set to the same folder the data'
                             'are loaded from.',
                        default='data_dir')
    parser.add_argument('--distance', dest='distance_type', type=str,
                        help='choose the distance to compute between entities, possible choises: euclidian, cosine, semantic',
                        default='euclidian')
    parser.add_argument('--semantic_data', dest='semantic_dir', type=str,
                        help="Data folder containing the dictionaries about relation domains and ranges, and classes"
                             "for the entities; tipically it's the folder in which the raw dataset files are stored."
                             "By default it is None, because by default the similarity function is the euclidean distance.",
                        default=None)
    parser.add_argument('--multiprocessing', dest='multiproc_flag', type=bool,
                        help='enables multiprocessing, by default is not enabled',
                        default=False)
    parser.add_argument('--dataset', dest='dataset_name', type=str,
                        required=True, help='Name of the dataset to process')

    global args
    args = parser.parse_args()

    data_folder = args.data_dir

    if args.distance_type != "euclidian" and args.distance_type != "cosine" and args.distance_type != "semantic":
        print(f"Distance type: {args.distance_type} is not a valid value")
        exit()

    # più comodo per specificare gli args
    if args.semantic_dir != None:
        args.distance_type = 'semantic'

    save_dir = args.save_dir
    if save_dir == 'data_dir':
        save_dir = f"{data_folder}{args.distance_type}/"

    args.save_dir = save_dir

    if args.distance_type == 'semantic' and args.semantic_dir is None:
        print("You had to provide a folder (--semantic_data) in which there are three files: entity2class_dict.pkl, "
              "rs_domain2id_dict.pkl, rs_range2id_dict.pkl. \nEXIT")
        exit()
    log = Log.get_logger(logs_dir=save_dir, name="general")  # logger for general communications
    log.info(f"Distance type: {args.distance_type}")
    log.info(f"Save folder: {save_dir}")

    # logger for semantic experiments
    """if args.distance_type == 'semantic':
        # Path(save_dir).mkdir(parents=True, exist_ok=True)
        ent_log = Log.get_logger(logs_dir=save_dir, name="ent",
                                 level=Log.Levels.DEBUG)  # to store semantic similarities
        rel_log = Log.get_logger(logs_dir=save_dir, name="rel", level=Log.Levels.DEBUG)
        ent_log.debug("ENTITIES WITH SEMANTIC SIMILARITY")
        rel_log.debug("RELATIONSHIPS WITH SEMANTIC SIMILARITY")"""

    ent2id_path = f"./data/{args.dataset_name}/entity2id.txt"
    rel2id_path = f"./data/{args.dataset_name}/relation2id.txt"
    ent, rel, inv, classes, domains, ranges = load_data(
        data_folder, ent2id_path, rel2id_path)  # classe, domains, ranges will be None if not semantic mode
    semantic_data.entity2class_dict = classes
    if args.multiproc_flag:
        manager1 = multiprocessing.Manager()
        return_dict = manager1.dict()

        processes_list = []
        log.info("Computing similarity between entities")
        p1 = multiprocessing.Process(target=compute_sim_dictionary,
                                     args=(ent, return_dict, "ent", args.distance_type, 'ent', classes, domains, ranges))
        processes_list.append(p1)
        p1.start()
        log.info("Computing similarity between relationships")
        p2 = multiprocessing.Process(target=compute_sim_dictionary,
                                     args=(rel, return_dict, "rel", args.distance_type, 'rel', classes, domains, ranges))
        processes_list.append(p2)
        p2.start()
        if args.distance_type != 'semantic':
            log.info("Computing similarity between inverse relationships")
            p3 = multiprocessing.Process(target=compute_sim_dictionary,
                                         args=(inv, return_dict, "inv", args.distance_type, 'rel'))
            processes_list.append(p3)
            p3.start()

        for proc in processes_list:
            proc.join()

    else:
        return_dict = {}
        log.info("Computing similarity between entities")
        compute_sim_dictionary(ent, return_dict, "ent", args.distance_type, 'ent', classes, domains, ranges)

        log.info("Computing similarity between relationships")
        compute_sim_dictionary(rel, return_dict, "rel", args.distance_type, 'rel', classes, domains, ranges)

        if args.distance_type != 'semantic':
            log.info("Computing similarity between inverse relationships")
            compute_sim_dictionary(inv, return_dict, "inv", args.distance_type, 'rel')


    sim_ent = return_dict['ent']
    sim_rel = return_dict['rel']
    if args.distance_type != 'semantic':
        sim_inv_rel = return_dict['inv']
    log.info("Data computed, start saving...")
    save_data(sim_ent, save_path=f"{save_dir}/", filename="sim_entities.pkl")
    save_data(sim_rel, save_path=f"{save_dir}/", filename="sim_rel.pkl")
    if args.distance_type != 'semantic':
        save_data(sim_inv_rel, save_path=f"{save_dir}/", filename="sim_inv_rel.pkl")
    log.info(f"All data  stored in {save_dir}")

    print()

    execution_time = time.time() - start_time  # Calcola il tempo di esecuzione
    log.info(f"Execution time: {execution_time:.2f} seconds")  # Stampalo nel log
    print(f"Execution time: {execution_time:.2f} seconds")  # Stampalo anche nella console