import argparse
from pathlib import Path
import pickle
import torch
import time
from global_logger import Log


def to_complex(embedding):
    dim = embedding.size(1) // 2
    real = embedding[:, :dim]
    imag = embedding[:, dim:]
    return torch.complex(real, imag)


def load_data(model_path: str, entity2id_path: str, relation2id_path: str):
    device = torch.device("cuda")
    model = torch.load(model_path, map_location=device)

    num_relations = model['relation_embeddings'].shape[0]
    split_point = num_relations // 2

    entity_emb = model['entity_embeddings'].to(device)
    rel_emb = model['relation_embeddings'][:split_point].to(device)
    inv_rel_emb = model['relation_embeddings'][split_point:].to(device)

    if args.use_complex:
        entity_emb = to_complex(entity_emb)
        rel_emb = to_complex(rel_emb)
        inv_rel_emb = to_complex(inv_rel_emb)

    entity2id = {line.split('\t')[0]: int(line.split('\t')[1])
                 for line in open(entity2id_path)}
    relation2id = {line.split('\t')[0]: int(line.split('\t')[1])
                   for line in open(relation2id_path)}

    entity2class_dict, rs_domain2id_dict, rs_range2id_dict = None, None, None
    if args.distance_type == 'semantic':
        with open(f"{args.semantic_dir}entity2class_dict.pkl", 'rb') as f:
            entity2class_dict = pickle.load(f)
        with open(f"{args.semantic_dir}rs_domain2id_dict.pkl", 'rb') as f:
            rs_domain2id_dict = pickle.load(f)
        with open(f"{args.semantic_dir}rs_range2id_dict.pkl", 'rb') as f:
            rs_range2id_dict = pickle.load(f)

    return entity_emb, rel_emb, inv_rel_emb, entity2class_dict, rs_domain2id_dict, rs_range2id_dict


def cosine_similarity_gpu(matrix, vector):
    if torch.is_complex(matrix):

        dot = torch.sum(matrix * vector.conj(), dim=1)
        norm_matrix = torch.norm(matrix, dim=1)
        norm_vector = torch.norm(vector)
        similarity = dot.real / (norm_matrix * norm_vector + 1e-8)
    else:
        similarity = torch.nn.functional.cosine_similarity(matrix, vector.unsqueeze(0), dim=1)
    return similarity


def euclidean_distance_gpu(matrix, vector):
    if torch.is_complex(matrix):
        diff = matrix - vector.unsqueeze(0)
        dists = torch.norm(diff.abs(), dim=1)
    else:
        dists = torch.norm(matrix - vector.unsqueeze(0), dim=1)
    return dists


def top_sim_indices(embeddings, emb_id, distance_type, top_n):
    emb = embeddings[emb_id]
    if distance_type == 'cosine':
        scores = cosine_similarity_gpu(embeddings, emb)
        scores[emb_id] = -1
        topk = torch.topk(scores, k=top_n)
    elif distance_type == 'euclidian':
        dists = euclidean_distance_gpu(embeddings, emb)
        dists[emb_id] = float('inf')
        topk = torch.topk(-dists, k=top_n)
    else:
        raise ValueError("Semantic similarity must be computed separately.")
    return topk.indices.tolist()


def compute_sim_dictionary_gpu(embeddings, distance_type, obj_type, top_n=15):
    sim_dict = {}
    for emb_id in range(embeddings.size(0)):
        sim_dict[emb_id] = top_sim_indices(embeddings, emb_id, distance_type, top_n)
    return sim_dict


def __jaccard(set1, set2):
    if not set1 or not set2:
        return 0
    return len(set1.intersection(set2)) / len(set1.union(set2))


def __semantic(emb_a, emb_b, id, other_id, obj_type, classes, domains, ranges, log):
    log = Log.get_logger(name="general")

    if obj_type == 'ent':
        ent_sim = __jaccard(set(classes.get(id, [])), set(classes.get(other_id, [])))
        if ent_sim > 0:
            log.debug(f"{id} ({classes.get(id, [])}),{other_id} ({classes.get(other_id, [])}), {ent_sim}")
        return (0.2 * ent_sim) + (cosine_similarity_gpu(emb_b.unsqueeze(0), emb_a).item() * 0.8)

    elif obj_type == 'rel':
        dom_jaccard = __jaccard(set(domains.get(id, [])), set(domains.get(other_id, [])))
        range_jaccard = __jaccard(set(ranges.get(id, [])), set(ranges.get(other_id, [])))
        rel_sim = dom_jaccard + range_jaccard
        if rel_sim > 0:
            log.debug(f"{id} (domain: {domains.get(id, [])}) (range: {ranges.get(id, [])}),"
                      f"{other_id} (domain: {domains.get(other_id, [])}) (range: {ranges.get(other_id, [])}), {rel_sim}")
        return (0.2 * rel_sim) + (cosine_similarity_gpu(emb_b.unsqueeze(0), emb_a).item() * 0.8)
    else:
        raise ValueError(f"obj_type can be either 'ent' or 'rel', not '{obj_type}'")


def compute_semantic_sim(embeddings, obj_type, classes, domains, ranges, log, top_n=15):
    sim_dict = {}
    for i in range(embeddings.size(0)):
        emb_i = embeddings[i]
        scores = {}
        for j in range(embeddings.size(0)):
            if i == j:
                continue
            sim = __semantic(emb_i, embeddings[j], i, j, obj_type, classes, domains, ranges, log)
            scores[j] = sim
        sorted_ids = sorted(scores, key=scores.get, reverse=True)
        sim_dict[i] = sorted_ids[:top_n]
    return sim_dict


def save_data(sim_dict, save_path, filename):
    Path(save_path).mkdir(parents=True, exist_ok=True)
    with open(f"{save_path}/{filename}", 'wb') as f:
        pickle.dump(sim_dict, f)


if __name__ == '__main__':
    start_time = time.time()
    parser = argparse.ArgumentParser(description='Compute similarity using GPU.')
    parser.add_argument('--data', dest='data_dir', type=str)
    parser.add_argument('--save_dir', dest='save_dir', type=str, default='data_dir')
    parser.add_argument('--distance', dest='distance_type', type=str, default='euclidian')
    parser.add_argument('--semantic_data', dest='semantic_dir', type=str, default=None)
    parser.add_argument('--dataset', dest='dataset_name', type=str, required=True)
    parser.add_argument('--complex', dest='use_complex', action='store_true',
                        help='Se specificato, trasforma embedding concatenati in tensori complessi')

    global args
    args = parser.parse_args()

    if args.semantic_dir and args.distance_type != 'semantic':
        args.distance_type = 'semantic'
    if args.save_dir == 'data_dir':
        args.save_dir = f"{args.data_dir}/{args.distance_type}/"
    if args.distance_type == 'semantic' and not args.semantic_dir:
        print("Semantic mode selected but semantic data not provided. EXIT")
        exit()

    log = Log.get_logger(logs_dir=args.save_dir, name="general")
    log.info(f"Distance type: {args.distance_type}")
    log.info(f"Save folder: {args.save_dir}")
    log.info(f"Using complex embeddings: {args.use_complex}")

    ent2id_path = f"./data/{args.dataset_name}/entity2id.txt"
    rel2id_path = f"./data/{args.dataset_name}/relation2id.txt"

    ent, rel, inv, classes, domains, ranges = load_data(args.data_dir, ent2id_path, rel2id_path)

    log.info(f"Entity embedding type: {ent.dtype}, shape: {ent.shape}")
    log.info(f"Relation embedding type: {rel.dtype}, shape: {rel.shape}")

    log.info("Computing similarity between entities...")
    if args.distance_type == 'semantic':
        sim_ent = compute_semantic_sim(ent, 'ent', classes, domains, ranges, log)
        sim_rel = compute_semantic_sim(rel, 'rel', classes, domains, ranges, log)
        save_data(sim_ent, args.save_dir, "sim_entities.pkl")
        save_data(sim_rel, args.save_dir, "sim_rel.pkl")
    else:
        sim_ent = compute_sim_dictionary_gpu(ent, args.distance_type, 'ent')
        sim_rel = compute_sim_dictionary_gpu(rel, args.distance_type, 'rel')
        sim_inv = compute_sim_dictionary_gpu(inv, args.distance_type, 'rel')
        save_data(sim_ent, args.save_dir, "sim_entities.pkl")
        save_data(sim_rel, args.save_dir, "sim_rel.pkl")
        save_data(sim_inv, args.save_dir, "sim_inv_rel.pkl")

    elapsed = time.time() - start_time
    log.info(f"Execution time: {elapsed:.2f} seconds")
    print(f"Execution time: {elapsed:.2f} seconds")
