import argparse
from pathlib import Path
import pickle
import torch
import time
from global_logger import Log
from typing import Optional, Dict, List
import numpy as np


def to_complex(embedding):
    dim = embedding.size(1) // 2
    real = embedding[:, :dim]
    imag = embedding[:, dim:]
    return torch.complex(real, imag)


def prepare_semantic_matrices(classes_dict, num_objects, device='cuda'):
    """Convert class/domain/range dictionaries to binary matrices."""
    if not classes_dict:
        return None

    all_classes = set()
    for class_list in classes_dict.values():
        all_classes.update(class_list)
    if not all_classes:
        return None

    num_classes = len(all_classes)
    class_to_idx = {cls: idx for idx, cls in enumerate(all_classes)}

    mat = torch.zeros((num_objects, num_classes), device=device)
    for obj_id, class_list in classes_dict.items():
        for cls in class_list:
            mat[obj_id, class_to_idx[cls]] = 1
    return mat


def batch_jaccard(matrix, vector_batch):
    """Vectorized Jaccard similarity calculation."""
    intersection = torch.mm(vector_batch, matrix.t())
    sum_vector = vector_batch.sum(dim=1, keepdim=True)
    sum_matrix = matrix.sum(dim=1)
    union = sum_vector + sum_matrix - intersection
    union[union == 0] = 1e-8
    return intersection / union


def load_data(model_path: str):
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

    class_matrix, domain_matrix, range_matrix = None, None, None

    if args.distance_type == 'semantic':
        with open(f"{args.semantic_dir}entity2class_dict.pkl", 'rb') as f:
            entity2class_dict = pickle.load(f)
        with open(f"{args.semantic_dir}rs_domain2id_dict.pkl", 'rb') as f:
            rs_domain2id_dict = pickle.load(f)
        with open(f"{args.semantic_dir}rs_range2id_dict.pkl", 'rb') as f:
            rs_range2id_dict = pickle.load(f)

        class_matrix = prepare_semantic_matrices(entity2class_dict, len(entity_emb))
        domain_matrix = prepare_semantic_matrices(rs_domain2id_dict, len(rel_emb))
        range_matrix = prepare_semantic_matrices(rs_range2id_dict, len(rel_emb))

        return entity_emb, rel_emb, inv_rel_emb, class_matrix, domain_matrix, range_matrix
    else:
        return entity_emb, rel_emb, inv_rel_emb, None, None, None


def compute_semantic_sim_gpu(embeddings, obj_type, class_matrix, domain_matrix, range_matrix, log, top_n=15):
    """Optimized semantic similarity calculation using vectorized operations."""
    sim_dict = {}
    n = embeddings.size(0)
    device = embeddings.device

    # Pre-compute norms for cosine similarity
    norms = torch.norm(embeddings, dim=1, keepdim=True)
    eps = torch.tensor(1e-8, device=device)

    for i in range(n):
        if i % 1000 == 0:
            log.info(f"Processing {i}/{n} entities...")

        emb_i = embeddings[i]
        if torch.is_complex(embeddings):
            cos_sim = torch.real(torch.mv(embeddings, emb_i.conj()))
        else:
            cos_sim = torch.mv(embeddings, emb_i)

        norm_i = torch.norm(emb_i)
        cos_sim = cos_sim / (norms.squeeze() * norm_i + eps)
        cos_sim[i] = -float('inf')

        if obj_type == 'ent' and class_matrix is not None:
            jaccard = batch_jaccard(class_matrix, class_matrix[i].unsqueeze(0)).squeeze(0)
            sem_sim = 0.2 * jaccard
        elif obj_type == 'rel' and domain_matrix is not None and range_matrix is not None:
            jaccard_dom = batch_jaccard(domain_matrix, domain_matrix[i].unsqueeze(0)).squeeze(0)
            jaccard_rng = batch_jaccard(range_matrix, range_matrix[i].unsqueeze(0)).squeeze(0)
            sem_sim = 0.2 * (jaccard_dom + jaccard_rng) / 2
        else:
            sem_sim = 0.0

        combined = 0.8 * cos_sim + sem_sim
        topk = torch.topk(combined, k=top_n)
        sim_dict[i] = topk.indices.tolist()

        del cos_sim, combined, topk
        torch.cuda.empty_cache()

    return sim_dict


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

    ent, rel, inv, class_matrix, domain_matrix, range_matrix = load_data(args.data_dir)

    log.info(f"Entity embedding type: {ent.dtype}, shape: {ent.shape}")
    log.info(f"Relation embedding type: {rel.dtype}, shape: {rel.shape}")

    log.info("Computing similarity...")
    if args.distance_type == 'semantic':
        sim_ent = compute_semantic_sim_gpu(ent, 'ent', class_matrix, None, None, log)
        sim_rel = compute_semantic_sim_gpu(rel, 'rel', None, domain_matrix, range_matrix, log)
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