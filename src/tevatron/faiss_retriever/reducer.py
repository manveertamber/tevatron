import glob
import faiss
from argparse import ArgumentParser
from tqdm import tqdm
from typing import Iterable, Tuple
from numpy import ndarray
import numpy as np
from .__main__ import pickle_load, write_ranking


def combine_faiss_results(results: Iterable[Tuple[ndarray, ndarray]]):
    counter = 0
    rh = None
    doc_id_dict = {}
    doc_id_int = 1
    for scores, indices in results:
        int_indices = []
        for index in indices:
            new_index = []
            for elem in index:
                doc_id_dict[doc_id_int] = elem
                new_index.append(doc_id_int)
                doc_id_int+=1
            int_indices.append(new_index)
        indices = np.array(int_indices, dtype='int64')

        if rh is None:
            print(f'Initializing Heap. Assuming {scores.shape[0]} queries.')
            rh = faiss.ResultHeap(scores.shape[0], scores.shape[1])
        rh.add_result(-scores, indices)    
    rh.finalize()
    corpus_scores, corpus_indices = -rh.D, rh.I
    indices = []
    for index in corpus_indices:
        new_index = []
        for elem in index:
            elem = doc_id_dict[elem]
            new_index.append(elem)
        indices.append(new_index)

    corpus_indices = np.array(indices)
    return corpus_scores, corpus_indices


def main():
    parser = ArgumentParser()
    parser.add_argument('--score_dir', required=True)
    parser.add_argument('--query', required=True)
    parser.add_argument('--save_ranking_to', required=True)
    args = parser.parse_args()

    partitions = glob.glob(f'{args.score_dir}/*')

    corpus_scores, corpus_indices = combine_faiss_results(map(pickle_load, tqdm(partitions)))

    _, q_lookup = pickle_load(args.query)
    write_ranking(corpus_indices, corpus_scores, q_lookup, args.save_ranking_to)


if __name__ == '__main__':
    main()
