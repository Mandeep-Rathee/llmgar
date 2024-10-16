from typing import Optional
import numpy as np
from collections import Counter
import pyterrier as pt
import pandas as pd
import ir_datasets
import time
import gc
from collections import deque



logger = ir_datasets.log.easy()

import json

import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class RankGAR(pt.Transformer):
    """
    A transformer that implements the Graph-based Adaptive Re-ranker algorithm from
    MacAvaney et al. "Adaptive Re-Ranking with a Corpus Graph" CIKM 2022.

    Required input columns: ['qid', 'query', 'docno', 'score', 'rank']
    Output columns: ['qid', 'query', 'docno', 'score', 'rank', 'iteration']
    where iteration defines the batch number which identified the document. Specifically
    even=initial retrieval   odd=corpus graph    -1=backfilled
    
    """
    def __init__(self,
        scorer: pt.Transformer,
        corpus_graph: 'CorpusGraph',
        num_results: int = 1000,
        top_k_docs: int=10,
        buffer_size : int = 20,
        batch_size: Optional[int] = None,
        top_down: bool = False,
        merge_rank: bool = False,
        backfill: bool = True,
        enabled: bool = True,
        verbose: bool = False):
        """
            GAR init method
            Args:
                scorer(pyterrier.Transformer): A transformer that scores query-document pairs. It will only be provided with ['qid, 'query', 'docno', 'score'].
                corpus_graph(pyterrier_adaptive.CorpusGraph): A graph of the corpus, enabling quick lookups of nearest neighbours
                num_results(int): The maximum number of documents to score (called "budget" and $c$ in the paper)
                batch_size(int): The number of documents to score at once (called $b$ in the paper). If not provided, will attempt to use the batch size from the scorer
                backfill(bool): If True, always include all documents from the initial stage, even if they were not re-scored
                enabled(bool): If False, perform re-ranking without using the corpus graph
                verbose(bool): If True, print progress information
        """
        self.scorer = scorer
        self.corpus_graph = corpus_graph
        self.num_results = num_results
        self.top_k_docs = top_k_docs
        self.buffer_size = buffer_size    
        if batch_size is None:
            batch_size = scorer.batch_size if hasattr(scorer, 'batch_size') else 16
        self.batch_size = batch_size
        self.merge_rank = merge_rank
        self.top_down = top_down
        self.backfill = backfill
        self.enabled = enabled
        self.verbose = verbose

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Applies Graph-based Adaptive Re-ranking to the provided dataframe. Essentially,
        Algorithm 1 from the paper.
        """
        result = {'qid': [], 'query': [], 'docno': [], 'rank': [], 'score': [], 'iteration': []}

        df = dict(iter(df.groupby(by=['qid'])))
        qids = df.keys()
        if self.verbose:
            qids = logger.pbar(qids, desc='adaptive re-ranking', unit='query')

        score_time = []
        total_time = []
        print("Starting GAR")
        for qid in qids:

            score_time_per_q = 0
            query = df[qid]['query'].iloc[0]

            scores = {}
            res_map = [Counter(dict(zip(df[qid].docno, df[qid].score)))] # initial results
            if self.enabled:
                res_map.append(Counter()) # frontier
            frontier_data = {'minscore': float('inf')}

            r1_upto_now = {}
            selected_docs = []
            top_10 = []

            query_start_time = time.perf_counter()
            iteration = 0
            while len(scores) < self.num_results and any(r for r in res_map):
                if len(res_map[iteration%len(res_map)]) == 0:
                    # if there's nothing available for the one we select, skip this iteration (i.e., move on to the next one)
                    iteration += 1
                    continue
                this_res = res_map[iteration%len(res_map)] # alternate between the initial ranking and frontier
                if iteration == 0:
                    size = self.buffer_size
                else:    
                    size = min(self.batch_size, self.num_results - len(scores)-self.top_k_docs) # get either the batch size or remaining budget (whichever is smaller)


                last_iteration = False
                if size == self.num_results - len(scores) - self.top_k_docs and len(this_res)>=size:
                    last_iteration = True
                    """
                    s = 50 - 0 - 10 = 40 , iteration = 0,  
                    s = 50 - 10 - 10 = 30
                    s = 50 - 20 - 10 = 20
                    s = 50 - 30 - 10 = 10
                    last_iteration = True
                    """

                # build batch of documents to score in this round
                batch = this_res.most_common(size)
                # get the document ids of the batch
                batch_docs = [doc for doc, _ in batch]

                if self.merge_rank:
                    # go score the batch of document with the re-ranker
                    if len(selected_docs)==0:                                 # iteration 0
                        batch = pd.DataFrame(batch_docs, columns=['docno'])
                        batch['qid'] = qid
                        batch['query'] = query
                        batch = self.scorer(batch)
                        selected_docs = batch.docno.tolist()
                    else:                
                        selected_docs =  merge_and_rank(selected_docs, batch_docs, self.buffer_size, self.scorer, query, qid)  # iteration > 0
                        #assert len(selected_docs)==(iteration+1)*self.batch_size or len(selected_docs)==self.num_results, f"frontier size={len(res_map[1])} len(selected_docs)={len(selected_docs)} iteration={iteration} batch_size={self.batch_size} num_results={self.num_results}"
                    psuedo_rank = np.arange(len(selected_docs))
                    psuedo_scores = [1/(r+1) for r in psuedo_rank]
                    scores = {k: (s, iteration) for k, s in zip(selected_docs, psuedo_scores)}
                    r1_upto_now = {k: s for k, s in zip(selected_docs, psuedo_scores)}

                elif self.top_down:
                    buffer =  top_10 + batch_docs
                    batch  = pd.DataFrame(buffer, columns=['docno'])
                    batch['qid'] = qid
                    batch['query'] = query

                    """time taken to score the batch"""
                    score_start = time.perf_counter()
                    batch = self.scorer(batch)
                    score_end = time.perf_counter()
                    score_time_per_q += (score_end - score_start) * 1000
                    ranked_buffer = batch.docno.tolist()
                                            
                    top_10  = ranked_buffer[:self.top_k_docs]
                    not_top_10 = ranked_buffer[self.top_k_docs:]

                    if last_iteration==True or len(selected_docs)+len(ranked_buffer)>=self.num_results:
                        selected_docs = ranked_buffer + selected_docs
                        psuedo_rank = np.arange(len(selected_docs))
                        psuedo_scores = [1/(r+1) for r in psuedo_rank]
                        scores = {k: (s, iteration) for k, s in zip(selected_docs, psuedo_scores)}

                    else:
                        selected_docs = not_top_10 + selected_docs
                        scores = {k: (iteration) for k in selected_docs}

                    
                    psuedo_rank = np.arange(len(top_10))
                    psuedo_scores = [1/(r+1) for r in psuedo_rank]
                    r1_upto_now = {k: s for k, s in zip(top_10, psuedo_scores)}
                    S = r1_upto_now

                    # psuedo_rank = np.arange(len(ranked_buffer))
                    # psuedo_scores = [1/(r+1) for r in psuedo_rank]
                    # r1_upto_now = {k: s for k, s in zip(ranked_buffer, psuedo_scores)}
                    # S = r1_upto_now


                else:
                    selected_docs.extend(batch_docs)
                    batch  = pd.DataFrame(selected_docs, columns=['docno'])
                    batch['qid'] = qid
                    batch['query'] = query
                    batch = self.scorer(batch)
                    scores = {k: (s, iteration) for k, s in zip(batch.docno, batch.score)}
                    r1_upto_now = {k: s for k, s in zip(batch.docno, batch.score)}    
                    S =  dict(Counter(r1_upto_now).most_common(self.top_k_docs))       # Take top s(hyper-parameter) documents from R1


                self._drop_docnos_from_counters(batch_docs, res_map)
                """
                if len(scores) = 40, STOP
                """

                if len(scores) < self.num_results and self.enabled:
                        res_map[1] = self._update_frontier(S, selected_docs)
                iteration += 1



            if last_iteration==False:
                print("Last iteration is False")
                selected_docs = top_10 + selected_docs
                psuedo_rank = np.arange(len(selected_docs))
                psuedo_scores = [1/(r+1) for r in psuedo_rank]
                scores = {k: (s, iteration) for k, s in zip(selected_docs, psuedo_scores)}
            
            
            assert len(scores) <= self.num_results, f"len(scores)={len(scores)} num_results={self.num_results}"


            score_time.append(score_time_per_q)
            query_end_time = time.perf_counter()

            non_scorer_time = (query_end_time - query_start_time) * 1000 - score_time_per_q  # Time excluding scorer
            total_time.append(non_scorer_time)


            # Add scored items to results
            result['qid'].append(np.full(len(scores), qid))
            result['query'].append(np.full(len(scores), query))
            result['rank'].append(np.arange(len(scores)))
            for did, (score, i) in Counter(scores).most_common():
                result['docno'].append(did)
                result['score'].append(score)
                result['iteration'].append(i)

            # Backfill unscored items
            if self.backfill and len(scores) < self.num_results:
                last_score = result['score'][-1] if result['score'] else 0.
                count = min(self.num_results - len(scores), len(res_map[0]))
                result['qid'].append(np.full(count, qid))
                result['query'].append(np.full(count, query))
                result['rank'].append(np.arange(len(scores), len(scores) + count))
                for i, (did, score) in enumerate(res_map[0].most_common()):
                    if i >= count:
                        break
                    result['docno'].append(did)
                    result['score'].append(last_score - 1 - i)
                    result['iteration'].append(-1)


        print("Mean Time taken by scorer per query:", np.mean(score_time), "milliseconds")
        print("Mean time taken by SlideGAR per query:", np.mean(total_time), "milliseconds")

        return pd.DataFrame({
            'qid': np.concatenate(result['qid']),
            'query': np.concatenate(result['query']),
            'docno': result['docno'],
            'rank': np.concatenate(result['rank']),
            'score': result['score'],
            'iteration': result['iteration'],
        })

    def _update_frontier(self, S, selected_docs):

        selected_docs = set(selected_docs)
        if self.top_down:  
            selected_docs.update(set(S.keys()))
            
        frontier = Counter()
        for did, score in S.items():
            target_neighbors = self.corpus_graph.neighbours(did)
            for n in target_neighbors:
                if n not in selected_docs:
                    if n not in frontier or score > frontier[n]:
                        frontier[n] = score

                    if len(frontier) >= self.batch_size:
                        break  

            if len(frontier) >= self.batch_size:
                break  



        return frontier                




    def _drop_docnos_from_counters(self, docnos, counters):
        for docno in docnos:
            for c in counters:
                del c[docno]


def merge_remaining_items(merged_list, buffer, ranked_buffer, remaining_queue):
    # Handle any remaining items in the buffer (no reranking)
    if ranked_buffer:  # ranked_buffer already holds the ranked elements
        merged_list.extend(ranked_buffer)  # Add the remaining ranked items to the merged list
        buffer.clear()  # Clear the buffer since we've processed it
    
    # Append the remaining items from the non-empty queue without reranking
    merged_list.extend(remaining_queue)    


def merge_and_rank(re_ranked_list, candidate_ranked_list, k, listwise_ranker, query, qid):
    merged_list = []  # Final merged list
    buffer = []  # Buffer to hold elements from both lists

    # Step 1: Treat the input lists as queues (FIFO)
    queue_re_ranked = deque(re_ranked_list)
    queue_candidate_ranked = deque(candidate_ranked_list)

    # Step 2: While either queue is not empty or buffer is not empty
    while queue_re_ranked or queue_candidate_ranked:
        # Step 3: Dequeue from the queues into the buffer, up to the buffer size limit k
        while len(buffer) < k and (queue_re_ranked or queue_candidate_ranked):
            if queue_re_ranked:
                buffer.append(queue_re_ranked.popleft())  # Dequeue from re-ranked list and add to the buffer
            if len(buffer) < k and queue_candidate_ranked:
                buffer.append(queue_candidate_ranked.popleft())  # Dequeue from candidate ranked list and add to the buffer

        # Step 4: Rank the buffer using the ListwiseLLMRanker
        buffer_batch = pd.DataFrame(buffer, columns=['docno'])
        buffer_batch['qid'] = qid
        buffer_batch['query'] = query
        ranked_buffer_batch = listwise_ranker(buffer_batch)
        ranked_buffer = ranked_buffer_batch.docno.tolist()

        # Step 5: Remove items from the buffer (most relevant to least) until only elements from one list remain
        while contains_elements_from_both_lists(buffer, re_ranked_list, candidate_ranked_list):
            most_relevant_element = ranked_buffer[0]  # Most relevant element
            merged_list.append(most_relevant_element)
            buffer.remove(most_relevant_element)
            ranked_buffer.remove(most_relevant_element)

        # # Step 6: Check if both the queues is empty and handle the remaining items from the buffer
        if not queue_candidate_ranked and buffer:
            if ranked_buffer:
                merged_list.extend(ranked_buffer)
                buffer.clear()

            # If queue_re_ranked still has items, append them
            if queue_re_ranked:
                merged_list.extend(queue_re_ranked)
            break
  


    return merged_list




# Helper function to check if buffer contains elements from both original lists
def contains_elements_from_both_lists(buffer, re_ranked_list, candidate_ranked_list):
    has_re_ranked = any(item in re_ranked_list for item in buffer)
    has_candidate_ranked = any(item in candidate_ranked_list for item in buffer)
    return has_re_ranked and has_candidate_ranked

