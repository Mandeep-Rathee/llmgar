from typing import Optional
import numpy as np
from collections import Counter
import pyterrier as pt
import pandas as pd
import ir_datasets
import time


logger = ir_datasets.log.easy()

import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class SlideGAR(pt.Transformer):
    """
    Required input columns: ['qid', 'query', 'docno', 'score', 'rank']
    Output columns: ['qid', 'query', 'docno', 'score', 'rank', 'iteration']
    where iteration defines the batch number which identified the document. Specifically
    even=initial retrieval   odd=corpus graph    -1=backfilled
    
    """
    def __init__(self,
        scorer: pt.Transformer,
        corpus_graph: 'CorpusGraph',
        num_results: int = 1000,
        step_size: int=10,
        buffer_size : int = 20,
        backfill: bool = True,
        enabled: bool = True,
        verbose: bool = False):
        """
            SlideGAR init method
            Args:
                scorer(pyterrier.Transformer): A transformer that scores query-document pairs. It will only be provided with ['qid, 'query', 'docno', 'score'].
                corpus_graph(pyterrier_adaptive.CorpusGraph): A graph of the corpus, enabling quick lookups of nearest neighbours
                num_results(int): The maximum number of documents to score (called "budget" and $c$ in the paper)
                step_size(int): The number of documents to select from the frontier in each iteration (called $b$ in the paper)
                buffer_size(int): The number of documents to keep in the current window (called $w$ in the paper)
                backfill(bool): If True, always include all documents from the initial stage, even if they were not re-scored
                enabled(bool): If False, perform re-ranking without using the corpus graph
                verbose(bool): If True, print progress information
        """
        self.scorer = scorer
        self.corpus_graph = corpus_graph
        self.num_results = num_results
        self.step_size = step_size
        self.buffer_size = buffer_size    
        self.backfill = backfill
        self.enabled = enabled
        self.verbose = verbose

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Applies Sliding Window based Adaptive Re-ranking to the provided dataframe. Essentially,
        Algorithm 1 from the paper.
        """
        result = {'qid': [], 'query': [], 'docno': [], 'rank': [], 'score': [], 'iteration': []}

        df = dict(iter(df.groupby(by=['qid'])))
        qids = df.keys()
        if self.verbose:
            qids = logger.pbar(qids, desc='adaptive re-ranking', unit='query')


        print("Starting SlideGAR")
        for qid in qids:
            query = df[qid]['query'].iloc[0]

            scores = {}
            res_map = [Counter(dict(zip(df[qid].docno, df[qid].score)))] # initial results
            if self.enabled:
                res_map.append(Counter()) # frontier

            r1_upto_now = {}
            selected_docs = []
            top_b = []

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
                    size = min(self.step_size, self.num_results - len(scores)-self.step_size) # get either the batch size or remaining budget (whichever is smaller)


                last_iteration = False
                if size == self.num_results - len(scores) - self.step_size and len(this_res)>=size:
                    last_iteration = True

                # build batch of documents to score in this round
                batch = this_res.most_common(size)
                # get the document ids of the batch
                batch_docs = [doc for doc, _ in batch]


                buffer =  top_b + batch_docs
                batch  = pd.DataFrame(buffer, columns=['docno'])
                batch['qid'] = [qid[0]] * len(batch)
                batch['query'] = query


                batch = self.scorer(batch)
                ranked_buffer = batch.docno.tolist()
                                        
                top_b  = ranked_buffer[:self.step_size]
                not_top_b = ranked_buffer[self.step_size:]

                if last_iteration==True or len(selected_docs)+len(ranked_buffer)>=self.num_results:
                    selected_docs = ranked_buffer + selected_docs
                    psuedo_rank = np.arange(len(selected_docs))
                    psuedo_scores = [1/(r+1) for r in psuedo_rank]
                    scores = {k: (s, iteration) for k, s in zip(selected_docs, psuedo_scores)}

                else:
                    selected_docs = not_top_b + selected_docs
                    scores = {k: (iteration) for k in selected_docs}

                
                psuedo_rank = np.arange(len(top_b))
                psuedo_scores = [1/(r+1) for r in psuedo_rank]
                r1_upto_now = {k: s for k, s in zip(top_b, psuedo_scores)}
                S = r1_upto_now

                # psuedo_rank = np.arange(len(ranked_buffer))
                # psuedo_scores = [1/(r+1) for r in psuedo_rank]
                # r1_upto_now = {k: s for k, s in zip(ranked_buffer, psuedo_scores)}
                # S = r1_upto_now



                self._drop_docnos_from_counters(batch_docs, res_map)

                if len(scores) < self.num_results and self.enabled:
                        res_map[1] = self._update_frontier(S, selected_docs)
                iteration += 1



            if last_iteration==False:
                print("Last iteration is False")
                selected_docs = top_b + selected_docs
                psuedo_rank = np.arange(len(selected_docs))
                psuedo_scores = [1/(r+1) for r in psuedo_rank]
                scores = {k: (s, iteration) for k, s in zip(selected_docs, psuedo_scores)}
            
            
            assert len(scores) <= self.num_results, f"len(scores)={len(scores)} num_results={self.num_results}"



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
        selected_docs.update(set(S.keys()))
            
        frontier = Counter()
        for did, score in S.items():
            target_neighbors = self.corpus_graph.neighbours(did)
            for n in target_neighbors:
                if n not in selected_docs:
                    if n not in frontier or score > frontier[n]:
                        frontier[n] = score

                    if len(frontier) >= self.step_size:
                        break  

            if len(frontier) >= self.step_size:
                break  



        return frontier                




    def _drop_docnos_from_counters(self, docnos, counters):
        for docno in docnos:
            for c in counters:
                del c[docno]
