from typing import Any

import pandas as pd
import pyterrier as pt
import pyterrier_alpha as pta


class RerankerPyterrierTransformer(pt.Transformer):
    def __init__(self, model: Any):
        self.model = model

    def transform(self, inp: pd.DataFrame) -> pd.DataFrame:

        pta.validate.result_frame(inp, ['query', 'text'])
        return pt.apply.by_query(self._transform_by_query)(inp)

    def _transform_by_query(self, query_inp: pd.DataFrame) -> pd.DataFrame:

        
        results = self.model.rank(
            query=query_inp['query'].iloc[0],
            docs=query_inp['text'].tolist(),
            doc_ids=query_inp['docno'].tolist(),
            rank_start=0,
            rank_end=len(query_inp['docno'].tolist()),
        
        )

        if results.has_scores:
            doc_score_mapping = {r.doc_id: r.score for r in results}
            new_scores = [doc_score_mapping[doc_id] for doc_id in query_inp['docno'].tolist()]

            res = query_inp.assign(score=new_scores)
            res = res.sort_values('score', ascending=False)

            return pt.model.add_ranks(res)
        
        else:
            doc_rank_mapping = {r.doc_id: r.rank for r in results}
            doc_score_mapping = {r.doc_id: 1/(r.rank+1) for r in results}  ### for RankLLM rank starts from 0 and RankGPT rank starts from 1
            rank = [doc_rank_mapping[doc_id] for doc_id in query_inp['docno'].tolist()]
            new_scores = [doc_score_mapping[doc_id] for doc_id in query_inp['docno'].tolist()]

            res = query_inp.assign(score=new_scores)
            res = res.sort_values('score', ascending=False)

            return pt.model.add_ranks(res)
