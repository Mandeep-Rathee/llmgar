import argparse

import pyterrier as pt
pt.init()
from pyterrier.measures import *
from pyterrier_adaptive import  CorpusGraph
from slide_gar import SlideGAR
from gar_rm3 import RM3GAR

from pyterrier_pisa import PisaIndex


from rerankers import Reranker

import torch
import transformers
import numpy as np
import pandas as pd
from dotenv import load_dotenv
import json
import os



parser = argparse.ArgumentParser()
parser.add_argument("--lk", type=int, default=16, help="the value of k for selecting k neighbourhood graph")
parser.add_argument("--graph_name", type=str, default="gbm25", help="name of the graph")
parser.add_argument("--dl_type", type=int, default=19, help="dl 19 or 20")
parser.add_argument("--budget", type=int, default=50, help="budget c")
parser.add_argument("--b", type=int, default=10, help="step size for listwise ranker")
parser.add_argument("--verbose", action="store_true", help="if show progress bar.")
parser.add_argument("--retriever", type=str, default="bm25", help="name of the retriever")
parser.add_argument("--model_name", type=str, default="zephyr", help="model name")
parser.add_argument("--model_type", type=str, default="rankllm", help="model type")
parser.add_argument("--buffer", type=int, default=20, help="buffer size")


args = parser.parse_args()

torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
np.random.seed(args.seed)
transformers.logging.set_verbosity_error()
load_dotenv()

dataset = pt.get_dataset('irds:msmarco-passage')


retriever = PisaIndex.from_dataset('msmarco_passage').bm25()
    
openai_API_KEY = os.getenv('YOUR_APIKEY')
base_url = os.getenv('YOUR_API_BASE')

ranker  = Reranker(model_name=args.model_name, model_type=args.model_type, api_key=openai_API_KEY, verbose=0)
scorer =  pt.text.get_text(dataset, 'text') >> ranker.as_pyterrier_transformer()



"""
We use the corpus graph released by the author of GAR for our experiments. 
"""

graph = CorpusGraph.from_hf('macavaney/msmarco-passage.corpusgraph.bm25.128').to_limit_k(args.lk)   


pd.set_option('display.max_columns', None) 
pd.set_option('display.width', None)  # Display full width of the terminal

########################### Listwise GAR ###########################


#save_dir=f"pyterrier_runs/{args.graph_name}_{args.model_name}_dl{args.dl_type}_ret_{args.retriever}_lk_{args.lk}"
#save_dir=f"pyterrier_runs/{args.graph_name}_{args.model_name}_dl{args.dl_type}_ret_{args.retriever}_lk_{args.lk}_topb/"

if args.budget==50:
    #ave_dir=f"pyterrier_runs/dl{args.dl_type}/{args.graph_name}/{args.retriever}>>{args.model_name}_lk_{args.lk}_topb/"
    save_dir=f"pyterrier_runs/{args.graph_name}_{args.model_name}_dl{args.dl_type}_ret_{args.retriever}_lk_{args.lk}/"
else:
    save_dir=f"pyterrier_runs/dl{args.dl_type}/{args.graph_name}/{args.retriever}>>{args.model_name}_lk_{args.lk}_c_{args.budget}_topb/"


# if not os.path.exists(save_dir):
#     print("not found, folder created")
#     print(save_dir)
#     exit()
#     os.makedirs(save_dir)


dataset = pt.get_dataset(f'irds:msmarco-passage/trec-dl-20{args.dl_type}/judged')

topics = dataset.get_topics()
qrels = dataset.get_qrels()


results = pt.Experiment(
    [retriever % args.budget >> scorer, 
    retriever % args.budget >> SlideGAR(scorer, graph,verbose=args.verbose, num_results=args.budget, 
                                        step_size=args.b, buffer_size=args.buffer),
    ],
    topics,
    qrels,
    [nDCG@10, R(rel=2)@b, Judged@10],
    names=[f'{args.retriever}_{args.model_name}-full', 
        f'{args.retriever}_GAR({args.model_name})'
        ],
    #save_dir=save_dir,
    #save_mode='reuse',
    #baseline=0,
    #correction='bonferroni', 
)
print(results.T)


















