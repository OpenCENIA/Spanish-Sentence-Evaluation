# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from __future__ import absolute_import, division, unicode_literals

import sys
import io
import numpy as np
import logging
import torch
import code
import argparse

# Set PATHs
PATH_TO_SENTEVAL = '../'
PATH_TO_DATA = '../data'

# import DiscoEval and SentEval
sys.path.insert(0, PATH_TO_SENTEVAL)
import discoeval
import senteval

from elmoformanylangs import Embedder

# prepare and batcher
def prepare(params, samples):
    pass

def batcher(params, batch):
    model = params["model"]
    tokenizer = params.tokenizer
    batch = [[token.lower() for token in sent] for sent in batch]
    embeddings = model.sents2elmo(batch)

    return np.stack([v.mean(0) for v in embeddings], axis=0)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--benchmark", default="discoeval", type=str,
                        choices=["senteval", "discoeval"],
                        help="the name of benchmark")
    parser.add_argument("--task_index", default=0, type=int,
                        help="which task to perform")
    parser.add_argument("--model_path", default='.', type=str,
                        help="path of model")
    parser.add_argument("--seed", default=1111, type=int,
                        help="which seed to use")
    args = parser.parse_args()

    # Set up logger
    logging.basicConfig(format='%(asctime)s : %(message)s', level=logging.DEBUG)
    model = Embedder(args.model_path)

    # Set params for DiscoEval or SentEval
    params = {'task_path': PATH_TO_DATA, 'usepytorch': True, 'kfold': 10, 'batch_size': 16,
              'model': model, 'seed': args.seed}
    params['classifier'] = {'nhid': 0, 'optim': 'adam', 'batch_size': 64,
                            'tenacity': 5, 'epoch_size': 4}

    if args.benchmark == 'discoeval':
        se = discoeval.engine.SE(params, batcher, prepare)
        transfer_tasks = [
            ['SPwiki', 'SPmlsum', 'SPthesis'],
            ['BSOwiki', 'BSOmlsum', 'BSOthesis'],
            ['DCwiki', 'DCopus', 'DCgdd'],
            ['SSPmlsum'],
            ['RST']]
    elif args.benchmark == 'senteval':
        se = senteval.engine.SE(params, batcher, prepare)
        transfer_tasks = [
            ['TASS1','TASS2','MC','FAC', 'SQC'],
            ['PAWS', 'NLI', 'SICKEntailment'],
            ['SICKRelatedness', 'STS14', 'STS15', 'STS17'],
            ['Length', 'WordContent', 'Depth', 'BigramShift',
             'Tense', 'SubjNumber', 'ObjNumber',
             'OddManOut', 'CoordinationInversion']]

    results = se.eval(transfer_tasks[args.task_index])
    print(results)
