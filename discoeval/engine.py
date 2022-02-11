# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

'''

Generic sentence evaluation scripts wrapper

'''
from __future__ import absolute_import, division, unicode_literals
import os

from discoeval import utils
from discoeval.pdtb import PDTBEval
from discoeval.so import SPEval, BSOEval, DCEval, SSPEval
from discoeval.rst import RSTEval

class SE(object):
    def __init__(self, params, batcher, prepare=None):
        # parameters
        params = utils.dotdict(params)
        params.usepytorch = True if 'usepytorch' not in params else params.usepytorch
        params.seed = 1111 if 'seed' not in params else params.seed

        params.batch_size = 128 if 'batch_size' not in params else params.batch_size
        params.nhid = 0 if 'nhid' not in params else params.nhid
        params.kfold = 5 if 'kfold' not in params else params.kfold

        if 'classifier' not in params or not params['classifier']:
            params.classifier = {'nhid': 0}

        assert 'nhid' in params.classifier, 'Set number of hidden units in classifier config!!'

        self.params = params

        # batcher and prepare
        self.batcher = batcher
        self.prepare = prepare if prepare else lambda x, y: None

        self.list_tasks = ['SPthesis', 'SPwiki', 'SPmlsum',
                           'DCopus', 'DCwiki', 'DCgdd',
                           'BSOthesis', 'BSOwiki', 'BSOmlsum', 
                           'SSPmlsum',
                           'PDTB-E', 'PDTB-I', 
                           'RST']

    def eval(self, name):
        # evaluate on evaluation [name], either takes string or list of strings
        if (isinstance(name, list)):
            self.results = {x: self.eval(x) for x in name}
            return self.results

        tpath = self.params.task_path
        assert name in self.list_tasks, str(name) + ' not in ' + str(self.list_tasks)

        if name == 'SPthesis':
            self.evaluation = SPEval(os.path.join(tpath, 'discoeval/SP/thesis'), "Sentence Position thesis", nclasses=5, seed=self.params.seed)
        elif name == 'SPwiki':
            self.evaluation = SPEval(os.path.join(tpath, 'discoeval/SP/wiki'), "Sentence Position wiki", nclasses=5, seed=self.params.seed)
        elif name == 'SPmlsum':
            self.evaluation = SPEval(os.path.join(tpath, 'discoeval/SP/mlsum'), "Sentence Position mlsum", nclasses=5, seed=self.params.seed)
        elif name == 'BSOthesis':
            self.evaluation = BSOEval(os.path.join(tpath, 'discoeval/BSO/thesis'), "Binary Sentence Ordering thesis", nclasses=2, seed=self.params.seed)
        elif name == 'BSOwiki':
            self.evaluation = BSOEval(os.path.join(tpath, 'discoeval/BSO/wiki'), "Binary Sentence Ordering wiki", nclasses=2, seed=self.params.seed)
        elif name == 'BSOmlsum':
            self.evaluation = BSOEval(os.path.join(tpath, 'discoeval/BSO/mlsum'), "Binary Sentence Ordering mlsum", nclasses=2, seed=self.params.seed)
        elif name == 'DCopus':
            self.evaluation = DCEval(os.path.join(tpath, 'discoeval/DC/opus'), "Discourse Coherence opus", nclasses=2, seed=self.params.seed)
        elif name == 'DCwiki':
            self.evaluation = DCEval(os.path.join(tpath, 'discoeval/DC/wiki'), "Discourse Coherence wiki", nclasses=2, seed=self.params.seed)
        elif name == 'DCgdd':
            self.evaluation = DCEval(os.path.join(tpath, 'discoeval/DC/gdd'), "Discourse Coherence gdd", nclasses=2, seed=self.params.seed)
        elif name == 'SSPmlsum':
            self.evaluation = SSPEval(os.path.join(tpath, 'discoeval/SSP/mlsum'), "Sentence Section Prediction mlsum", nclasses=2, seed=self.params.seed)
        elif name == 'RST':
            self.evaluation = RSTEval(os.path.join(tpath, 'discoeval/RST'), seed=self.params.seed)

        self.params.current_task = name
        self.evaluation.do_prepare(self.params, self.prepare)

        self.results = self.evaluation.run(self.params, self.batcher)

        return self.results
