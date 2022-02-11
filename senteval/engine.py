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

from senteval import utils
from senteval.binary import MCEval, FACEval
from senteval.nli import NLIEval
from senteval.sqc import SQCEval
from senteval.sick import SICKRelatednessEval, SICKEntailmentEval
from senteval.paws import PAWSEval
from senteval.sts import STS14Eval, STS15Eval, STS17Eval
from senteval.tass import TASSEval
from senteval.probing import *

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

        self.list_tasks = ['MC', 'FAC', 'TASS1', 'TASS2', 'SQC', 'PAWS',
                           'SICKRelatedness', 'SICKEntailment', 'NLI',
                           'STS14', 'STS15', 'STS17',
                           'Length', 'WordContent', 'Depth',
                           'BigramShift', 'Tense', 'SubjNumber', 'ObjNumber',
                           'OddManOut', 'CoordinationInversion']

    def eval(self, name):
        # evaluate on evaluation [name], either takes string or list of strings
        if (isinstance(name, list)):
            self.results = {x: self.eval(x) for x in name}
            return self.results

        tpath = self.params.task_path
        assert name in self.list_tasks, str(name) + ' not in ' + str(self.list_tasks)

        # Original SentEval tasks
        if name == 'MC':
            self.evaluation = MCEval(tpath + '/senteval/MC', seed=self.params.seed)
        elif name == 'FAC':
            self.evaluation = FACEval(tpath + '/senteval/FAC', seed=self.params.seed)
        elif name == 'TASS1':
            self.evaluation = TASSEval(tpath + '/senteval/TASS/task1', nclasses=3, seed=self.params.seed)
        elif name == 'TASS2':
            self.evaluation = TASSEval(tpath + '/senteval/TASS/task2', nclasses=7, seed=self.params.seed)
        elif name == 'SQC':
            self.evaluation = SQCEval(tpath + '/senteval/SQC', seed=self.params.seed)
        elif name == 'PAWS':
            self.evaluation = PAWSEval(tpath + '/senteval/PAWS', seed=self.params.seed)
        elif name == 'SICKRelatedness':
            self.evaluation = SICKRelatednessEval(tpath + '/senteval/SICK', seed=self.params.seed)
        elif name == 'SICKEntailment':
            self.evaluation = SICKEntailmentEval(tpath + '/senteval/SICK', seed=self.params.seed)
        elif name == 'NLI':
            self.evaluation = NLIEval(tpath + '/senteval/NLI', seed=self.params.seed)
        elif name in ['STS14', 'STS15', 'STS17']:
            fpath = name + '-es-test'
            self.evaluation = eval(name + 'Eval')(tpath + '/senteval/STS/' + fpath, seed=self.params.seed)

        # Probing Tasks
        elif name == 'Length':
                self.evaluation = LengthEval(tpath + '/senteval/Probing', seed=self.params.seed)
        elif name == 'WordContent':
                self.evaluation = WordContentEval(tpath + '/senteval/Probing', seed=self.params.seed)
        elif name == 'Depth':
                self.evaluation = DepthEval(tpath + '/senteval/Probing', seed=self.params.seed)
        elif name == 'BigramShift':
                self.evaluation = BigramShiftEval(tpath + '/senteval/Probing', seed=self.params.seed)
        elif name == 'Tense':
                self.evaluation = TenseEval(tpath + '/senteval/Probing', seed=self.params.seed)
        elif name == 'SubjNumber':
                self.evaluation = SubjNumberEval(tpath + '/senteval/Probing', seed=self.params.seed)
        elif name == 'ObjNumber':
                self.evaluation = ObjNumberEval(tpath + '/senteval/Probing', seed=self.params.seed)
        elif name == 'OddManOut':
                self.evaluation = OddManOutEval(tpath + '/senteval/Probing', seed=self.params.seed)
        elif name == 'CoordinationInversion':
                self.evaluation = CoordinationInversionEval(tpath + '/senteval/Probing', seed=self.params.seed)

        self.params.current_task = name
        self.evaluation.do_prepare(self.params, self.prepare)

        self.results = self.evaluation.run(self.params, self.batcher)

        return self.results
