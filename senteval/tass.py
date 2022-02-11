# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

'''
TASS - multiclass classification
'''

from __future__ import absolute_import, division, unicode_literals

import os
import io
import logging
import numpy as np

from senteval.tools.validation import SplitClassifier


class TASSEval(object):
    def __init__(self, task_path, nclasses=3, seed=1111):
        self.seed = seed

        # 3 clases: 0 NEU, 1 NEG, 2 POS
        # 7 clases: 0 others, 1 joy, 2 sadness, 3 anger, 4 surprise, 5 disgust, 6 fear
        assert nclasses in [3, 7]
        self.nclasses = nclasses
        self.task_name = 'Polarity' if self.nclasses == 3 else 'Emotion'
        logging.debug('***** Transfer task : TASS %s classification *****\n\n', self.task_name)

        train = self.loadFile(os.path.join(task_path, 'train.txt'))
        dev = self.loadFile(os.path.join(task_path, 'dev.txt'))
        test = self.loadFile(os.path.join(task_path, 'test.txt'))
        self.tass_data = {'train': train, 'dev': dev, 'test': test}

    def do_prepare(self, params, prepare):
        samples = self.tass_data['train']['X'] + self.tass_data['dev']['X'] + \
                  self.tass_data['test']['X']
        return prepare(params, samples)

    def loadFile(self, fpath):
        tass_data = {'X': [], 'y': []}
        with io.open(fpath, 'r', encoding='utf-8') as f:
            for line in f:
                sample = line.strip().split('\t')
                tass_data['y'].append(int(sample[1]))
                tass_data['X'].append(sample[0].split())
        assert max(tass_data['y']) == self.nclasses - 1
        return tass_data

    def run(self, params, batcher):
        tass_embed = {'train': {}, 'dev': {}, 'test': {}}
        bsize = params.batch_size

        for key in self.tass_data:
            logging.info('Computing embedding for {0}'.format(key))
            # Sort to reduce padding
            sorted_data = sorted(zip(self.tass_data[key]['X'],
                                     self.tass_data[key]['y']),
                                 key=lambda z: (len(z[0]), z[1]))
            self.tass_data[key]['X'], self.tass_data[key]['y'] = map(list, zip(*sorted_data))

            tass_embed[key]['X'] = []
            for ii in range(0, len(self.tass_data[key]['y']), bsize):
                batch = self.tass_data[key]['X'][ii:ii + bsize]
                embeddings = batcher(params, batch)
                tass_embed[key]['X'].append(embeddings)
            tass_embed[key]['X'] = np.vstack(tass_embed[key]['X'])
            tass_embed[key]['y'] = np.array(self.tass_data[key]['y'])
            logging.info('Computed {0} embeddings'.format(key))

        config_classifier = {'nclasses': self.nclasses, 'seed': self.seed,
                             'usepytorch': params.usepytorch,
                             'classifier': params.classifier}

        clf = SplitClassifier(X={'train': tass_embed['train']['X'],
                                 'valid': tass_embed['dev']['X'],
                                 'test': tass_embed['test']['X']},
                              y={'train': tass_embed['train']['y'],
                                 'valid': tass_embed['dev']['y'],
                                 'test': tass_embed['test']['y']},
                              config=config_classifier)

        devacc, testacc = clf.run()
        logging.debug('\nDev acc : {0} Test acc : {1} for \
            TASS {2} classification\n'.format(devacc, testacc, self.task_name))

        return {'devacc': devacc, 'acc': testacc,
                'ndev': len(tass_embed['dev']['X']),
                'ntest': len(tass_embed['test']['X'])}
