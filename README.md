# Evaluation Benchmarks for Spanish Sentence Representations

We have introduced **Spanish SentEval** and **Spanish DiscoEval**, aiming to assess the capabilities of stand-alone and discourse-aware sentence representations, respectively. Our benchmarks include considerable pre-existing and newly constructed datasets that address different tasks from various domains. Besides, we evaluate and analyze the most recent pre-trained Spanish language models to exhibit their capabilities and limitations. Our benchmarks consist of a single pipeline that attempts a fair and less cumbersome assessment across multiple tasks with text from different domains.

## Dependencies

This code is written in python. The dependencies are:

* Python 3
* Pytorch >= 1.10.0
* scikit-learn >= 0.22.2
* numpy >= 1.18.2
* [transformers](https://github.com/huggingface/transformers) >= 4.0.0
* [sent2vec](https://github.com/epfml/sent2vec)
* [elmoformanylangs](https://github.com/HIT-SCIR/ELMoForManyLangs)

## Tasks

Our benchmark allows you to evaluate your sentence embeddings as features for the following type of tasks. They are mentioned below with consisting datasets:

**Spanish SentEval**

| ID | Group                               | Datasets                                                                 |
|----|-------------------------------------|--------------------------------------------------------------------------|
| 0  | Sentence Classification (SC)        | TASS1, TASS2, MC, FAC, SQC                                               |
| 1  | Sentence Pair Classification (SPC)  | PAWS-es, NLI-es, SICK-es-E                                               |
| 2  | Semantic Similarity (SS)            | SICK-es-R, STS14, STS15, STS17                                           |
| 3  | Linguistic Probing Tasks (LPT)      | SentLen, WC, Tree Depth, BiShift, tense, SubjNum, ObjNum, SOMO, CoordInv |

**Spanish DiscoEval**

| ID | Group                             | Datasets            |
|----|-----------------------------------|---------------------|
| 0  | Sentence Position (SP)            | wiki, mlsum, thesis |
| 1  | Binary Sentence Ordering (BSO)    | wiki, mlsum, thesis |
| 2  | Discourse Coherence (DC)          | wiki, opus, gdd     |
| 3  | Sentence Section Prediction (SSP) | mlsum               |
| 4  | Discourse Relations (DR)          | RST                 |

## Run the Experiments:
"examples" folder contain all the scripts to run our experiments.

For elmo:
```python
python elmo.py --benchmark discoeval --task_index 0
```
For sent2vec:
```python
python sent2vec.py --benchmark discoeval --task_index 0
```
For transformer:
```python
python transformer.py --benchmark senteval --task_index 0 --model_name beto --layer avg
```

Arguments to set:
```bash
benchmark:             # senteval or discoeval
task_index:            # id of the group of tasks
model_name:            # (only for transformer) beto, bertin, roberta, electra, mbert
layer:                 # (only for transformer) pooling method
model_path:            # (only for sent2vec and elmo) path of the pretrained model
seed:                  # random seed value
```

## How to use:
In order to evaluate your sentence embeddings, you need to use/implement two functions:
1. **prepare** (sees the whole dataset of each task and can thus construct the word vocabulary, the dictionary of word vectors etc)
2. **batcher** (transforms a batch of text sentences into sentence embeddings)

### 1.) prepare (params, samples) (optional)
*batcher* only sees one batch at a time while the *samples* argument of *prepare* contains all the sentences of a task.

```
prepare (params, samples)
```
* *params*: senteval parameters.
* *samples*: list of all sentences from the transfer task.
* *output*: No output. Arguments stored in "params" can further be used by *batcher*.

### 2.) batcher (params, batch)
```
batcher(params, batch)
```
* *params*: senteval parameters.
* *batch*: numpy array of text sentences (of size params.batch_size)
* *output*: numpy array of sentence embeddings (of size params.batch_size)

*Example*: in sent2vec.py, batcher is used to return batch embeddings given a particular model based on specific toenizer.  

### 3.) Evaluation on probing tasks
After having implemented the batch and prepare function for your own sentence encoder,

1) To perform the actual evaluation, first import senteval and discoeval and set its parameters:
```python
import senteval
import discoeval

For elmo.py and sent2vec.py : 
params = {'task_path': PATH_TO_DATA, 'usepytorch': True, 'kfold': 10, 'batch_size': 16, 'model': model}

For transformer.py :
params = {'task_path': PATH_TO_DATA, 'usepytorch': True, 'kfold': 10, 'batch_size': 16, 'tokenizer': tokenizer, "layer": args.layer, "model": model}

```

2) Set the parameters of the classifier (when applicable):
```python
params['classifier'] = {'nhid': 0, 'optim': 'adam', 'batch_size': 64,
                            'tenacity': 5, 'epoch_size': 4}
```
3) Create an instance of the class SE:
```python
se = senteval.engine.SE(params, batcher, prepare) 
or 
se = discoeval.engine.SE(params, batcher, prepare)
```

4) The current list of available tasks is:
```python
Discoeval:
transfer_tasks = [
            ['SPthesis', 'SPnews', 'SPwiki'], 
            ['BSOthesis', 'BSOnews', 'BSOwiki'], 
            ['DCopus', 'DCwiki', 'DCgdd'], 
            ['SSPmlsum'],
            ['RST']]
            
Senteval:
transfer_tasks = [
            ['TASS1','TASS2','MC','BR','FAC', 'SQC'],
            ['PAWS', 'NLI', 'SICKEntailment'],
            ['SICKRelatedness', 'STS14', 'STS15', 'STS17'],
            ['Length', 'WordContent', 'Depth', 'BigramShift', 
             'Tense', 'SubjNumber', 'ObjNumber',
             'OddManOut', 'CoordinationInversion']]
```

5) To run the evaluation:
```
results = se.eval(transfer_tasks)
```

## Parameters
Global parameters :
```bash
task_path                   # path to datasets (required)
seed                        # seed
usepytorch                  # use cuda-pytorch (else scikit-learn) where possible
kfold                       # k-fold validation
batch_size					# batch size used to feed the encoder

```

Parameters of the classifier:
```bash
nhid:                       # number of hidden units 
optim:                      # optimizer ("sgd,lr=0.1", "adam", "rmsprop" ..)
tenacity:                   # how many times dev acc does not increase before training stops
epoch_size:                 # each epoch corresponds to epoch_size pass on the train set
max_epoch:                  # max number of epoches
dropout:                    # dropout for MLP
batch_size					# batch size used to train the classifier

```

To produce results that are **comparable to the literature**:
For elmo.py and sent2vec.py : 
```python
params = {'task_path': PATH_TO_DATA, 'usepytorch': True, 'kfold': 10, 'batch_size': 16,'model': model}
params['classifier'] = {'nhid': 0, 'optim': 'adam', 'batch_size': 64,
'tenacity': 5, 'epoch_size': 4}
```

For transformer.py : 
```python
params = {'task_path': PATH_TO_DATA, 'usepytorch': True, 'kfold': 10, 'batch_size': 16, 'tokenizer': tokenizer, "layer": args.layer, "model": model}
params['classifier'] = {'nhid': 0, 'optim': 'adam', 'batch_size': 64, 'tenacity': 5, 'epoch_size': 4}

```

## Reference

Please considering citing [[1]](https://arxiv.org/abs/2204.07571) if using this code for evaluating Spanish sentence representations.

```
@inproceedings{araujo2022evaluation,
  title="Evaluation Benchmarks for Spanish Sentence Representations",
  author="Araujo, Vladimir and Carvallo, Andr{\'e}s and Kundu, Souvik and Ca{\~n}ete, Jos{\'e} and Mendoza, Marcelo and Mercer, Robert E and Bravo-Marquez, Felipe and Moens, Marie-Francine and Soto, Alvaro",
  booktitle = "Proceedings of the 12th Language Resources and Evaluation Conference",
  year = "2022",
  address = "Marseille, France",
  publisher = "European Language Resources Association",
}
```

Contact: [vgaraujo@uc.cl](mailto:vgaraujo@uc.cl), [afcarvallo@uc.cl](mailto:afcarvallo@uc.cl), [skundu6@uwo.ca](mailto:skundu6@uwo.ca)
