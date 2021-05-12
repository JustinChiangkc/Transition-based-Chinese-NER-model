
# Transition-based-Chinese-NER-model
The model is a Chinese named-entity recognition model using transition-based method that can achieve squential time in conducting word segmentation and NER tagging for Chinese corpus.

## Environment setting:
* Python 3.6
* Pytorch 1.0.0


## Desired labeling

* BIO scheme NER labeling 
* CoNLL tagging format

ex: 

	你	O
	從	O
	日	B-GPE
	本	I-GPE
	回	O
	來	O
	嘛	O


## Data structures

 * **buffer** - sequence of tokens to be processed
 * **stack** - working memory for entity candidate
 * **output buffer** - sequence of NER labeled segments

## Operations(transitions)

 * `SHIFT` - move token from **buffer** to top of **stack**
 * `REDUCE(X)` - all words on **stack** are popped, combined to form a segment and labeled with `X` and copied to **output buffer**
 * `OUT` - move one token from **buffer** to **output buffer**


## Usage


### Data preprocessing - generate configuration
The first step of using transition-based method is to convert the BIO NER tagging to transition-based configuration

```bash
	python3 Oracle_NER.py -f CoNLL_inputfile -o "output_directory"
```
ex: 

	你	O
	從	O
	日	B-GPE
	本	I-GPE
	回	O
	來	O
	嘛	O 

Corresponding sequence of operations, Oracle:

	OUT
	OUT
	SHIFT
	SHIFT
	REDUCE(GPE)
	OUT
	OUT
	OUT




### Training

```bash
	python3 main.py --data_path "traning_data_directory"
```


### Evaluation

```bash
	python3 test.py --model_path "model_path" --testset_path "testing_data_directory"
```

### Run API


```bash
	python3 run.py --model_path "model_path"
```
*  Send POST requests to the url, [URL]/ner_tagger, to check NER result, ex: (http://chineseNER.com/ner_tagger/)







