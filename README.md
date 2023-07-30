# Detecting Stress From Social Media Posts

## Environment
This project has the following dependencies:
```
ax==0.52.0
numpy==1.23.4
scikit_learn==1.1.3
torch==1.13.0
transformers==4.24.0
```


## Functions
Using the `main.py` file you can execute three primary functions:


### 1. Training
```
python main.py train --model model_name --bert language_model_name --main_dataset path/to/train/set --dev_file path/to/dev/set --test_file path/to/test/set --log --logfile path/to/log/file
```

Helpful Arguments:
- `main_dataset` is the training set. It should be a .csv file with two columns (text and labels).
- `dev_file` is the development set. It should be a .csv file with two columns (text and labels).
- `test_file` is the test set. It should be a .csv file with two columns (text and labels).
- `model` is used to run different architectures. More information about how to select this is provided in the "How To Run Different Architectures" section. There are 6 options:
    - bert
    - roberta
    - multi_alt
    - multi_alt-roberta
    - multi
    - multi-roberta
- `bert` is used to specify a path to a language model or the name of a huggingface model (e.g., "bert-base-uncased"). More information about how to select this is provided in the "How To Use Different Language Models" section.
- `save_path` appends `-params.pth` onto this to save the model, and will also 
save meta-information by appending `-meta.pkl` onto your save path.
- `save_lm` saves the language model only. This requires `save-path` and will append `-lm` to the path. This is used for the Fine-Tune models.
- `aux_datasets` is used to specify a list of auxiliary training datasets for the Multi<sup>*Alt*</sup> models.
- `log` saves the log to a file.
- `logfile` specifies where to save the log.


### 2. Prediction
```
python main.py predict --model model_name --model_path path/to/saved/model --data path/to/data/set --out_path path/to/output/predictions --log -- logfile path/to/log/file
```

Helpful Arguments:
- `model_path` is the path to a model that you have already trained. It should be the same name as the `save_path` argument that you passed, but without the `-params.pth` or `-meta.pkl` at the end of it. 
- `data` is the dataset that you want to be get predictions for. It should be a .csv file with two columns (text and labels). Whatever is originally in the labels column will be overwritten.
- `model` is used to run different architectures. More information about how to select this is provided in the "How To Run Different Architectures" section. There are 6 options:
    - bert
    - roberta
    - multi_alt
    - multi_alt-roberta
    - multi
    - multi-roberta
- `log` saves the log to a file.
- `logfile` specifies where to save the log.

### 3. Evaluation
```
python main.py evaluate --model model_name --model_path path/to/saved/model --data path/to/data/set --out_path path/to/output/predictions --log -- logfile path/to/log/file
```
Helpful Arguments:
- `model_path` is the path to a model that you have already trained. It should be the same name as the `save_path` argument that you passed, but without the `-params.pth` or `-meta.pkl` at the end of it. 
- `data` is the dataset that you want to evaluate your mode on. It should be a .csv file with two columns (text and labels).
- `model` is used to run different architectures. More information about how to select this is provided in the "How To Run Different Architectures" section. There are 6 options:
    - bert
    - roberta
    - multi_alt
    - multi_alt-roberta
    - multi
    - multi-roberta
- `log` saves the log to a file.
- `logfile` specifies where to save the log.

## How To Run Different Architectures

All of these examples will use the BERT as the language model. More information about how to use different language models is provided below. 

### **Single-Task**
The Single-Task model can be run all in one step. 
```
python main.py train --model bert --bert bert-base-uncased --main_dataset path/to/train/set --dev_file path/to/dev/set --test_file path/to/test/set --log --logfile path/to/log/file
```

### **Fine-Tune**
The Fine-Tune model must be trained in two steps. First, train a Single-Task model using the emotion detection datasets, and save the language model.
```
python main.py train --model bert --bert bert-base-uncased --main_dataset path/to/train/set --dev_file path/to/dev/set --test_file path/to/test/set --log --logfile path/to/log/file --save_lm --save_path path/to/save/language/model
```
Then run another Single-Task model using the stress detection datasets, but this time instead of loading the standard BERT model give the path to the language model that you saved earlier. This should be a folder ending wit `-lm` that contains a `config.json` file and a `pytorch_model.bin` file. 

```
python main.py train --model bert --bert path/to/language/model --main_dataset path/to/train/set --dev_file path/to/dev/set --test_file path/to/test/set --log --logfile path/to/log/file
```

### **Multi<sup>*Alt*</sup>**
Running Multi<sup>*Alt*</sup> is very similar to to Single-Task, but you must change the model name to `multi_alt` and specify three auxiliary datasets. These datasets should be the train, dev, and test set for emotion detection (provided in that order.)
```
python main.py train --model multi_alt --bert bert-base-uncased --main_dataset path/to/train/set --dev_file path/to/dev/set --test_file path/to/test/set --log --logfile path/to/log/file --aux_datasets path/to/emotion/train path/to/emotion/dev path/to/emotion/test
```

### **Multi**
The Multi model must be trained in multiple steps. First, train a Single-Task model using the emotion detection datasets, and save the model. 
```
python main.py train --model bert --bert bert-base-uncased --main_dataset path/to/train/set --dev_file path/to/dev/set --test_file path/to/test/set --log --logfile path/to/log/file --save_path path/to/save/language/model
```
Next use the `predict` function to label the stress training set with emotions. (Note: this will overwrite the original stress labels)
```
python main.py predict --model bert --model_path path/to/saved/model --data path/to/training/set --out_path path/to/output/predictions --log -- logfile path/to/log/file
```
Then create a new three column training set that has the text, stress labels, and emotion labels in that order. 

Finally, train a Multi model using that new training set. 

```
python main.py train --model multi --bert bert-base-uncased --main_dataset path/to/emotion/labeled/training/set --dev_file path/to/dev/set --test_file path/to/test/set --log --logfile path/to/log/file
```

## How To Use Different Language Models

### **BERT-Based Models**

To train the four architectures with different types of BERT models (like MentalBERT) all you need to do is change the ``bert`` parameter from `bert-base-uncased` to the name of another BERT-based huggingface model or to the filepath of a saved language model.

Here is an example of how to run a Single-Task model with MentalBERT:
```
python main.py train --model bert --bert mental/mental-bert-base-uncased --main_dataset path/to/train/set --dev_file path/to/dev/set --test_file path/to/test/set --log --logfile path/to/log/file
```

### **RoBERTa-Based Models**
To train the four architectures with different types of RoBERTa models (like MentalRoBERTa) you will also need to change the `model` parameter. 

- In any cases where the model would be `bert` you change it to `roberta`.
- In any cases where the model would be `multi_alt` you change it to `multi_alt-roberta`.
- In any cases where the model would be `multi` you change it to `multi-roberta`.

Additionally, you need to change the ``bert`` parameter from `bert-base-uncased` to the name of another RoBERTa-based huggingface model or to the filepath of a saved language model. 

Here is an example of how to run a Single-Task model with RoBERTa:
```
python main.py train --model roberta --bert roberta-base --main_dataset path/to/train/set --dev_file path/to/dev/set --test_file path/to/test/set --log --logfile path/to/log/file
```

Here is an example of how to run a Single-Task model with MentalRoBERTa:
```
python main.py train --model roberta --bert mental/mental-roberta-base --main_dataset path/to/train/set --dev_file path/to/dev/set --test_file path/to/test/set --log --logfile path/to/log/file
```