# DistilBERT Sentiment Analysis on IMDb Movie Reviews vs Amazon Fine Foods Reviews
### Horia-Gabriel Radu and Alexandru-Liviu Bratosin
## Structure

```bash
├── data
│   ├── amazon (on Google Drive)
│   └── imdb   (on Google Drive)
├── demo
│   └── main.py
├── experiments
│   ├── test.yaml
│   └── train.yaml
├── lib
│   ├── data.py
│   ├── test.py
│   └── train.py
├── models
│   └── distilbert.py
└── notebooks
    └── main.ipynb
```

* `train.yaml` and `test.yaml`: Config files for training and testing.

* `data.py`: Dataset utilities.

* `test.py` and `train.py`: Training and testing functions.

* `distilbert.py`: PyTorch-Lightning model wrapper class.

* `notebooks/main.ipynb`: Notebook used for running the training and testing experiments.

### Datasets

- [IMDb Movie Reviews](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews)
- [Amazon Fine Food Reviews](https://www.kaggle.com/datasets/snap/amazon-fine-food-reviews)

Saved versions of the dataset splits we have used are also uploaded on [Google Drive](https://drive.google.com/drive/folders/1QPsU2oStKSmq7XB-qevKKtVoqf9-ySW2?usp=sharing).

### Models

Saved checkpoints and training logs are uploaded on [Google Drive](https://drive.google.com/file/d/1eoQ3kPVBSByyOL6iOH9vhgHjwy_LjkFM/view).

## Evaluation and Experiment Results

The following results are achieved with the parameters found in `experiments/train.yaml` and `experiments/test.yaml` config files.

```
Experiment: Train on IMDb,   test on IMDb,   w/o pretrain
    | Test accuracy: 0.850320 | pretrain=false;train=imdb;test=imdb     |

Experiment: Train on Amazon, test on Amazon, w/o pretrain
    | Test accuracy: 0.861111 | pretrain=false;train=amazon;test=amazon |

Experiment: Train on IMDb,   test on Amazon, w/o pretrain
    | Test accuracy: 0.730711 | pretrain=false;train=imdb;test=amazon   |

Experiment: Train on Amazon, test on IMDb,   w/o pretrain
    | Test accuracy: 0.725720 | pretrain=false;train=amazon;test=imdb   |


Experiment: Train on IMDb,   test on IMDb,   w/ pretrain
    | Test accuracy: 0.905840 | pretrain=true;train=imdb;test=imdb      |

Experiment: Train on Amazon, test on Amazon, w/ pretrain
    | Test accuracy: 0.920089 | pretrain=true;train=amazon;test=amazon  |

Experiment: Train on IMDb,   test on Amazon, w/ pretrain
    | Test accuracy: 0.868133 | pretrain=true;train=imdb;test=amazon    |

Experiment: Train on Amazon, test on IMDb,   w/ pretrain
    | Test accuracy: 0.849640 | pretrain=true;train=amazon;test=imdb    |
```
## Demo

The demo file is `demo/main.py`.

Usage:
```bash
usage: main.py [-h] --checkpoint CHECKPOINT [--max_sequence_length MAX_SEQUENCE_LENGTH] [--padding PADDING] [--truncation TRUNCATION]

NLU Sentiment Classification demo.

optional arguments:
  -h, --help            show this help message and exit
  --checkpoint CHECKPOINT
                        Path to model checkpoint.
  --max_sequence_length MAX_SEQUENCE_LENGTH
                        Maximum sequence length.
  --padding PADDING     Padding type.
  --truncation TRUNCATION
                        Truncation.
```

## References

* IMDb Dataset - Andrew L. Maas, Raymond E. Daly, Peter T. Pham, Dan Huang, Andrew Y. Ng, and Christopher Potts. (2011). Learning Word Vectors for Sentiment Analysis. The 49th Annual Meeting of the Association for Computational Linguistics (ACL 2011).
* Amazon Fine Food Dataset - J. McAuley and J. Leskovec. From amateurs to connoisseurs: modeling the evolution of user expertise through online reviews. WWW, 2013.
* DistilBERT model from Hugging Face - https://huggingface.co/docs/transformers/model_doc/distilbert (Accessed 11 May 2022)
