# ALF at SemEval-2024 Task 9
This repository contains the source code of our proposed system for [SemEval 2024 Task 9: BrainTeaser](https://aclanthology.org/2024.semeval-1.274/), a QA benchmark designed to evaluate NLP models’ lateral thinking and creative reasoning abilities. Our experiments focus on two prominent families of pre-trained models, BERT and T5. More details are explained in the [corresponding paper](https://aclanthology.org/2024.semeval-1.218/).

## Requirements
It is recommended to create a python environment before installing the requirements.
```bash
pip install -r requirements.txt
```

## Train a model
`finetune_bert.py` and `finetune_t5.py` follow the same command line interface.
```bash
# Multi-dataset training on BrainTeaser and RiddleSense
python finetune_bert.py \
    --dataset "bt_fold0|rs" \
    --checkpoint "microsoft/deberta-v3-base" \
    --name "bt_rs_debertav3" \
    --log_steps 0.25
```

## Citation
```bibtex
@inproceedings{farokh-zeinali-2024-alf,
    title = "{ALF} at {S}em{E}val-2024 Task 9: Exploring Lateral Thinking Capabilities of {LM}s through Multi-task Fine-tuning",
    author = "Farokh, Seyed Ali  and Zeinali, Hossein",
    booktitle = "Proceedings of the 18th International Workshop on Semantic Evaluation (SemEval-2024)",
    month = jun,
    year = "2024",
    address = "Mexico City, Mexico",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2024.semeval-1.218",
    doi = "10.18653/v1/2024.semeval-1.218",
    pages = "1523--1528",
}
```

## Contact
Seyed Ali Farokh: s.alifarrokh@gmail.com
