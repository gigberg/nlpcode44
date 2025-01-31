# HiTrans: A Transformer-Based Context- and Speaker-Sensitive Model for Emotion Detection in Conversations

Source code for COLING 2020 paper: [HiTrans: A Transformer-Based Context- and Speaker-Sensitive Model for Emotion Detection in Conversations](https://www.aclweb.org/anthology/2020.coling-main.370/)

## 1. Environments

- python (3.6.8)
- cuda (10.1)

## 2. Dependencies

- numpy (1.17.3)
- torch (1.6.0)
- transformers (3.5.1)
- pandas (1.1.5)
- scikit-learn (0.23.2)

## 3. Preparation

- Download [MELD](https://github.com/declare-lab/MELD) dataset
- Put `train_sent_emo.csv`, `dev_sent_emo.csv` and `test_sent_emo.csv` into the directory `data/`

## 4. Training

```bash
>> python main.py
```

## 5. Evaluation

```bash
>> python main.py --evaluate
```

## 6. License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 7. Citation

If you use this work or code, please kindly cite the following paper:

```bib
@inproceedings{li-etal-2020-hitrans,
    title = "{H}i{T}rans: A Transformer-Based Context- and Speaker-Sensitive Model for Emotion Detection in Conversations",
    author = "Li, Jingye  and
      Ji, Donghong  and
      Li, Fei  and
      Zhang, Meishan  and
      Liu, Yijiang",
    booktitle = "Proceedings of the 28th International Conference on Computational Linguistics",
    year = "2020",
    publisher = "International Committee on Computational Linguistics",
    url = "https://www.aclweb.org/anthology/2020.coling-main.370",
    pages = "4190--4200",
}
```

## 要做的实验
基本结构: QA_flow + UniAttenEncoder+ spkloss

1-5 QA_flow+spkloss:
dia_enc: "MultiUniAtten, MultiSpanRNN, MultidilaCNN, MultiUniGCN, DiaRemove"

FlowCellQA (0.8183)        0.6827错了     (最优0.8384)  (最差0.7232)  (0.2982错了)

FlowCellQQ  0.6811                        0.7053        0.6972(对话太短)

FlowCellQE  0.5100                              0.6107        0.6827

FlowCellQIntent

FlowRemove

2-4 UniAttenEncoder+ spkloss
flow_type: FlowCellQA, FlowCellQQ, FlowCellQE, FlowRemove

3-3QA_flow + UniAttenEncoder+ spkloss
-spkloss
-UniAttenEncoder
-position
