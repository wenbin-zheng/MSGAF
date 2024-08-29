# Multi-level Spatiotemporal Graph Attention Fusion for Multimodal Depression Detection



## Requirements

All our experiments are implemented based on the PyTorch framework with one 24G NVIDIA Geforce RTX 4090 GPUs, and we recommend installing the following package versions:

- Python=3.10
- PyTorch=2.3.1



## Dataset

The DAIC-WOZ and E-DAIC depression datasets were utilized in this thesis. Here is the official [link](https://dcapswoz.ict.usc.edu/), where you can send a request and receive your own username and password to access the database.



## Training

- **DIAC-WOZ**

```bash
cd experiments/
bash ./daicwoz/train-baseline-daicwoz-modality-ablation.sh
```

- **E-DAIC**

```bash
cd experiments/
bash ./edaic/train-baseline-edaic-modality-ablation.sh
```



## Evaluation

- **DIAC-WOZ**

```bash
cd experiments/
bash ./daicwoz/evaluate-baseline-daicwoz-modality-ablation.sh
```

- **E-DAIC**

``` bash
cd experiments/
bash ./edaic/evaluate-baseline-edaic-modality-ablation.sh
```



