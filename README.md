## FMFP - Official PyTorch Implementation
![](./framework.png)

### [ECCV2022] Rethinking Confidence Calibration for Failure Prediction
Fei Zhu, Zhen Cheng, Xu-Yao Zhang, Cheng-Lin Liu<br>

[Paper](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136850512.pdf)

### Abstract
Reliable confidence estimation for the predictions is important in many safety-critical applications. However, modern deep neural networks are often overconfident for their incorrect predictions. Recently, many calibration methods have been proposed to alleviate the overconfidence problem. With calibrated confidence, a primary and practical purpose is to detect misclassification errors by filtering out low-confidence predictions (known as failure prediction). In this paper, we find a general,
widely-existed but actually-neglected phenomenon that most confidence calibration methods are useless or harmful for failure prediction. We investigate this problem and reveal that popular confidence calibration methods often lead to worse confidence separation between correct and incorrect samples, making it more difficult to decide whether to trust a prediction or not. Finally, inspired by the natural connection between flat minima and confidence separation, we propose a simple hypothesis:
flat minima is beneficial for failure prediction. We verify this hypothesis via extensive experiments and further boost the performance by combining two different flat minima techniques.

### Usage 
We run the code with torch version: 1.10.0, python version: 3.9.7
* Baseline and other calibration methods
```
python main_base.py
```
* Flat minima for failure prediction
```
python main_fmfp.py
```
### DS340W extension: failure prediction under corruption severity

Adds a post-hoc Risk Advisor meta-model and evaluates both it and plain MSP
across CIFAR-10-C severity levels, to see whether a confidence estimator stays
reliable as the input distribution moves away from training.

Download CIFAR-10-C (2.9GB) from <https://zenodo.org/records/2535967> and
unpack it so that `./data/CIFAR-10-C/` holds `labels.npy` and the 15 corruption
`.npy` files.

```
python eval_shift.py --ckpt ./output/cifar10_resnet18_baseline/model.pth --tag baseline
python eval_shift.py --ckpt ./output_flat/cifar10_resnet18_fmfp/model.pth  --tag fmfp
python plot_shift.py
```

Both runs append to `./output_shift/results.csv`; `plot_shift.py` reads it and
writes `auroc_vs_severity.png`. Pass `--metric aurc` (or `eaurc`, `fpr95`,
`acc`) to plot a different column. `--corruptions gaussian_noise fog` limits the
sweep to a subset while testing.

New files:
* `risk_advisor.py` — gradient-boosted meta-model, fit on validation logits
* `utils/cifar10c.py` — CIFAR-10-C loader, one severity at a time
* `utils/fp_metrics.py` — AUROC / AURC / E-AURC / FPR95 for any confidence score
* `eval_shift.py`, `plot_shift.py` — sweep and figure

### Useful links
A list of papers that studies out-of-distribution (OOD) detection and misclassification detection (MisD)
* <https://github.com/Impression2805/Awesome-Failure-Detection>

### Reference
Our implementation references the codes in the following repositories:
* <https://github.com/daintlab/confidence-aware-learning>

### Contact for issues
Fei Zhu (zhufei2018@ia.ac.cn)
