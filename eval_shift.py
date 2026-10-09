import argparse
import csv
import os

import numpy as np
import torch

from model import resnet, resnet18, densenet_BC, vgg, mobilenet, efficientnet, wrn, convmixer
from risk_advisor import RiskAdvisor
import cifar10c
from utils import data as dataset
from fp_metrics import evaluate, msp

FIELDS = ['tag', 'severity', 'scorer', 'acc', 'auroc', 'aupr_err', 'aurc', 'eaurc', 'fpr95']


def build_model(name, num_classes):
    if name == 'resnet18':
        return resnet18.ResNet18(num_classes=num_classes)
    if name == 'res110':
        return resnet.resnet110(num_classes=num_classes)
    if name == 'dense':
        return densenet_BC.DenseNet3(depth=100, num_classes=num_classes, growth_rate=12,
                                     reduction=0.5, bottleneck=True, dropRate=0.0)
    if name == 'vgg':
        return vgg.vgg16(num_classes=num_classes)
    if name == 'wrn':
        return wrn.WideResNet(28, num_classes, 10)
    if name == 'efficientnet':
        return efficientnet.efficientnet(num_classes=num_classes)
    if name == 'mobilenet':
        return mobilenet.mobilenet(num_classes=num_classes)
    if name == 'cmixer':
        return convmixer.ConvMixer(256, 16, kernel_size=8, patch_size=1, n_classes=num_classes)
    raise ValueError(name)


def load_weights(model, path):
    sd = torch.load(path, map_location='cpu')
    sd = {k[len('module.'):] if k.startswith('module.') else k: v
          for k, v in sd.items() if k != 'n_averaged'}
    model.load_state_dict(sd)
    return model


@torch.no_grad()
def collect(model, loader, device):
    model.eval()
    logits, correct = [], []
    for x, y, _ in loader:
        out = model(x.to(device))
        pred = out.argmax(1).cpu()
        logits.append(out.cpu().numpy())
        correct.append(pred.eq(torch.as_tensor(y)).numpy().astype(int))
    return np.concatenate(logits), np.concatenate(correct)


def main():
    p = argparse.ArgumentParser(description='Failure prediction under corruption severity')
    p.add_argument('--ckpt', required=True, help='path to model.pth')
    p.add_argument('--tag', required=True, help='label for this checkpoint, e.g. baseline or fmfp')
    p.add_argument('--model', default='resnet18')
    p.add_argument('--data', default='cifar10')
    p.add_argument('--data_path', default='./data/')
    p.add_argument('--cifar10c_path', default='./data/CIFAR-10-C/')
    p.add_argument('--batch_size', default=128, type=int)
    p.add_argument('--num_workers', default=4, type=int)
    p.add_argument('--corruptions', nargs='*', default=None,
                   help='subset of corruption names, default all 15')
    p.add_argument('--out', default='./output_shift/results.csv')
    p.add_argument('--gpu', default='0')
    args = p.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    num_classes = 100 if args.data == 'cifar100' else 10

    model = build_model(args.model, num_classes)
    load_weights(model, args.ckpt)
    model.to(device)

    _, valid_loader, test_loader, _, _ = dataset.get_loader(
        args.data, args.data_path, args.batch_size, args)

    valid_logits, valid_correct = collect(model, valid_loader, device)
    print('valid acc {:.2f}  errors {}'.format(100 * valid_correct.mean(),
                                               int((1 - valid_correct).sum())))

    advisor = RiskAdvisor().fit(valid_logits, valid_correct)

    rows = []
    loaders = [(0, test_loader)] + [
        (s, cifar10c.get_loader(args.cifar10c_path, s, args.batch_size,
                                args.corruptions, args.num_workers))
        for s in range(1, 6)
    ]

    for severity, loader in loaders:
        logits, correct = collect(model, loader, device)
        for scorer, score in [('msp', msp(logits)), ('risk_advisor', advisor.score(logits))]:
            row = {'tag': args.tag, 'severity': severity, 'scorer': scorer}
            row.update({k: round(v, 2) for k, v in evaluate(score, correct).items()})
            rows.append(row)
            print('sev {}  {:<13} acc {acc:.2f}  auroc {auroc:.2f}  '
                  'aurc {aurc:.2f}  fpr95 {fpr95:.2f}'.format(severity, scorer, **row))

    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    new = not os.path.exists(args.out)
    with open(args.out, 'a', newline='') as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new:
            w.writeheader()
        w.writerows(rows)
    print('wrote', args.out)


if __name__ == '__main__':
    main()
