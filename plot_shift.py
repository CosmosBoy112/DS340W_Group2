import argparse
import csv
from collections import defaultdict

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

STYLE = {
    ('baseline', 'msp'): ('#4c72b0', '--', 'o', 'Baseline + MSP'),
    ('baseline', 'risk_advisor'): ('#4c72b0', '-', 's', 'Baseline + Risk Advisor'),
    ('fmfp', 'msp'): ('#c44e52', '--', 'o', 'FMFP + MSP'),
    ('fmfp', 'risk_advisor'): ('#c44e52', '-', 's', 'FMFP + Risk Advisor'),
}


def read(path):
    series = defaultdict(dict)
    with open(path) as f:
        for row in csv.DictReader(f):
            key = (row['tag'], row['scorer'])
            series[key][int(row['severity'])] = row
    return series


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--csv', default='./output_shift/results.csv')
    p.add_argument('--metric', default='auroc', choices=['auroc', 'aurc', 'eaurc', 'fpr95', 'acc'])
    p.add_argument('--out', default='./output_shift/auroc_vs_severity.png')
    args = p.parse_args()

    series = read(args.csv)
    fig, ax = plt.subplots(figsize=(6, 4.2))

    for key, points in sorted(series.items()):
        sev = sorted(points)
        vals = [float(points[s][args.metric]) for s in sev]
        color, ls, marker, label = STYLE.get(key, (None, '-', 'o', '{} + {}'.format(*key)))
        ax.plot(sev, vals, color=color, linestyle=ls, marker=marker,
                markersize=5, linewidth=1.8, label=label)

    ax.set_xlabel('Corruption severity  (0 = clean CIFAR-10)')
    ax.set_ylabel(args.metric.upper())
    ax.set_xticks(range(6))
    ax.grid(alpha=0.3, linewidth=0.6)
    ax.legend(frameon=False, fontsize=9)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    fig.tight_layout()
    fig.savefig(args.out, dpi=200)
    print('wrote', args.out)


if __name__ == '__main__':
    main()
