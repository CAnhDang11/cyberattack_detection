"""Readable per-run logs, including reconstruction from saved artifacts."""
import argparse
import json
import logging
import sys
from pathlib import Path

import pandas as pd

LOGGER_NAME = 'unsw.training'


def setup(output):
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    for handler in logger.handlers[:]:
        handler.close()
        logger.removeHandler(handler)
    formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s')
    for handler in (logging.FileHandler(out / 'training.log', mode='a', encoding='utf-8'),
                    logging.StreamHandler(sys.stdout)):
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    logger.info('==================== NEW RUN ====================')
    return logger


def close():
    logger = logging.getLogger(LOGGER_NAME)
    for handler in logger.handlers[:]:
        handler.close()
        logger.removeHandler(handler)


def backfill(root):
    """Render historical results without claiming to have captured runtime output."""
    created = []
    for path in sorted(Path(root).rglob('results.json')):
        out = path.parent
        target = out / 'training.log'
        if target.exists():
            continue
        sections = ['RECONSTRUCTED LOG from saved artifacts; original runtime timestamps unavailable.']
        config = out / 'config.json'
        if config.exists():
            sections += ['CONFIG', config.read_text()]
        history = out / 'history.csv'
        if history.exists():
            sections += ['SAVED EPOCH/ROUND HISTORY', pd.read_csv(history).to_string(index=False)]
        sections += ['FINAL TEST RESULTS', path.read_text()]
        target.write_text('\n\n'.join(sections) + '\n', encoding='utf-8')
        created.append(target)
    return created


def main():
    parser = argparse.ArgumentParser(description='Create missing training.log files from saved results')
    parser.add_argument('root', nargs='?', default='experiments')
    args = parser.parse_args()
    for path in backfill(args.root):
        print(path)


if __name__ == '__main__':
    main()
