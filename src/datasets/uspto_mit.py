# -*- coding: utf-8 -*-
"""
A dataset containing 480k mapped reactions from USPTO. It was introduced by authors of: arxiv.org/abs/1709.04555
"""
import logging
import os

import numpy as np
import pandas as pd
from src.datasets import Dataset
from src.datasets.util import unzip_and_clean
from rdkit import RDLogger
from tqdm import tqdm

from src.datasets.util import download_url

logger = logging.getLogger(__name__)


class UsptoMit(Dataset):
    def __init__(self):
        super(UsptoMit, self).__init__()

    @property
    def meta_info(self) -> dict:
        return {'max_n_nodes': 200}

    @property
    def key(self) -> str:
        return 'uspto_mit'

    def _download(self):
        url = "https://github.com/wengong-jin/nips17-rexgen/raw/master/USPTO/data.zip"
        archive_file = 'data.zip'
        archive_path = os.path.join(self.feat_dir, archive_file)
        split_paths = [os.path.join(self.feat_dir, f'data/{split_key}.txt')
                       for split_key in ('train', 'valid', 'test')]

        if all(os.path.exists(split_path) for split_path in split_paths):
            logger.info(f"USPTO-MIT split files already exist in {os.path.join(self.feat_dir, 'data')}; "
                        f"skipping download and unzip")
            return

        if os.path.exists(archive_path) and os.path.getsize(archive_path) > 0:
            logger.info(f"Using existing USPTO-MIT archive at {archive_path}")
        else:
            logger.info(f"Downloading raw data from {url}")
            try:
                download_url(url, archive_path)
            except RuntimeError as exc:
                raise RuntimeError(
                    f"{exc} Manual fallback: download {url} and place it at {archive_path}, then rerun "
                    f"'python bin/acquire.py uspto_mit'."
                )

        unzip_and_clean(self.feat_dir, archive_file, delete_archive_after_extract=False)
        logger.info(f"Files downloaded and unpacked to {self.feat_dir}")

    def _preprocess(self):
        x = {
            'product': [],
            'substrates': [],
        }

        split = []
        meta = []
        split_keys = ['train', 'valid', 'test']

        # there is a warning about hydrogen atoms that do not have neighbors that could not be deleted (that is OK)
        RDLogger.DisableLog('rdApp.*')

        for split_i, split_key in enumerate(split_keys):
            split_path = os.path.join(self.feat_dir, f'data/{split_key}.txt')

            file_len = sum(1 for _ in open(split_path, 'r'))
            for line in tqdm(open(split_path, 'r'), desc=f'reading {split_key} reactions', total=file_len):
                split_line = line.split(' ')
                reaction = split_line[0]
                meta_info = split_line[1].strip()
                subs, prod = tuple(reaction.split('>>'))
                subs = subs.strip()
                prod = prod.strip()
                x['substrates'].append(subs)
                x['product'].append(prod)
                split.append(split_i)
                meta.append(meta_info)
            logger.info(f'Saved {file_len} {split_key} reactions')

        split = np.asarray(split, dtype=int)
        split_df = dict((k, (split == i).astype(int)) for i, k in enumerate(split_keys))

        meta = {
            'uspto_mit_split': split,
            'meta_info': meta
        }

        logger.info(f"Saving 'x' to {self.x_path}")
        pd.DataFrame(x).to_csv(self.x_path, sep='\t')

        logger.info(f"Saving {self.metadata_path}")
        pd.DataFrame(meta).to_csv(self.metadata_path, sep='\t')

        split_path = os.path.join(self.dir, 'default_split.csv')
        logger.info(f"Saving default split to {split_path}")
        pd.DataFrame(split_df).to_csv(split_path)

    def acquire(self):
        self._download()
        self._preprocess()
