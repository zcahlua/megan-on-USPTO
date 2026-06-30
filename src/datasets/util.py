# -*- coding: utf-8 -*-
"""
Utility functions for datasets
"""
import gzip
import logging
import os
import shutil
import zipfile

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

logger = logging.getLogger(__name__)


def unzip_and_clean(archive_dir: str, file_name: str, delete_archive_after_extract: bool = True):
    archive_path = os.path.join(archive_dir, file_name)

    if file_name.endswith('.zip'):
        with zipfile.ZipFile(archive_path) as f:
            f.extractall(path=archive_dir)
    elif file_name.endswith('.gz'):
        output_path = archive_path.replace('.gz', '')
        with gzip.open(archive_path, 'rb') as f_in:
            with open(output_path, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
    else:
        raise ValueError(f'Unsupported archive format for file: {archive_path}')

    if delete_archive_after_extract:
        os.remove(archive_path)


def download_url(url, save_path, chunk_size=1024 * 1024, timeout=60, retries=3):
    """Download a URL with retries and clear errors for manual fallback."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    session = requests.Session()
    retry_kwargs = dict(total=retries, connect=retries, read=retries, status=retries,
                        backoff_factor=1, status_forcelist=(429, 500, 502, 503, 504))
    try:
        retry = Retry(**retry_kwargs, allowed_methods=("GET",))
    except TypeError:
        retry = Retry(**retry_kwargs, method_whitelist=("GET",))
    session.mount('http://', HTTPAdapter(max_retries=retry))
    session.mount('https://', HTTPAdapter(max_retries=retry))
    tmp_path = save_path + '.part'
    try:
        with session.get(url, stream=True, timeout=timeout, allow_redirects=True) as r:
            r.raise_for_status()
            total = int(r.headers.get('content-length', 0) or 0)
            logger.info(f"Downloading {url} to {save_path} ({total or 'unknown'} bytes)")
            bytes_written = 0
            with open(tmp_path, 'wb') as fd:
                for chunk in r.iter_content(chunk_size=chunk_size):
                    if chunk:
                        fd.write(chunk)
                        bytes_written += len(chunk)
        if total and bytes_written != total:
            raise IOError(f"Downloaded {bytes_written} bytes, expected {total} bytes")
        os.replace(tmp_path, save_path)
        logger.info(f"Downloaded {bytes_written} bytes to {save_path}")
    except Exception as exc:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise RuntimeError(
            f"Failed to download {url}. You can manually download it and place it at {save_path}. "
            f"Original error: {exc}"
        )
