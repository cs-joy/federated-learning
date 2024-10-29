import os
import zipfile
import logging
import requests
import torchvision

logger = logging.getLogger(__name__)

__all__ = ['download_data']



URL = {
    'femnist': [
        'https://s3.amazonaws.com/nist-srd/SD19/by_class.zip',
        'https://s3.amazonaws.com/nist-srd/SD19/by_write.zip'
    ],
    'shakespeare': ['http://www.gutenberg.org/files/100/old/1994-01-100.zip'],
    'sent140': [
        'http://cs.stanford.edu/people/alecmgo/trainingandtestdata.zip',
        'http://nlp.stanford.edu/data/glove.6B.zip' # GloVe embedding for vocabularies
    ],
    'celeba': [ 
        '1_ee_0u7vcNLOfNLegJRHmolfH5ICW-XS', # Google Drive link ID
        '0B7EVK8r0v71pblRyaVFSWGxPY0U', # Google Drive link ID
        'https://cseweb.ucsd.edu/~weijian/static/datasets/celeba/img_align_celeba.zip' # img_align_celeba.zip
    ],
    'reddit': ['1ISzp69JmaIJqBpQCX-JJ8-kVyUns8M7o']  # Google Drive link ID
}
OPT = { # md5 checksum if direct URL link is provided, file name if Google Drive link ID is provided  
    'femnist': ['79572b1694a8506f2b722c7be54130c4', 'a29f21babf83db0bb28a2f77b2b456cb'],
    'shakespeare': ['b8d60664a90939fa7b5d9f4dd064a1d5'],
    'sent140': ['1647eb110dd2492512e27b9a70d5d1bc', '056ea991adb4740ac6bf1b6d9b50408b'],
    'celeba': ['identity_CelebA.txt', 'list_attr_celeba.txt', '00d2c5bc6d35e252742224ab0c1e8fcb'],
    'reddit': ['reddit_subsampled.zip']
}


def download_data(download_root, dataset_name):
    """
    Download data from Google Drive and extract if it is archived.
    """
    def _get_confirm_token(response):
        pass

    def _save_response_content(download_root, response):
        pass

    def _download_file_from_google_drive(download_root, file_name, identifier):
        pass