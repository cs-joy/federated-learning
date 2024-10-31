import os
import json
import logging

from collections import Counter, defaultdict

logger = logging.getLogger(__name__)



def preprocess(root):
    def _make_path(path, dir_name):
        if not os.path.exists(os.path.join(path, dir_name)):
            os.makedirs(os.path.join(path, dir_name))

    def _load_data(path):
        with open(path, 'r') as file:
            data = json.load(file)
        return data

    def _save_data(path, data):
        with open(path, 'w') as file:
            json.dump(data, file)

    def _refine_data(data):
        num_samples = []
        for user in data['users']:
            num_samples.append(len(data['user_data'][user]['x']))   # get correct sample counts
            data['user_data'][user]['y'] = [original if (type(original) is list) else original['target_tokens'] for original in data['user_data'][user]['y']]   # don't know why... but some samples are not parsed (i.e., in `dict` format, not `list`)
        else:
            data['num_samples'] = num_samples
        return data

    def _build_counter(train_data):
        all_tokens = []
        for u in train_data:
            for c in train_data[u]['x']:
                for s in c:
                    all_tokens.extend(s)
        
        counter = Counter()
        counter.update(all_tokens)

        return counter

    def _build_vocab(counter):
        vocab_size = 10000
        pad_symbol, unk_symbol, bos_symbol, eos_symbol = 0, 1, 2, 3
        count_pairs = sorted(counter.items(), key=lambda x: (-x[1], x[0]))
        count_pairs = count_pairs[:vocab_size - 1]

        words, _ = list(zip(*count_pairs))
        words = list(words)
        vocab = {}
        vocab['<PAD>'] = pad_symbol
        vocab['<UNK>'] = unk_symbol
        vocab['<BOS>'] = bos_symbol
        vocab['<EOS>'] = eos_symbol

        idx = 4 # due to special tokens
        while len(words) > 0:
            w = words.pop()
            if w in ['<PAD>', '<UNK>', '<BOS>', '<EOS>']:
                continue
            vocab[w] = idx
            idx += 1
        
        vocab = {'vocab': vocab, 'size': vocab_size, 'unk_symbol': unk_symbol, 'pad_symbol': pad_symbol, 'bos_symbol': bos_symbol, 'eos_symbol': eos_symbol}

        return vocab

    def _convert_to_ids_and_get_length(raw, vocab):
        def _tokens_to_word_ids(tokens, vocab):
            pass

        def _convert_to_id(container, key):
            pass

    # 