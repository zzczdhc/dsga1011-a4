import os
from functools import lru_cache

from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

from transformers import T5TokenizerFast
import torch

PAD_IDX = 0
T5_CHECKPOINT = 'google-t5/t5-small'
TASK_PREFIX = 'translate English to SQL: '


@lru_cache(maxsize=1)
def get_t5_tokenizer():
    return T5TokenizerFast.from_pretrained(T5_CHECKPOINT)

class T5Dataset(Dataset):

    def __init__(self, data_folder, split):
        '''
        Skeleton for the class for performing data processing for the T5 model.

        Some tips for implementation:
            * You should be using the 'google-t5/t5-small' tokenizer checkpoint to tokenize both
              the encoder and decoder output. 
            * You want to provide the decoder some beginning of sentence token. Any extra-id on the
              T5Tokenizer should serve that purpose.
            * Class behavior should be different on the test set.
        '''
        self.examples = self.process_data(data_folder, split, get_t5_tokenizer())

    def process_data(self, data_folder, split, tokenizer):
        texts = load_lines(os.path.join(data_folder, f'{split}.nl'))
        inputs = tokenizer([TASK_PREFIX + text for text in texts])['input_ids']
        if split == 'test':
            return [(torch.tensor(ids), None) for ids in inputs]

        queries = load_lines(os.path.join(data_folder, f'{split}.sql'))
        if len(texts) != len(queries):
            raise ValueError(f'Input and SQL counts differ for {split}')
        targets = tokenizer(text_target=queries)['input_ids']
        return [(torch.tensor(x), torch.tensor(y)) for x, y in zip(inputs, targets)]
    
    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]

def normal_collate_fn(batch):
    '''
    Collation function to perform dynamic padding for training and evaluation with the
    development or validation set.

    Inputs:
        * batch (List[Any]): batch is a list of length batch_size, where each index contains what
                             the dataset __getitem__ function returns.

    Returns: To be compatible with the provided training loop, you should be returning
        * encoder_ids: The input ids of shape BxT to be fed into the T5 encoder.
        * encoder_mask: Mask of shape BxT associated with padding tokens in the encoder input
        * decoder_inputs: Decoder input ids of shape BxT' to be fed into T5 decoder.
        * decoder_targets: The target tokens with which to train the decoder (the tokens following each decoder input)
        * initial_decoder_inputs: The very first input token to be decoder (only to be used in evaluation)
    '''
    encoder_ids, encoder_mask, initial_decoder_inputs = test_collate_fn(batch)
    targets = pad_sequence([target for _, target in batch], batch_first=True, padding_value=PAD_IDX)
    decoder_inputs = torch.cat((initial_decoder_inputs, targets[:, :-1]), dim=1)
    return encoder_ids, encoder_mask, decoder_inputs, targets, initial_decoder_inputs

def test_collate_fn(batch):
    '''
    Collation function to perform dynamic padding for inference on the test set.

    Inputs:
        * batch (List[Any]): batch is a list of length batch_size, where each index contains what
                             the dataset __getitem__ function returns.

    Recommended returns: 
        * encoder_ids: The input ids of shape BxT to be fed into the T5 encoder.
        * encoder_mask: Mask of shape BxT associated with padding tokens in the encoder input
        * initial_decoder_inputs: The very first input token to be decoder (only to be used in evaluation)
    '''
    encoder_ids = pad_sequence([source for source, _ in batch], batch_first=True, padding_value=PAD_IDX)
    encoder_mask = encoder_ids.ne(PAD_IDX).long()
    initial_decoder_inputs = torch.full((len(batch), 1), PAD_IDX, dtype=torch.long)
    return encoder_ids, encoder_mask, initial_decoder_inputs

def get_dataloader(batch_size, split):
    data_folder = 'data'
    dset = T5Dataset(data_folder, split)
    shuffle = split == "train"
    collate_fn = normal_collate_fn if split != "test" else test_collate_fn

    dataloader = DataLoader(dset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)
    return dataloader

def load_t5_data(batch_size, test_batch_size):
    train_loader = get_dataloader(batch_size, "train")
    dev_loader = get_dataloader(test_batch_size, "dev")
    test_loader = get_dataloader(test_batch_size, "test")
    
    return train_loader, dev_loader, test_loader


def load_lines(path):
    with open(path, 'r') as f:
        lines = f.readlines()
        lines = [line.strip() for line in lines]
    return lines

def load_prompting_data(data_folder):
    names = ('train.nl', 'train.sql', 'dev.nl', 'dev.sql', 'test.nl')
    return tuple(load_lines(os.path.join(data_folder, name)) for name in names)
