import torch
import numpy as np
import os
import time as tm
from transformers import AutoModel, AutoTokenizer
import argparse
import json
import pandas as pd

CACHE_DIR = "./cache"

# Use GPU if possible
device = "cuda:0" if torch.cuda.is_available() else "cpu"


@torch.inference_mode()
def get_model_layer_representations(args, text_array, word_ind_to_extract):
    seq_len = args.sequence_length

    model, model_config, n_total_layers, tokenizer = load_model(args, seq_len)

    # get the token embeddings
    token_embeddings = extract_token_embeddings(model, text_array, tokenizer)

    # where to store layer-wise embeddings of particular length
    words_layers_representations = {}
    for layer in range(n_total_layers):
        words_layers_representations[layer] = []
    words_layers_representations[-1] = token_embeddings

    words_layers_representations = extract_sliding_window_context_representations(model, model_config, seq_len,
                                                                                  text_array, tokenizer,
                                                                                  word_ind_to_extract,
                                                                                  words_layers_representations)

    return words_layers_representations


def extract_sliding_window_context_representations(model, model_config, seq_len, text_array, tokenizer,
                                                   word_ind_to_extract, words_layers_representations):
    # Before we've seen enough words to make up the seq_len
    # Extract index 0 after supplying tokens 0 to 0, extract 1 after 0 to 1, 2 after 0 to 2, ... , 19 after 0 to 19
    # Then, use sequences of length seq_len, still adding the embedding of the last word in a sequence
    start_time = tm.time()
    for end_curr_seq in range(1, len(text_array)):
        if end_curr_seq <= seq_len:
            word_seq = text_array[:end_curr_seq]
            from_start_word_ind_to_extract = -1 + end_curr_seq
        else:
            word_seq = text_array[end_curr_seq - seq_len + 1:end_curr_seq + 1]
            if word_ind_to_extract < 0:
                from_start_word_ind_to_extract = seq_len + word_ind_to_extract
            else:
                from_start_word_ind_to_extract = word_ind_to_extract

        words_layers_representations = add_avrg_token_embedding_for_specific_word(word_seq, tokenizer, model,
                                                                                  from_start_word_ind_to_extract,
                                                                                  words_layers_representations,
                                                                                  model_config)

        if end_curr_seq % 100 == 0:
            print('Completed {} out of {}: {}'.format(end_curr_seq, len(text_array), tm.time() - start_time))
            start_time = tm.time()
    return words_layers_representations


def extract_token_embeddings(model, text_array, tokenizer):
    token_embeddings = []
    for word in text_array:
        current_token_embedding = get_model_token_embeddings([word], tokenizer, model)
        token_embeddings.append(np.mean(current_token_embedding.detach().numpy(), 1))
    return token_embeddings


def load_model(args, seq_len):
    with open('text_model_config.json', 'r') as f:
        model_config = json.load(f)[args.model]
        model_hf_path = model_config['huggingface_hub']
    print(model_config, model_hf_path, seq_len)
    n_total_layers = model_config['num_layers']
    model = AutoModel.from_pretrained(model_hf_path, cache_dir=CACHE_DIR).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_hf_path, cache_dir=CACHE_DIR)
    model.eval()
    return model, model_config, n_total_layers, tokenizer


# extracts layer representations for all words in words_in_array
# encoded_layers: list of tensors, length num layers. each tensor of dims num tokens by num dimensions in representation
# word_ind_to_token_ind: dict that maps from index in words_in_array to index in array of tokens when words_in_array is tokenized,
#                       with keys: index of word, and values: array of indices of corresponding tokens when word is tokenized
@torch.inference_mode()
def predict_model_embeddings(words_in_array, tokenizer, model, model_config):
    #     for word in words_in_array:
    #         if word in remove_chars:
    #             print('An input word is also in remove_chars. This word will be removed and may lead to misalignment. Proceed with caution.')
    #             return -1

    n_seq_tokens = 0
    seq_tokens = []

    word_ind_to_token_ind = {}  # dict that maps index of word in words_in_array to index of tokens in seq_tokens

    for i, word in enumerate(words_in_array):
        word_ind_to_token_ind[i] = []  # initialize token indices array for current word
        word_tokens = tokenizer.tokenize(word)

        for token in word_tokens:
            #             if token not in remove_chars:  # don't add any tokens that are in remove_chars
            seq_tokens.append(token)
            word_ind_to_token_ind[i].append(n_seq_tokens)
            n_seq_tokens = n_seq_tokens + 1

    # convert token to vocabulary indices
    indexed_tokens = tokenizer.convert_tokens_to_ids(seq_tokens)
    tokens_tensor = torch.tensor([indexed_tokens]).to(device)

    if model_config['model_type'] == 'encoder' or model_config['model_type'] == 'decoder':
        outputs = model(tokens_tensor, output_hidden_states=True)
        hidden_states = outputs['hidden_states'][1:]  # This is a tuple: (layer1, layer2, ..., layer6)
        all_layers_hidden_states = hidden_states
    elif model_config['model_type'] == 'encoder-decoder':
        outputs = model(tokens_tensor, decoder_input_ids=tokens_tensor, output_hidden_states=True)
        encoder_hidden_states = outputs['encoder_hidden_states'][1:]  # This is a tuple: (layer1, layer2, ..., layer6)
        decoder_hidden_states = outputs['decoder_hidden_states'][1:]
        all_layers_hidden_states = encoder_hidden_states + decoder_hidden_states

    return all_layers_hidden_states, word_ind_to_token_ind, None


# add the embeddings for a specific word in the sequence
# token_inds_to_avrg: indices of tokens in embeddings output to avrg
@torch.inference_mode()
def add_word_model_embedding(model_dict, embeddings_to_add, token_inds_to_avrg, specific_layer=-1):
    if specific_layer >= 0:  # only add embeddings for one specified layer
        layer_embedding = embeddings_to_add[specific_layer]
        full_sequence_embedding = layer_embedding.cpu().detach().numpy()
        model_dict[specific_layer].append(np.mean(full_sequence_embedding[0, token_inds_to_avrg, :], 0))
    else:
        for layer, layer_embedding in enumerate(embeddings_to_add):
            full_sequence_embedding = layer_embedding.cpu().detach().numpy()
            model_dict[layer].append(np.mean(full_sequence_embedding[0, token_inds_to_avrg, :],
                                             0))  # avrg over all tokens for specified word
    return model_dict


# predicts representations for specific word in input word sequence, and adds to existing layer-wise dictionary
#
# word_seq: numpy array of words in input sequence
# tokenizer: Auto tokenizer
# model: Auto model
# remove_chars: characters that should not be included in the represention when word_seq is tokenized
# from_start_word_ind_to_extract: the index of the word whose features to extract, INDEXED FROM START OF WORD_SEQ
# model_dict: where to save the extracted embeddings
@torch.inference_mode()
def add_avrg_token_embedding_for_specific_word(word_seq, tokenizer, model, from_start_word_ind_to_extract, model_dict,
                                               model_config):
    word_seq = list(word_seq)
    all_sequence_embeddings, word_ind_to_token_ind, _ = predict_model_embeddings(word_seq, tokenizer, model,
                                                                                 model_config)
    token_inds_to_avrg = word_ind_to_token_ind[from_start_word_ind_to_extract]
    model_dict = add_word_model_embedding(model_dict, all_sequence_embeddings, token_inds_to_avrg)

    return model_dict


# get the model token embeddings
@torch.inference_mode()
def get_model_token_embeddings(words_in_array, tokenizer, model):
    #     for word in words_in_array:
    #         if word in remove_chars:
    #             print('An input word is also in remove_chars. This word will be removed and may lead to misalignment. Proceed with caution.')
    #             return -1

    n_seq_tokens = 0
    seq_tokens = []

    word_ind_to_token_ind = {}  # dict that maps index of word in words_in_array to index of tokens in seq_tokens

    for i, word in enumerate(words_in_array):
        word_ind_to_token_ind[i] = []  # initialize token indices array for current word
        word_tokens = tokenizer.tokenize(word)

        for token in word_tokens:
            #             if token not in remove_chars:  # don't add any tokens that are in remove_chars
            seq_tokens.append(token)
            word_ind_to_token_ind[i].append(n_seq_tokens)
            n_seq_tokens = n_seq_tokens + 1

    # convert token to vocabulary indices
    indexed_tokens = tokenizer.convert_tokens_to_ids(seq_tokens)
    tokens_tensor = torch.tensor([indexed_tokens]).to(device)

    input_embedding_module = model.base_model.get_input_embeddings()
    token_embeddings = input_embedding_module(tokens_tensor.to(torch.long)).cpu()

    return token_embeddings


def load_words_from_file(file):
    if '.npy' in file:
        words = np.load(args.input_file)
    elif '.csv' in file:
        data_words = pd.read_csv(file, header=None)
        words = data_words[1]
    else:
        words = open(file, 'r').read().strip().split('\n')
    return words


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CheXpert NN argparser")
    parser.add_argument("--input_file", help="Choose inputfile", type=str, required=True)
    parser.add_argument("--model", help="Choose model", type=str, required=True)
    parser.add_argument("--sequence_length", help="Choose context", type=int)
    parser.add_argument("--output_file", help="Choose output filename", type=str)
    args = parser.parse_args()

    print(args.input_file)

    word_ind_to_extract = -1

    if os.path.isfile(args.input_file):
        words = load_words_from_file(args.input_file)
        extracted_features = get_model_layer_representations(args, np.array(words), word_ind_to_extract)
    elif os.path.isdir(args.input_file):
        stories_files = {}
        for story in sorted(os.listdir(args.input_file)):
            words = load_words_from_file(os.path.join(args.input_file, story))
            embeddings = get_model_layer_representations(args, np.array(words), word_ind_to_extract)
            stories_files[story] = embeddings
        extracted_features = stories_files
    else:
        raise ValueError("input_file should be an existing file or a directory")

    np.save(args.output_file, extracted_features)
