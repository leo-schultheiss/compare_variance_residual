
import torch
import numpy as np
import os
import time as tm
from transformers import AutoModel, AutoTokenizer
import json

CACHE_DIR = "../cache"
DATA_DIR = "../data"
grid_dir = "../stimuli/grids"
trfiles_dir = "../stimuli/trfiles"
language_model_layers = 12

OVERWRITE = False

device = "cuda:0" if torch.cuda.is_available() else "cpu"

def load_model(model_name):
    with open('../text_model_config.json', 'r') as f:
        model_config = json.load(f)[model_name]
    model_hf_path = model_config['huggingface_hub']
    model = AutoModel.from_pretrained(model_hf_path, cache_dir=CACHE_DIR).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_hf_path, cache_dir=CACHE_DIR)
    model.eval()
    return model, model_config, tokenizer


@torch.inference_mode()
def get_model_layer_representations(model_name, text_array):
    model, model_config, tokenizer = load_model(model_name)

    # get the token embeddings
    token_embeddings = []
    for word in text_array:
        current_token_embedding = get_model_token_embeddings([word], tokenizer, model)
        token_embeddings.append(np.mean(current_token_embedding.detach().numpy(), 1))

    # layer-wise embeddings of particular length
    words_layers_representations = {}
    n_total_layers = model_config['num_layers']
    for layer in range(n_total_layers):
        words_layers_representations[layer] = []
    words_layers_representations[-1] = token_embeddings

    words_layers_representations = extract_context_representations(model, model_config, text_array, tokenizer,
                                                                   words_layers_representations)
    return words_layers_representations


def extract_context_representations(model, model_config, text_array, tokenizer, words_layers_representations):
    # Before we've seen enough words to make up the seq_len
    # Extract index 0 after supplying tokens 0 to 0, extract 1 after 0 to 1, 2 after 0 to 2, ... , 19 after 0 to 19
    start_time = tm.time()
    for seq_len in range(len(text_array)):
        if seq_len < sequence_length:
            word_seq = text_array[:seq_len + 1]
            extract_index = seq_len
        else:
            word_seq = text_array[seq_len - sequence_length + 1:seq_len + 1]
            extract_index = sequence_length - 1

        words_layers_representations = add_avrg_token_embedding_for_specific_word(word_seq, tokenizer, model,
                                                                                  extract_index,
                                                                                  words_layers_representations,
                                                                                  model_config)

        if seq_len % 100 == 0:
            print('Completed {} out of {}: {}'.format(seq_len, len(text_array), tm.time() - start_time))
            start_time = tm.time()
    print(f'Done extracting sequences of length {sequence_length}')
    return words_layers_representations


@torch.inference_mode()
def predict_model_embeddings(words_in_array, tokenizer, model, model_config):
    """
    extracts layer representations for all words in words_in_array
    :param encoded_layers: list of tensors, length num layers. each tensor of dims num tokens by num dimensions in representation
    :param word_ind_to_token_ind: dict that maps from index in words_in_array to index in array of tokens when words_in_array is tokenized,
                      with keys: index of word, and values: array of indices of corresponding tokens when word is tokenized
    """
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
    else:
        raise ValueError("model_type should be either encoder, decoder or encoder-decoder")

    return all_layers_hidden_states, word_ind_to_token_ind, None


@torch.inference_mode()
def add_word_model_embedding(model_dict, embeddings_to_add, token_inds_to_avrg, specific_layer=-1):
    """
    add the embeddings for a specific word in the sequence

    :param token_inds_to_avrg: indices of tokens in embeddings output to avrg
    """
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


@torch.inference_mode()
def add_avrg_token_embedding_for_specific_word(word_seq, tokenizer, model, from_start_word_ind_to_extract, model_dict,
                                               model_config):
    """
    predicts representations for specific word in input word sequence, ad adds to existing layer-wise dictionary

    :param word_seq: numpy array of words in input sequence
    :param tokenizer: Auto tokenizer
    :param model: Auto model
    :param from_start_word_ind_to_extract: the index of the word whose features to extract, INDEXED FROM START OF WORD_SEQ
    :param model_dict: where to save the extracted embeddings
    """
    word_seq = list(word_seq)
    all_sequence_embeddings, word_ind_to_token_ind, _ = predict_model_embeddings(word_seq, tokenizer, model,
                                                                                 model_config)
    token_inds_to_avrg = word_ind_to_token_ind[from_start_word_ind_to_extract]
    model_dict = add_word_model_embedding(model_dict, all_sequence_embeddings, token_inds_to_avrg)

    return model_dict


@torch.inference_mode()
def get_model_token_embeddings(words_in_array, tokenizer, model):
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

# Extract Context Representations

stories_path = "../stimuli/formatted"
sequence_length = 20
model_name = "bert-base"
representations_file = f"../{model_name}{sequence_length}.npy"

if os.path.exists(representations_file) and not OVERWRITE:
    print("File already exists, loading embeddings")
    stories_embeddings = np.load(representations_file, allow_pickle=True)
else:
    print("Extracting embeddings, either file does not exist or OVERWRITE set")
    stories_embeddings = {}
    for story in sorted(os.listdir(stories_path)):
        print(story)
        story_path = os.path.join(stories_path, story)
        words = open(story_path, 'r').read().strip().split('\n')
        embeddings = get_model_layer_representations(model_name, np.array(words))
        print(len(embeddings))
        stories_embeddings[story] = embeddings

    np.save(representations_file, stories_embeddings, allow_pickle=True)
print(f"Done extracting embeddings {type(stories_embeddings)}")

downsampled_semanticseqs_file = f"../{model_name}{sequence_length}_downsampled.npy"
if not OVERWRITE:
    if os.path.exists(downsampled_semanticseqs_file):
        print("File already exists")
        exit()
else:
    print("OVERWRITE set, extracting downsampled stimuli")

training_story_names = ['alternateithicatom', 'avatar', 'howtodraw', 'legacy',
                        'life', 'myfirstdaywiththeyankees', 'naked',
                        'odetostepfather', 'souls', 'undertheinfluence']
# Pstories are the test (or Prediction) stories (well, story), which we will use to test our models
prediction_story_names = ['wheretheressmoke']
all_story_names = training_story_names + prediction_story_names
print(all_story_names)

from compare_variance_residual.stimuli_utils.textgrid import TextGrid

grids = {}
for story in all_story_names:
    print(story)
    gridfile = [os.path.join(grid_dir, gf) for gf in os.listdir(grid_dir) if gf.startswith(story)][0]
    with open(gridfile) as f:
        grids[story] = TextGrid(f.read())

from compare_variance_residual.stimuli_utils.trfile import TRFile

trfiles = dict()

for story in all_story_names:
    try:
        trf = TRFile(os.path.join(trfiles_dir, "%s.report" % story))
        trfiles[story] = [trf]
    except Exception as e:
        print(e)

from compare_variance_residual.stimuli_utils.SemtanticModel import SemanticModel
from ridge_utils.dsutils import make_word_ds

# Make word and phoneme datasequences
wordseqs = make_word_ds(grids, trfiles)  # dictionary of {storyname : word DataSequence}
eng1000 = SemanticModel.load(os.path.join(DATA_DIR, "english1000sm.hf5"))

from ridge_utils.dsutils import make_semantic_model

# semanticseqs = dict()  # dictionary to hold projected stimuli {story name : projected DataSequence}
# for story in all_story_names:
#     semanticseqs[story] = make_semantic_model(wordseqs[story], [eng1000], [985])
story_filenames = ['alternateithicatom', 'avatar', 'howtodraw', 'legacy',
                   'life', 'myfirstdaywiththeyankees', 'naked',
                   'odetostepfather', 'souls', 'undertheinfluence', 'wheretheressmoke']
semanticseqs = dict()
for i in np.arange(len(all_story_names)):
    print(all_story_names[i])
    semanticseqs[all_story_names[i]] = []
    for layer in np.arange(language_model_layers):
        temp = make_semantic_model(wordseqs[all_story_names[i]], [eng1000], [985])
        temp.data = np.nan_to_num(stories_embeddings.item()[story_filenames[i]][layer])
        semanticseqs[all_story_names[i]].append(temp)

# Downsample stimuli
print("Downsampling stimuli")
interptype = "lanczos"  # filter type
window = 3  # number of lobes in Lanczos filter
downsampled_semanticseqs = dict()  # dictionary to hold downsampled stimuli
for story in all_story_names:
    downsampled_semanticseqs[story] = []
    for layer in np.arange(language_model_layers):
        temp = semanticseqs[story][layer].chunksums(interptype, window=window)
        downsampled_semanticseqs[story].append(temp)

# Save downsampled stimuli
print("Saving downsampled stimuli")
np.save(downsampled_semanticseqs_file, downsampled_semanticseqs, allow_pickle=True)

print("Done")