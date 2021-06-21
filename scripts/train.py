"""Train company embedding using w2v"""
import configparser
import numpy as np
import random
import argparse
from datetime import datetime
from collections import defaultdict, Counter
from itertools import permutations
from tensorflow.keras.layers import Input, Embedding, Dot, Reshape, Dense
from tensorflow.keras.models import Model
from keras.optimizers import SGD
from tensorflow.random import set_seed
from tensorflow.keras.callbacks import TensorBoard

from utils import load_data, save_mapping, save_embedding, chunks

def extract_embedding(model: Model) -> (list, list):
    """
    Extract embeddings from Keras model, save embedding to disk
    :param model: Keras model
    :return: skill embedding and title embedding
    """
    # Extract company embeddings
    company_layer = model.get_layer('company_embedding')
    company_weights = company_layer.get_weights()[0]

    return company_weights

def generate_company_pairs_list(data: list, company_index: dict) -> (dict, list):
    """
    Generate company pairs list from raw data
    :param data: List of sets, raw json file
    :param company_index: Companies - indices
    :return pairs_count: pairs of company indices - freq
    """
    perm = [permutations(search, 2) for search in data]

    pairs_count = defaultdict(int)
    for iterate in perm:
        for pair in iterate:
            tmp_key = '-'.join([str(company_index[pair[0]]), str(company_index[pair[1]])])
            if not tmp_key in pairs_count.keys():
                pairs_count[tmp_key] = 1
            else:
                pairs_count[tmp_key] += 1
    companies_pairs = [tuple(int(x) for x in key.split('-')) for key in pairs_count.keys()]

    return pairs_count, companies_pairs

def generate_index(data: list) -> (dict, dict, dict):
    """
    Generate company-index, index-company pairs
    :param data: All possible companies
    :return company_index: company-index mapping
    :return index_company: index-company mapping
    """

    companies = []
    for search in data:
        companies += search

    company_index = {company: idx for idx, company in enumerate(set(companies))}
    index_company = {idx: company for company, idx in company_index.items()}

    companies_count = Counter(companies)
    companies_count = {company_index[k]: v for k,v in companies_count.items()}
    companies_count = {k:v for k,v in sorted(companies_count.items(), key = lambda item:item[0])}

    # Save mappings to disk
    save_mapping(company_index, '../data/company_index.pickle')
    save_mapping(index_company, '../data/index_company.pickle')
    save_mapping(companies_count, '../data/companies_count.pickle')

    return company_index, index_company, companies_count

def val_company_in_train(val_data: list, company_index: dict) -> list:
    val_companies = []
    for search in val_data:
        val_companies += search
        val_companies = list(set(val_companies))
    val_company_list = []
    for vc in val_companies:
        if vc in company_index.keys():
            val_company_list.append(vc)
    return val_company_list

def generate_val_company_pairs_list(val_data: list, val_company_list: list, companies_pairs: list, company_index: dict) -> list:
    """
    Generate validation company pairs list from val data
    :param val_data: List of validation user search
    :param val_company_list: List of target companies that in validation, also appear in training data
    :param companies_pairs: Positive pairs for training
    :return val_companies_pairs: companies pairs for validation
    """
    perm = [permutations(search, 2) for search in val_data if set(search) & set(val_company_list) != set()]
    val_companies_pairs = []
    for iterate in perm:
        for pair in iterate:
            if (pair[0] in val_company_list) & (pair[1] in val_company_list):
                val_companies_pairs.append(tuple([company_index[pair[0]], company_index[pair[1]]]))
    val_companies_pairs = set(val_companies_pairs) - set(companies_pairs)
    return val_companies_pairs

def name_embedding_model(company_index, pairs_count, lr=0.001, embedding_size=64):
    """
    Model to embed skills and titles using the functional API.
    Trained to discern if a title is present in a the skill
    :param skill_index: skill-index mapping
    :param title_index: title-index mapping
    :param embedding_size: embedding size
    :return: keras model
    """

    # Both inputs are 1-dimensional
    target_company = Input(name='target', shape=[1])
    object_company = Input(name='object', shape=[1])

    # Embedding the target company (shape will be (None, 1, 64))
    embedding = Embedding(name='company_embedding',
                                input_dim=len(company_index),
                                output_dim=embedding_size)

    # Embedding the object company (shape will be (None, 1, 64))
    target_embedding = embedding(target_company)
    object_embedding = embedding(object_company)

    # Merge the layers with a dot product along the second axis (shape will be (None, 1, 1))
    merged = Dot(name='dot_product', normalize=True, axes=2)([target_embedding, object_embedding])

    # Reshape to be a single number (shape will be (None, 1))
    merged = Reshape(target_shape=[1])(merged)

    # Add dense layer and loss function is binary cross entropy
    merged = Dense(1, activation='sigmoid')(merged)
    model = Model(inputs=[target_company, object_company], outputs=merged)

    model.compile(optimizer=SGD(learning_rate=lr), loss='categorical_crossentropy', metrics=['accuracy'])

    return model

def generate_batch(companies_pairs: list, index_company: dict, pairs_count: dict, companies_count:dict,
                   n_positive: int = 1, n_negative: int = 5, seed: int = 100):
    """
    Generate batches of samples for training
    :param companies_pairs: Positive pairs
    :param index_company: index-company mapping
    :param pairs_count: weights for each positive pair
    :param n_positive:
    :param n_negative:
    :param seed:
    :return:
    """
    random.seed(seed)
    np.random.seed(seed)
    set_seed(seed)
    batch_size = int(n_positive * (1 + n_negative))
    batch = np.zeros((batch_size, 4))

    total_count = sum(companies_count.values())
    companies_freq = {k: v / total_count for k,v in companies_count.items()}
    
    
    for positive_per_batch in chunks(companies_pairs, n_positive):
        # This creates a generator
        idx = 0
        for (target_company, object_company) in positive_per_batch:
            weight_of_pair = pairs_count['-'.join([str(target_company), str(object_company)])]
            batch[idx, :] = (target_company, object_company, 1, weight_of_pair)
            # Increment idx by 1
            idx += 1
            # Add negative examples until reach batch size
            k = 0
            while k < n_negative:
                # negative selection based on unigram ^ (3/4)
                negative_company = random.choices(population=range(len(index_company)), weights=[x**(3/4) for x in companies_freq.values()], k=1)[0]
                # Check to make sure this is not a positive example
                if (target_company, negative_company) not in companies_pairs:
                    k += 1
                    weight_of_pair = pairs_count['-'.join([str(target_company), str(object_company)])]
                    batch[idx, :] = (target_company, negative_company, 0, weight_of_pair)
                    idx += 1

        # Make sure to shuffle order
        np.random.shuffle(batch)
        yield {'target': batch[:, 0], 'object': batch[:, 1]}, batch[:, 2], batch[:, 3]

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-config', type=str, help='Name of the configuration file')
    return parser.parse_args()

def train(args) -> list:
    """
    Train the embedding model, save the company embedding
    :return: company embedding.
    """
    # load configuration from file
    config = configparser.ConfigParser()
    config.read(args.config)
    # load {company: companies appear at the same time}
    data = load_data(config['DEFAULT']['data_path'])
    val_data = load_data(config['DEFAULT']['val_data'])

    # Generate index mappings
    company_index, index_company, companies_count = generate_index(data)
    
    # Generate company pairs with counts
    pairs_count, companies_pairs = generate_company_pairs_list(data, company_index)
    
    # validation company list
    val_company_list = val_company_in_train(val_data, company_index)
    val_companies_pairs = generate_val_company_pairs_list(val_data, val_company_list, companies_pairs, company_index)

    # Instantiate model and show parameters
    model = name_embedding_model(company_index, pairs_count, lr=float(config['DEFAULT']['learning_rate']), embedding_size=int(config['DEFAULT']['embedding_size']))
    print(model.summary())
    n_positive = int(config['DEFAULT']['n_positive'])
    gen = generate_batch(companies_pairs, index_company, pairs_count, companies_count, n_positive,
                         n_negative=int(config['DEFAULT']['n_negative']),
                         seed=int(config['DEFAULT']['random_seed']))
    
    # Train
    epochs = int(config['DEFAULT']['epochs'])
    log_dir = "../logs/fit/" + datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = TensorBoard(log_dir="logs")
    
    model.fit(gen, epochs=epochs, validation_data=val_companies_pairs,
                        steps_per_epoch=len(companies_pairs) // n_positive,
                        verbose=int(config['DEFAULT']['verbose']), callbacks=[tensorboard_callback])

    company_weights = extract_embedding(model)

    # save embeddings to disk.
    save_embedding(company_weights, '../data/company_weights_' + args.config.split('.')[0] + '.npy')

    return company_weights

if __name__ == '__main__':
    args = parse_args()
    train(args)
