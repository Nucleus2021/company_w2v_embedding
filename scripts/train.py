"""Train company embedding using w2v
    Usage:  python3 train.py -config train_config_long_competitors.ini
    Author: Changpeng Lu
"""

import configparser
import numpy as np
import random
import argparse
import tensorflow as tf
from itertools import permutations
import ujson
import numpy as np
import pickle
from datetime import datetime
from tensorflow.keras.layers import Input, Embedding, Dot, Reshape, Dense, Layer
from tensorflow.keras.models import Model
from keras.optimizers import SGD, Adam
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.random import set_seed
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint

from gen_utils import *

def company_embedding_w2v_model(company_index, lr=0.01, embedding_size=64, seed=1):
    """
    Model to embed skills and titles using the functional API.
    Trained to discern if a title is present in a the skill
    :param skill_index: skill-index mapping
    :param title_index: title-index mapping
    :param embedding_size: embedding size
    :return: keras model
    """

    random.seed(seed)
    np.random.seed(seed)
    set_seed(seed)

    # Both inputs are 1-dimensional
    target_company = Input(name='target', shape=[1])
    object_company = Input(name='object', shape=[1])

    # Embedding the target company (shape will be (None, 1, 64))
    input_embedding = Embedding(name='company_embedding',
                                input_dim=len(company_index),
                                output_dim=embedding_size)
    output_embedding = Embedding(name='weight',
                                input_dim=len(company_index),
                                output_dim=embedding_size)

    # Embedding the object company (shape will be (None, 1, 64))
    target_embedding = input_embedding(target_company)
    object_embedding = output_embedding(object_company)

    # Merge the layers with a dot product along the second axis (shape will be (None, 1, 1))
    merged = Dot(name='dot_product', normalize=True, axes=2)([target_embedding, object_embedding])

    # Reshape to be a single number (shape will be (None, 1))
    merged = Reshape(target_shape=[1])(merged)

    # Add dense layer and loss function is binary cross entropy
    merged = Dense(1, activation='sigmoid')(merged)
    model = Model(inputs=[target_company, object_company], outputs=merged)

    model.compile(optimizer=SGD(learning_rate=lr), loss='binary_crossentropy', metrics=['accuracy'])

    return model

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
#     val_data = load_data(config['DEFAULT']['val_data'])
    if config['DEFAULT']['whole_data'] != 'None':
        whole_data = load_data(config['DEFAULT']['whole_data'])
        tag_whole = True
    else:
        tag_whole = False
    output_name = config['DEFAULT']['output_name']
    # Generate index mappings
    load_path = config['DEFAULT']['load_path']
    if load_path == 'None':
        company_index, index_company, companies_count = generate_index(data, output_name)
    else:
        company_index = pkl.load(open(os.path.join(load_path, 'company_index_' + output_name + '.pickle'), 'rb'))
        index_company = pkl.load(open(os.path.join(load_path, 'index_company_' + output_name + '.pickle'), 'rb'))
        companies_count = pkl.load(open(os.path.join(load_path, 'companies_count_' + output_name + '.pickle'), 'rb'))


    # Generate company pairs with counts
    pairs_count, companies_pairs = generate_company_pairs_list(data, company_index)

    # validation company list
#     val_company_list = val_company_in_train(val_data, company_index)
#     val_companies_pairs = generate_val_company_pairs_list(val_data, val_company_list, companies_pairs, company_index)

    checkpoint_path = config['DEFAULT']['checkpoint_path']
    # Instantiate model and show parameters
    if checkpoint_path == 'None':
        model = company_embedding_w2v_model(company_index, lr=float(config['DEFAULT']['learning_rate']), embedding_size=int(config['DEFAULT']['embedding_size']), seed=int(config['DEFAULT']['random_seed']))
    else:
        model = tf.keras.models.load_model(checkpoint_path)

    print(model.summary())
    n_positive = int(config['DEFAULT']['n_positive'])

    num_positives = int(config['DEFAULT']['num_positives'])
    if num_positives != 0:
        if load_path == 'None':
            companies_pairs = random.choices(population=companies_pairs, weights=pairs_count.values(), k=num_positives)
            save_mapping(companies_pairs, 'data/companies_pairs_filtered_' + output_name + '.pickle')
        else:
            companies_pairs = pkl.load(open(os.path.join(load_path, 'companies_pairs_filtered_' + output_name + '.pickle'), 'rb'))
    else:
        companies_pairs = companies_pairs
    if tag_whole == True:
        gen = generate_batch(companies_pairs, index_company, pairs_count, companies_count,
                         whole_data, n_positive,
                         n_negative=int(config['DEFAULT']['n_negative']),
                         seed=int(config['DEFAULT']['random_seed']))
    else:
        gen = generate_batch_clean(companies_pairs, index_company, pairs_count, companies_count,
                         n_positive,
                         n_negative=int(config['DEFAULT']['n_negative']),
                         seed=int(config['DEFAULT']['random_seed']))

    # Train
    epochs = int(config['DEFAULT']['epochs'])
    log_dir = "logs/fit/" + datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = TensorBoard(log_dir="logs")
    weights_callback = ModelCheckpoint('logs/checkpoint_{epoch: 02d}_{accuracy:.2f}.hdf5', monitor='accuracy')
    hist=model.fit(gen, epochs=epochs,
                steps_per_epoch=len(companies_pairs) // n_positive,
                verbose=int(config['DEFAULT']['verbose']), callbacks=[weights_callback, tensorboard_callback]) # validation_data=val_companies_pairs,

    company_weights = extract_embedding(model)

    # save embeddings to disk.
    save_embedding(company_weights, 'data/company_weights_' + output_name + '.npy')

    return hist, company_weights, index_company

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-config', type=str, help='Name of the configuration file')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    hist, company_weights, index_company = train(args)
