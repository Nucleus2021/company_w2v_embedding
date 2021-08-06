"""general utility functions by Sherly"""
import configparser
import numpy as np
import pandas as pd
import random
from collections import defaultdict, Counter
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import pickle as pkl
import ujson as json
from itertools import permutations
import networkx as nx
import os
import tensorflow as tf
from collections import Counter, defaultdict
from sklearn.metrics.pairwise import cosine_similarity
from tensorflow.keras.models import Model
from tensorflow.random import set_seed
#from gensim.models.callbacks import CallbackAny2Vec

def read_json(filename, data_path):
    fp = open(os.path.join(data_path, filename),)
    data = json.load(fp)
    fp.close()
    return data

def str_to_list(industry, string='[]'):
    if industry == string:
        return []
    else:
        return [x.strip("' \"") for x in industry.split(string[0])[1].split(string[1])[0].split(',')]
    
def save_mapping(mapping: dict, path: str):
    """
    Save mapping dict to file.
    :param mapping: Mapping dictionary.
    :param path: File Location.
    :return:
    """
    with open(path, 'wb') as handle:
        pkl.dump(mapping, handle)
        
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

def finalize_embedding(load_path: str, index_company_path: str, final_name: str):
    """
    Extract company embedding to match the input format of the two-tower model, save embedding to ML_engine_meta
    load_path: data path of the keras model
    index_company_path: data path that saves dictionary of index - company mapping
    final_name: specific name given to the embedding JSON file
    """
    weights = load_weight(load_path,'company_embedding')
    index_company = pkl.load(open(index_company_path, 'rb'))
    tmp = {index_company[int(i)]: weights[i].tolist() for i in index_company.keys()}
    json.dump(tmp, open('/home/sherlylu/ML-Engine_Meta/data/embedding/cpy/weights_' + final_name + '.json','w'))
    
def save_embedding(embedding: np.array, path: str):
    """
    Save embedding to file
    :param embedding: embedding to be saved
    :param path: location to save to
    :return:
    """
    np.save(path, embedding)

def rename_checkpoint(list_path, max_epoch, last_epoch):
    os.chdir(list_path)
    filename = []
    for i in range(1, max_epoch+1):
        filename.append('checkpoint_ ' + str(i) + '_1.00.hdf5')
    for file in reversed(filename):
#     if file.split('.')[-1] = 'hdf5':
        num = file.split('_')[1].strip()
        try:
            os.rename(file, 'checkpoint_ ' + str(int(num)+last_epoch) + '_1.00.hdf5')
        except FileNotFoundError:
            continue
            
def generate_index(data: list, output_name: str) -> (dict, dict, dict):
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
#     save_mapping(skills, 'data/skill_count.pickle')
#     save_mapping(titles, 'data/title_count.pickle')
    save_mapping(company_index, 'data/company_index_' + output_name + '.pickle')
    save_mapping(index_company, 'data/index_company_' + output_name + '.pickle')
    save_mapping(companies_count, 'data/companies_count_' + output_name + '.pickle')

    return company_index, index_company, companies_count

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

def generate_searchA_companyB_pairs_list(data: list, company_index: dict) -> (dict, list):
    """
    Generate company pairs list from raw data
    Raw data format: List of [searchA, companyB]
    :param data: List of sets, raw json file
    :param company_index: Companies - indices
    :return pairs_count: pairs of company indices - freq
    """

    pairs_count = defaultdict(int)
    for pair in data:
        tmp_key = '-'.join([str(company_index[pair[0]]), str(company_index[pair[1]])])
        if not tmp_key in pairs_count.keys():
            pairs_count[tmp_key] = 1
        else:
            pairs_count[tmp_key] += 1
    companies_pairs = [tuple(int(x) for x in key.split('-')) for key in pairs_count.keys()]

    return pairs_count, companies_pairs

def generate_val_company_pairs_list(val_data: list, company_index: dict) -> list: # val_company_list: list, 
    """
    Generate validation company pairs list from val data
    :param val_data: List of validation user search
    :param val_company_list: List of target companies that in validation, also appear in training data
    :param companies_pairs: Positive pairs for training
    :return val_companies_pairs: companies pairs for validation
    """
    perm = [permutations(search, 2) for search in val_data] #if set(search) & set(val_company_list) != set()]
    val_companies_pairs = []
    for iterate in perm:
        for pair in iterate:
            #if (pair[0] in val_company_list) & (pair[1] in val_company_list):
            val_companies_pairs.append(tuple([company_index[pair[0]], company_index[pair[1]]]))
    val_companies_pairs = np.array(list(val_companies_pairs)) #- set(companies_pairs)
    counts = defaultdict(int)
    for pair in val_companies_pairs:
        key = '-'.join([str(pair[0]), str(pair[1])])
        if not key in counts.keys():
            counts[key] = 1
        else:
            counts[key] += 1
        #counts.append(pairs_count[key])
    val_companies_pairs = np.array([[int(y) for y in x.split('-')] for x in counts.keys()])
    counts = np.array(list(counts.values()))
    y = np.ones(len(counts))
    return {'target': val_companies_pairs[:, 0], 'object': val_companies_pairs[:, 1]}, y.reshape(-1,1), counts.reshape(-1,1)

# def val_company_in_train(val_data: list, company_index: dict) -> list:
#     val_companies = []
#     for search in val_data:
#         val_companies += search
#         val_companies = list(set(val_companies))
#     val_company_list = []
#     for vc in val_companies:
#         if vc in company_index.keys():
#             val_company_list.append(vc)
#     return val_company_list

def train_val_split(data, val_prop):
    val_inds = np.random.choice(np.arange(0,len(data)), size=int((val_prop+0.01)*len(data)), replace=False)
    val_data = np.array(data,dtype=object)[val_inds]
    train_inds = list(set(range(len(data))) - set(list(val_inds)))
    train_data = np.array(data,dtype=object)[train_inds].tolist()
    companies = get_companies(data)
    train_companies = get_companies(train_data)
    tmp = set(companies) - set(train_companies)
    val_search_ind = []
    np.random.shuffle(val_data)
    for i,search in enumerate(val_data):
        if set(tmp) & set(search) != set():
            val_search_ind.append(i)
            tmp = list(set(tmp) - (set(tmp) & set(search)))
        continue
    final_train_data = train_data + val_data[val_search_ind].tolist()
    final_val_data = []
    final_val_ind = list(set(range(len(val_data))) - set(val_search_ind))
    for i in final_val_ind:
        final_val_data.append(val_data[i].tolist())
    print('Train:Validation: {0}/{1}'.format(len(final_train_data)/len(data), len(final_val_data)/len(data)))
    return final_train_data, final_val_data

def positive_pairs(corpus1, window_size, output_name):
    company_index, _, _ = generate_index(corpus1, output_name)
    positive_pairs = []
    windows = []
    for sentence in corpus1:
        for i, word in enumerate(sentence):
            center = word
            first_word = i - window_size
            if first_word < 0 :
                first_word = 0
            end_word = i + window_size
            if end_word >= len(sentence):
                end_word = len(sentence) - 1
            for context in sentence[first_word : end_word + 1]:
                if context != center:
                    positive_pairs.append((company_index[center], company_index[context]))
    return company_index, positive_pairs

def get_companies(data):
    companies = []
    for search in data:
        companies += search
    companies = list(set(companies))
    return companies

def chunks(l, n):
    n = max(1, n)
    return (l[i:i+n] for i in range(0, len(l), n))

def load_weight(checkpoint_path, layer_name):
    model = tf.keras.models.load_model(checkpoint_path)
    company_layer = model.get_layer(layer_name)
    return company_layer.get_weights()[0]

def tsne_clean(X, n_components=2, metric='euclidean', rs=100, save=None):
    from sklearn.manifold import TSNE
    import matplotlib.pyplot as plt
    Y = TSNE(n_components=n_components, metric=metric,random_state=rs).fit_transform(X)
    plt.figure(figsize=(20,20))
    text = []
    for i in range(Y.shape[0]):
        x = Y[i,0]
        y = Y[i,1]
        plt.scatter(x,y, marker='x', color='red')
    if save != None:
        plt.savefig(save,format='png',dpi=400)
    plt.show()
    return Y

def tsne(X, size_threshold, user_search_path, company_index_path, n_components=2, metric='euclidean',random_state=100,save=None):
    from sklearn.manifold import TSNE
    import matplotlib.pyplot as plt
    from adjustText import adjust_text
    Y = TSNE(n_components=n_components, metric=metric, random_state=random_state).fit_transform(X)
    #df = pd.read_csv("user_search_companies_profile.csv")
    df = pd.read_csv("intermediate_table/id_type_industry_year_revenue_headquarter.csv")
    user_search = load_data(user_search_path)
    company_index = pkl.load(open(company_index_path, 'rb'))
    select_df = df.loc[(df['id'].isin(company_index.keys())) & (df['raw_size'] > size_threshold),:]
    select_df['index'] = [company_index[x] for x in select_df['id'].values]
    company_names = select_df.sort_values(by=['index'])['name'].values
    inds = [company_index[cp] for cp in select_df.sort_values(by=['index'])['id'].values]
    final_Y = []
    for i,row in enumerate(Y):
        if i in inds:
            final_Y.append(list(row))
    final_Y = np.array(final_Y)
    plt.figure(figsize=(20,20))
    text = []
    for i, txt in enumerate(company_names):
        x = final_Y[i,0]
        y = final_Y[i,1]
    #     plt.annotate(txt, (x,y), xytext=(10,10), textcoords='offset points')
        text.append(plt.text(x,y,txt))
        plt.scatter(x,y, marker='x', color='red')
    adjust_text(text, only_move={'points':'y', 'texts':'y'})
    if save != None:
        plt.savefig(save,format='png',dpi=400, bbox_inches='tight')
    plt.show()
    return Y, final_Y, company_names

def hist_prop(data, title=None, bins=20, xticks=None, xlim=None):
    from matplotlib.ticker import PercentFormatter
    plt.figure(figsize=(10,7))
    plt.title(title)
    plt.hist(data, bins=bins,weights=np.ones(len(data)) / len(data))
    plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
    plt.xticks(xticks)
    plt.xlim(xlim)
    plt.show()

def load_data(path: str) -> dict:
    """
    Load dict from json file
    :param path: data path.
    :return: read dict from json file.
    """
    with open(path) as f:
        data = json.load(f)
    return data

#deprecated#
def company_weights_from_node2vec(weights_path, user_search_path):
    user_search = load_data(user_search_path)
    company_index, index_company, companies_count = generate_index(user_search)
    fp = open(weights_path, 'r')
    weights = defaultdict(list)
    for i,line in enumerate(fp):
        if i > 0:
            tmp = line.strip().split(' ')
            weights[int(tmp[0])] = [float(x) for x in tmp[1:]]
    fp.close()
    weights = {k:v for k,v in sorted(weights.items(), key=lambda item: item[0])}
    company_weights = [weights[k] for k in range(1, len(company_index)+1)]
    company_ids = company_index.keys()
    return company_weights, company_ids

def map_index_name(company_index_path, data='user_search'):
    company_index = pkl.load(open(company_index_path, 'rb'))
    company_index_df = pd.DataFrame(company_index.items(), columns=['id', 'index'])
    if data == 'user_search':
        df = pd.read_csv("intermediate_table/user_search_companies_profile.csv")
        select_df = df.loc[df['id'].isin(company_index_df['id'].values)]
    elif data == 'long_competitors':
        df = pd.read_csv("intermediate_table/company_similar_20210507.csv")
        select_df1 = df.loc[df['search_id'].isin(company_index_df['id'].values)][['search_id','search_company']]
        select_df2 = df.loc[df['similar_id'].isin(company_index_df['id'].values)][['similar_id', 'similar_company']]
        select_df1 = select_df1.rename(columns={'search_id': 'id', 'search_company': 'name'})
        select_df2 = select_df2.rename(columns={'similar_id': 'id', 'similar_company': 'name'})
        select_df = pd.concat([select_df1, select_df2])
        select_df = select_df.drop_duplicates()
    elif data == 'jobhopping':
        df = pd.read_csv("intermediate_table/jobhopping_company_profiles.csv")
        select_df = df.loc[df['id'].isin(company_index_df['id'].values)]
    print(select_df.shape)
    return select_df.merge(company_index_df, how='inner',on='id')

def make_cos_df(weights, select_df=None):
    cos_df = pd.DataFrame(cosine_similarity(weights))
    if select_df is None:
        cos_df.index = weights.index
        cos_df.columns = weights.index
    else:
        cos_df.index = select_df.sort_values(by='index')['name'].values
        cos_df.columns = select_df.sort_values(by='index')['name'].values
    return cos_df

def select_search(user_search_path, df, names):
    select_search = []
    user_search = load_data(user_search_path)
    for search in user_search:
        ids = list(df.loc[df['name'].isin(names)]['id'].values)
        if set(ids) & set(search) != set():
            select_search.append(search)
    return select_search

def generate_search_from_input(input_ids, user_search):
    search_has_company = []
    search_filtered = []
    ids = input_ids
    for search in user_search:
        tmp = set(search) & set(ids)
        if tmp != set():
            search_has_company.append(tuple(search))
            if len(tmp) > 1:
                search_filtered.append(tuple(tmp))
    return search_filtered, search_has_company

def search2name(user_search, df):
    name_search = []
    for search in user_search:
        name_search.append(list(df.loc[df['id'].isin(search)]['name'].values))
    return name_search

def count_search(target_df, data_path, df):
    data = load_data(data_path)
    #company_index = pkl.load(open(company_index_path, 'rb'))
    index_company = {k:v for v,k in company_index.items()}

    pairs_count = defaultdict(int)
    companies_count = defaultdict(int)
#     whole_pairs = defaultdict(int)

    names = target_df.index[1:].values
    target = target_df.index[0]
    target_id = df.loc[df['name']==target]['id'].values[0]
    names_id = df.loc[df['name'].isin(names)]['id'].values # order is not the same as names

    perm = [permutations(search, 2) for search in data]
    for iterate in perm:
        for pair in iterate:
#             tmp_key = '-'.join([str(company_index[pair[0]]), str(company_index[pair[1]])])
#             if not tmp_key in pairs_count.keys():
#                 whole_pairs[tmp_key] = 1
#             else:
#                 whole_pairs[tmp_key] += 1

            if pair[0] == target_id:
                name = df.loc[df['id']==pair[1]]['name'].values[0]
                if not name in pairs_count.keys():
                    pairs_count[name] = 1
                else:
                    pairs_count[name] += 1
            if pair[0] in names_id:
                name = df.loc[df['id']==pair[0]]['name'].values[0]
                if not name in companies_count.keys():
                    companies_count[name] = 1
                else:
                    companies_count[name] += 1
#     if filter != None:
#         filter_companies_count = defaultdict(int)
#         filter_pair_count = defaultdict(int)
#         filter_pair = pkl.load(open(filter, 'rb'))
#         for pair in filter_pair:
#             if index_company[pair[0]] in names_id:
#                 name = df.loc[df['id']==index_company[pair[0]]]['name'].values[0]
#                 name_1 = df.loc[df['id']==index_company[pair[1]]]['name'].values[0]
#                 if not name in filter_companies_count.keys():
#                     filter_companies_count[name] = whole_pairs['-'.join([str(pair[0]), str(pair[1])])]
#                 else:
#                     filter_companies_count[name] += whole_pairs['-'.join([str(pair[0]), str(pair[1])])]
#             if index_company[pair[0]] == target_id:
#                 name = df.loc[df['id']==index_company[pair[1]]]['name'].values[0]
#                 if not name in filter_pair_count.keys():
#                     filter_pair_count[name] = whole_pairs['-'.join([str(pair[0]), str(pair[1])])]
#                 else:
#                     filter_pair_count[name] += whole_pairs['-'.join([str(pair[0]), str(pair[1])])]


    pairs_df = pd.DataFrame(pairs_count.items(), columns=['name','target_query_count'])
    pairs_df = pd.DataFrame({'name':target_df.index, 'cosine' : target_df}).merge(pairs_df, how='inner', on='name')
    pairs_df = pairs_df.merge(pd.DataFrame(companies_count.items(), columns=['name', 'query_pairs_count']), how='inner', on = 'name')
#     if filter != None:
#         pairs_df = pairs_df.merge(pd.DataFrame(filter_pair_count.items(), columns=['name', 'target_query_count_used']), how='left', on='name')
#         pairs_df = pairs_df.merge(pd.DataFrame(filter_pair_count.items(), columns=['name','query_pairs_count_used']), how='left', on='name')
    return pairs_df

def tsne_w2v_from_dict(X_df, select_company_ids=None, n_components=2, metric='euclidean',save=None, data='user_search'):
    from sklearn.manifold import TSNE
    import matplotlib.pyplot as plt
    from adjustText import adjust_text
    Y = TSNE(n_components=n_components, metric=metric).fit_transform(X_df.values)
    Y_df = pd.DataFrame(Y)
    Y_df.index = X_df.index
    #final_Y_df = Y_df.copy()
    if select_company_ids is None:
        select_company_ids = X_df.index.values
    if data == 'user_search':
        df = pd.read_csv("intermediate_table/user_search_companies_profile.csv")
        select_df = df.loc[df['id'].isin(select_company_ids)]
    elif data == 'long_competitors':
        df = pd.read_csv("intermediate_table/company_similar_20210507.csv")
        select_df1 = df.loc[df['search_id'].isin(select_company_ids)][['search_id','search_company']]
        select_df2 = df.loc[df['similar_id'].isin(select_company_ids)][['similar_id', 'similar_company']]
        select_df1 = select_df1.rename(columns={'search_id': 'id', 'search_company': 'name'})
        select_df2 = select_df2.rename(columns={'similar_id': 'id', 'similar_company': 'name'})
        select_df = pd.concat([select_df1, select_df2])
        select_df = select_df.drop_duplicates()
    elif data == 'jobhopping':
        df = pd.read_csv("intermediate_table/jobhopping_company_profiles.csv")
        select_df = df.loc[df['id'].isin(select_company_ids)]
#     df = pd.read_csv("intermediate_table/user_search_companies_profile.csv")
#     select_df = df.loc[(df['id'].isin(X_df.index.values)) & (df['raw_size'] > size_threshold),:]
#     select_df = df.loc[(df['id'].isin(X_df.index.values))]
#     final_Y_df = Y_df.copy() #Y_df.loc[Y_df.index.isin(select_df['id'].values)]
    final_Y_df = Y_df.loc[Y_df.index.isin(select_df['id'].values)]

    plt.figure(figsize=(20,20))
    text = []
    for i, txt in enumerate(final_Y_df.index.values): #final_
        x = final_Y_df.iloc[i,0]
        y = final_Y_df.iloc[i,1]
        #if txt in final_Y_df.index:
    #     plt.annotate(txt, (x,y), xytext=(10,10), textcoords='offset points')
        text.append(plt.text(x,y, df.loc[df['id'] == txt]['name'].values[0])) #select_df.loc[select_df['id'] == txt]['name'].values[0]))
        plt.scatter(x,y, marker='x', color='red')
    adjust_text(text, only_move={'points':'y', 'texts':'y'})
    if save != None:
        plt.savefig(save,format='png',dpi=400)
    plt.show()
    return Y_df, final_Y_df

def generate_pairs_count_df(data: list, df: pd.DataFrame) -> (pd.DataFrame):
    """
    Generate pairs and their counts from raw data
    :param data: List of sets, raw json file
    :return pairs_count: pairs of company A - B - freq
    """
    perm = [permutations(search, 2) for search in data]
    pairs_count = defaultdict(int)
    
    for iterate in perm:
        for pair in iterate:
            tmp_key = '~'.join([pair[0], pair[1]])
            if not tmp_key in pairs_count.keys():
                pairs_count[tmp_key] = 1
            else:
                pairs_count[tmp_key] += 1
                
    companies_pairs = [[x for x in key.split('~')] for key in pairs_count.keys()]
    pairs_df = pd.DataFrame(companies_pairs)
    print(pairs_df.shape)
    if not df is None:
        pairs_df.columns = ['idA','idB']
        df['idA'] = df['id']
        df['idB'] = df['id']
        df['nameA'] = df['name']
        df['nameB'] = df['name']
        pairs_df = pairs_df.merge(df[['idA','nameA']], how='left',on = 'idA')
        pairs_df = pairs_df.merge(df[['idB','nameB']], how='left',on = 'idB')
    else:
        pairs_df.columns = ['nameA','nameB']
    pairs_df['count'] = pairs_count.values()
    
    return pairs_df

def generate_batch(companies_pairs: list, index_company: dict, pairs_count: dict, companies_count:dict,
                   whole_data: list, n_positive: int = 1, n_negative: int = 5, seed: int = 100):
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
    # The whole batch generation can be repeated. Each epoch has different batches.
    # Batches in one epoch are also different.
    random.seed(seed)
    np.random.seed(seed)
    set_seed(seed)
    batch_size = int(n_positive * (1 + n_negative))
    batch = np.zeros((batch_size, 4))

    total_count = sum(companies_count.values())
    companies_freq = {k: v / total_count for k,v in companies_count.items()}
        
    while True:
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
                    neg = 1
                    # negative selection based on unigram ^ (3/4)
                    negative_company = random.choices(population=range(len(index_company)), weights=[x**(3/4) for x in companies_freq.values()], k=1)[0]
                    # Check to make sure this is not a positive example
                    if (target_company, negative_company) not in companies_pairs:
                        if negative_company != target_company:
                            for search in whole_data:
                                if (set([target_company, negative_company]) - set(search)) == set():
                                    neg = 0
                                    break
                            if neg == 1:
                                k += 1
                                weight_of_pair = pairs_count['-'.join([str(target_company), str(object_company)])]
                                batch[idx, :] = (target_company, negative_company, 0, weight_of_pair)
                                idx += 1

            # Make sure to shuffle order
            np.random.shuffle(batch)
            yield {'target': batch[:, 0], 'object': batch[:, 1]}, batch[:, 2].reshape(-1,1), batch[:, 3].reshape(-1,1)
            
def generate_batch_clean(companies_pairs: list, index_company: dict, pairs_count: dict, companies_count:dict,
                   n_positive: int = 1, n_negative: int = 5, seed: int = 100, exclude_search = False):
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
    
    if exclude_search == True:
        new_companies_count = defaultdict(int)
        for k,v in index_company.items():
            if v.split('_')[0] != 'search':
                new_companies_count[k] = companies_count[k]
    else:
        new_companies_count = companies_count
    total_count = sum(new_companies_count.values())
    companies_freq = {k: v / total_count for k,v in new_companies_count.items()}
    
#     shared_common_list = load_data('input/shared_common_list_user_search_company_long_small.json')
#     tmp_company_index = {k:v for v, k in index_company.items()}
#     shared_common_index = [tmp_company_index[x] for x in shared_common_list if x in tmp_company_index.keys()]
    #google_index = tmp_company_index['cmp-5bbdc411-41b2-4d92-8a93-735876a325fc']
    #microsoft_index = tmp_company_index['cmp-4623d626-7efd-4d00-8b4b-08c7d4e793da']
    
    while True:
#         ep += 1
#         ng_pairs = open('data/negative_pairs_microsoft_google_epoch_' + str(ep) + '.txt','a')
        for positive_per_batch in chunks(companies_pairs, n_positive):
            # This creates a generator
            idx = 0
            for (target_company, object_company) in positive_per_batch:
                weight_of_pair = pairs_count['-'.join([str(target_company), str(object_company)])]
#                 if target_company in [google_index, microsoft_index]:
#                     if object_company in shared_common_index:
#                         weight_of_pair = 50000
                        #print(index_company[target_company], index_company[object_company])
                batch[idx, :] = (target_company, object_company, 1, weight_of_pair)
                # Increment idx by 1
                idx += 1
                # Add negative examples until reach batch size
                k = 0
                while k < n_negative:
                    # negative selection based on unigram ^ (3/4)
                    negative_company = random.choices(population=list(new_companies_count.keys()), weights=[x**(3/4) for x in companies_freq.values()], k=1)[0]
                    # Check to make sure this is not a positive example
                    if (target_company, negative_company) not in companies_pairs:
                        k += 1
                        weight_of_pair = pairs_count['-'.join([str(target_company), str(object_company)])]
                        batch[idx, :] = (target_company, negative_company, 0, weight_of_pair)
#                         ng_pairs.write(index_company[target_company] + '\t' + index_company[negative_company] + '\n')
                        idx += 1

            # Make sure to shuffle order
            np.random.shuffle(batch)
            yield {'target': batch[:, 0], 'object': batch[:, 1]}, batch[:, 2].reshape(-1,1), batch[:, 3].reshape(-1,1)
#         ng_pairs.close()



def draw_graph(data, map_df, target_company, query_company, target_all_neighbors=True, query_all_neighbors=True):
    if not map_df is None:
        target_id = map_df.loc[map_df['name'] == target_company]['id'].values[0]
        query_id = map_df.loc[map_df['name'] == query_company]['id'].values[0]
        
    pairs_df = generate_pairs_count_df(data, map_df)
    select_pairs_target = pairs_df.loc[pairs_df['nameA'] == target_company]
    select_pairs_query = pairs_df.loc[pairs_df['nameB'] == query_company]
    G=nx.Graph()
    target_neighbors = select_pairs_target['nameB'].values.tolist()
    query_neighbors = select_pairs_query['nameA'].values.tolist()
    target_edges = [tuple(x) for x in select_pairs_target[['nameA','nameB']].values.tolist()]
    query_edges = [tuple(x) for x in select_pairs_query[['nameA','nameB']].values.tolist()]
    G.add_nodes_from([target_company, query_company])
    intersect_nodes = set(target_neighbors) & set(query_neighbors)
    G.add_nodes_from(intersect_nodes)
    intersect_edges = [set([target_company, x]) for x in intersect_nodes] + [set([x, query_company]) for x in intersect_nodes]
    G.add_edges_from(intersect_edges)

    if target_all_neighbors == True:
        G.add_nodes_from(target_neighbors)
        G.add_edges_from(target_edges)
    if query_all_neighbors == True:
        G.add_nodes_from(query_neighbors) #omit redundancy automatically
        G.add_edges_from(query_edges)

#     edgewidth = []
#     for (u, v, d) in G.edges(data=True):
#         edgewidth.append(pairs_df.loc[(pairs_df['nameA']==u) & (pairs_df['nameB'] == v)]['count'].values[0])

    pos = nx.spring_layout(G)  # positions for all nodes

    plt.figure(figsize=(10,10))
    # nodes
    remain_nodes = [u[0] for u in G.nodes(data=True) if (u[0] != target_company) and (u[0] != query_company)]
    nx.draw_networkx_nodes(G, pos, nodelist=[target_company, query_company], node_color = 'r', node_size=700)

    nx.draw_networkx_nodes(G, pos, nodelist=remain_nodes, node_size=700)
    nx.draw_networkx_edges(G, pos)

    nx.draw_networkx_labels(G, pos, font_size=15, font_family="sans-serif")
    
    texts = []
    for (u,v,d) in G.edges(data=True):
        adj_axis = (pos[u] + pos[v]) / 2 + 0.1
        plt.text(adj_axis[0], adj_axis[1], str(pairs_df.loc[(pairs_df['nameA']==u) & (pairs_df['nameB'] == v)]['count'].values[0]))

    plt.axis("off")
    plt.show()
    return remain_nodes

# class EpochLogger(CallbackAny2Vec): #, output_name
#     '''Callback to log information about training'''
    
    
#     def __init__(self):
#         self.epoch = 0
#         self.loss = []
#         self.modified_loss = []

#     def on_epoch_begin(self, model):
#         pass


#     def on_epoch_end(self, model):
#         loss = model.get_latest_training_loss()
#         #model.wv.save_word2vec_format('data/company_weights_' + output_name + '/epoch_' + 
#         #                              str(self.epoch) + '_hs_w2v_gensim.dict')
#         if self.epoch == 0:
#             print('Loss after epoch {}: {}'.format(self.epoch, loss))
#             self.modified_loss.append(loss)
#         else:
#             print('Loss after epoch {}: {}'.format(self.epoch, loss- self.loss_previous_step))
#             self.modified_loss.append(loss- self.loss_previous_step)
#         self.epoch += 1
#         self.loss_previous_step = loss

#         if self.epoch % 100 == 0:
#             plt.plot(self.modified_loss, '-')
#             plt.show()