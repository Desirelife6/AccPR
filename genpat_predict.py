import torch
import pandas as pd
import numpy as np
import warnings
from gensim.models.word2vec import Word2Vec
import os
from tqdm import tqdm

warnings.filterwarnings('ignore')
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import argparse

parser = argparse.ArgumentParser(description="Choose project_name and bug_id")
parser.add_argument('--project_name')
parser.add_argument('--bug_id')
parser.add_argument('--predict_baseline')
parser.add_argument('--aggregation')

args = parser.parse_args()
if not args.project_name:
    print("No specified project_name")
    exit(1)
if not args.bug_id:
    print("No specified bug_id")
    exit(1)
if not args.predict_baseline:
    print("No specified predict type")
    exit(1)

PREDICT_BASE = True
project_name = args.project_name
bug_id = args.bug_id
base_url = ''
if args.predict_baseline == 'true':
    base_url = 'genpat_supervised_data/' + project_name + '/' + bug_id + '/'
    PREDICT_BASE = True
else:
    PREDICT_BASE = False
    base_url = 'genpat_unsupervised_data/' + project_name + '/' + bug_id + '/'

PREDICT_BASE = True
USE_GPU = True if torch.cuda.is_available() else False
HIDDEN_DIM = 100
ENCODE_DIM = 128
LABELS = 1
EPOCHS = 5
BATCH_SIZE = 32
W2V_SIZE = 30000
W2V_PATH = 'all_words_embedding/all_words_w2v_' + str(W2V_SIZE)

if PREDICT_BASE:
    from base_model import BatchProgramCC

    model_path = 'base_result/{}/base_model_{}.pth.tar'.format(str(W2V_SIZE), str(W2V_SIZE))
else:
    from unsupervised_model import BatchProgramCC

    model_path = 'unsupervised_result/{}/unsupervised_model_{}.pth.tar'.format(str(W2V_SIZE), str(W2V_SIZE))


def get_batch(dataset, idx, bs):
    tmp = dataset.iloc[idx: idx + bs]
    x1, x2, labels, id = [], [], [], 0
    for _, item in tmp.iterrows():
        x1.append(item['code_x'])
        x2.append(item['code_y'])
        labels.append([item['label']])
        id = [item['id2']]
    return x1, x2, torch.FloatTensor(labels), id


def load_model():
    word2vec = Word2Vec.load(W2V_PATH).wv

    max_tokens = word2vec.syn0.shape[0]
    embedding_dim = word2vec.syn0.shape[1]
    embeddings = np.zeros((max_tokens + 1, embedding_dim), dtype="float32")
    embeddings[:word2vec.syn0.shape[0]] = word2vec.syn0

    model = BatchProgramCC(embedding_dim, HIDDEN_DIM, max_tokens + 1, ENCODE_DIM, LABELS, BATCH_SIZE,
                           USE_GPU, embeddings)

    parameters = model.parameters()
    optimizer = torch.optim.Adamax(parameters)
    checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    precision = checkpoint['precision']
    f1 = checkpoint['f1']
    print('Checkpoint Loaded!')
    print('precision = {}, f1 = {}'.format(precision, f1))
    return model


if __name__ == '__main__':

    model = load_model()

    predict_data = pd.read_pickle(base_url + 'blocks.pkl').sample(frac=1)

    batch = get_batch(predict_data, 0, 1)
    predict1_inputs, predict2_inputs, predict_labels, id = batch
    print(predict1_inputs)
    print(predict2_inputs)

    if USE_GPU:
        predict1_inputs, predict2_inputs, predict_labels, id = predict1_inputs, predict2_inputs, predict_labels.cuda()

    model.zero_grad()
    model.batch_size = len(predict_labels)
    model.hidden = model.init_hidden()

    candidate_encode = model.encode(predict2_inputs)

    pattern_embeddings = ['2', '3', '4', '5']
    patterns = np.load('genpat_supervised_data/pattern_res1.npy')

    for i in pattern_embeddings:
        tmp = np.load('genpat_supervised_data/pattern_res{}.npy'.format(i))
        patterns = np.row_stack((patterns, tmp))

    # print(patterns.shape)
    # for index, pattern in tqdm(enumerate(patterns)):
    #     print(int(pattern[0]))

    dic = {}
    for index, pattern in tqdm(enumerate(patterns)):
        # print(patterns)
        buggy_code = candidate_encode.detach().numpy()
        a_norm = np.linalg.norm(buggy_code)
        id = int(pattern[0])
        pattern = np.delete(pattern, 0)
        b_norm = np.linalg.norm(pattern)
        cosine_sim = float(np.dot(buggy_code, pattern.T) / (a_norm * b_norm))

        dic[id] = cosine_sim

    res_dic = dic.copy()
    aggregation = args.aggregation

    dic = sorted(dic.items(), key=lambda e: e[1], reverse=True)

    if aggregation == 'true':
        tmp = dic[0][1]

        for i in range(1, len(dic)):
            if tmp - dic[i][1] <= 0.0001:
                res_dic.pop(dic[i][0])
            else:
                tmp = dic[i][1]

    dic = sorted(res_dic.items(), key=lambda e: e[1], reverse=True)
    print(len(dic))
    print(dic)

    dict_result = pd.DataFrame(dic)
    dict_result.to_csv(base_url + '/dict_result.csv', index=False)
