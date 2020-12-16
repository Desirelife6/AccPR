import torch
import pandas as pd
import numpy as np
import warnings
from gensim.models.word2vec import Word2Vec
# from origin_model import BatchProgramCC
import os

warnings.filterwarnings('ignore')
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import argparse

parser = argparse.ArgumentParser(description="Choose project_name and bug_id")
parser.add_argument('--project_name')
parser.add_argument('--bug_id')
parser.add_argument('--predict_baseline')
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
    base_url = 'simfix_supervised_data/' + project_name + '/' + bug_id + '/'
    PREDICT_BASE = True
else:
    PREDICT_BASE = False
    base_url = 'simfix_unsupervised_data/' + project_name + '/' + bug_id + '/'


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
    # precision = checkpoint['precision']
    # f1 = checkpoint['f1']
    print('Checkpoint Loaded!')
    # print('precision = {}, f1 = {}'.format(precision, f1))
    return model


if __name__ == '__main__':
    model = load_model()

    # predict_data = pd.read_pickle('simfix_data/blocks.pkl').sample(frac=1)

    # File is too big for pickle to load
    from sklearn.externals import joblib

    predict_data = joblib.load(base_url + 'blocks.pkl').sample(frac=1)

    predict_data = predict_data.sort_values(['id2'], ascending=True)
    i = 0
    dict = {}
    pattern_res = []

    while i < len(predict_data):
        batch = get_batch(predict_data, i, 1)
        i += 1
        predict1_inputs, predict2_inputs, predict_labels, id = batch
        print(predict1_inputs)
        print(predict2_inputs)

        if USE_GPU:
            predict1_inputs, predict2_inputs, predict_labels, id = predict1_inputs, predict2_inputs, predict_labels.cuda()

        if PREDICT_BASE:
            model.zero_grad()
            model.batch_size = len(predict_labels)
            model.hidden = model.init_hidden()

            buggy_code_encode = model.encode(predict1_inputs)
            candidate_encode = model.encode(predict2_inputs)

            # Generate embeddings of GenPat patterns
            # tmp = candidate_encode.detach().numpy()
            # tmp = np.squeeze(tmp)
            # tmp = np.insert(tmp, 0, id)
            # pattern_res.append(tmp)

            import torch.nn.functional as F

            buggy_code_encode = F.normalize(buggy_code_encode)
            candidate_encode = F.normalize(candidate_encode)

            distance = float(buggy_code_encode.mm(candidate_encode.t()))
            dict[str(id[0])] = distance

        else:
            model.zero_grad()
            model.batch_size = len(predict_labels)
            model.hidden = model.init_hidden()
            model.hidden_decode = model.init_hidden_decode()

            _, buggy_code_encode, _ = model.encode(predict1_inputs)
            _, candidate_encode, _ = model.encode(predict2_inputs)

            # Generate embeddings of GenPat patterns
            # tmp = candidate_encode.detach().numpy()
            # tmp = np.squeeze(tmp)
            # tmp = np.insert(tmp, 0, id)
            # pattern_res.append(tmp)

            import torch.nn.functional as F

            buggy_code_encode = F.normalize(buggy_code_encode)
            candidate_encode = F.normalize(candidate_encode)

            distance = float(buggy_code_encode.mm(candidate_encode.t()))
            dict[str(id[0])] = distance

    # pattern_res = np.array(pattern_res)
    # np.save('simfix_data/pattern_res', pattern_res)

    dict_result = pd.DataFrame(list(dict.items()))

    dict_result.to_csv(base_url + '/dict_result.csv')

    print(len(dict))
    print(sorted(dict.items(), key=lambda e: e[1], reverse=True))
