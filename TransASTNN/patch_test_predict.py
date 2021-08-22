import torch
import pandas as pd
import numpy as np
import warnings
from gensim.models.word2vec import Word2Vec
# from origin_model import BatchProgramCC
import os
import time
import torch.nn.functional as F

warnings.filterwarnings('ignore')
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import argparse

parser = argparse.ArgumentParser(description="Choose project_name and bug_id")
parser.add_argument('--project_name')
parser.add_argument('--bug_id')
parser.add_argument('--predict_baseline')
args = parser.parse_args()
# if not args.project_name:
#     print("No specified project_name")
#     exit(1)
# if not args.bug_id:
#     print("No specified bug_id")
#     exit(1)
# if not args.predict_baseline:
#     print("No specified predict type")
#     exit(1)

PREDICT_BASE = True
project_name = args.project_name
bug_id = args.bug_id

base_url = ''
if args.predict_baseline == 'true':
    base_url = 'patches/' + project_name + '/' + bug_id + '/'
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
    # print('Checkpoint Loaded!')
    # print('precision = {}, f1 = {}'.format(precision, f1))
    return model


if __name__ == '__main__':


    model = load_model()

    base_url = 'patches/' + project_name + '/' + bug_id + '/'

    predict_data = pd.read_pickle(base_url + 'blocks.pkl').sample(frac=1)

    # File is too big for pickle to load
    from sklearn.externals import joblib

    # predict_data = joblib.load(base_url + 'blocks.pkl').sample(frac=1)

    predict_data = predict_data.sort_values(['id2'], ascending=True)
    i = 0
    dict = {}

    embedding_dict = {}

    pattern_res = []

    time_url = base_url + "time.txt"

    if not os.path.exists(time_url):
        with open(time_url, "w") as f:
            text = f.write(str(0) + "\n")
            print(text)
        f.close()

    with open(time_url, "r") as f:
        tmp = f.read()
        total_time = float(tmp)
    f.close()

    while i < len(predict_data):
        batch = get_batch(predict_data, i, 1)
        i += 1

        predict1_inputs, predict2_inputs, predict_labels, id = batch
        # print(predict1_inputs)
        # print(predict2_inputs)
        model.zero_grad()
        model.batch_size = len(predict_labels)
        model.hidden = model.init_hidden()

        buggy_code_encode = model.encode(predict1_inputs)
        candidate_encode = model.encode(predict2_inputs)

        buggy_code_encode = F.normalize(buggy_code_encode)
        candidate_encode = F.normalize(candidate_encode)
        
        embedding_dict[0] = buggy_code_encode
        embedding_dict[str(id[0])] = candidate_encode

        # buggy_code_encode = np.linalg.norm(buggy_code_encode.detach().numpy(), ord=1)
        # candidate_encode = np.linalg.norm(candidate_encode.detach().numpy(), ord=1)

        start = time.time()
        distance = np.linalg.norm(buggy_code_encode.detach().numpy() - candidate_encode.detach().numpy())
        # distance = float(buggy_code_encode.mm(candidate_encode.t()))
        end = time.time()

        dict[str(id[0])] = distance
        total_time = total_time + (end - start)

    import csv

    # 1. 创建文件对象
    # f = open('/data/yc/OverfitSimfix/TransASTNN/norm1_patch_res/' + project_name + "_" + bug_id + ".csv", 'a', encoding='utf-8')

    # 2. 基于文件对象构建 csv写入对象
    # csv_writer = csv.writer(f)

    # 3. 构建列表头
    # csv_writer.writerow(["rank", "src_id", "norm1", "min_vector_similarity", "neighbor_vector_similarity",
    #                      "neighbor_similarity_diff"])

    result_file = open(base_url + '/dict_result.csv', 'w', encoding='utf-8')

    # 2. 基于文件对象构建 csv写入对象
    result_writer = csv.writer(result_file)

    dict = sorted(dict.items(), key=lambda e: e[1], reverse=False)
    dict = dict[0:50]
    # dict = list(dict.items())
    n = 0
    #diff_neighbor = 0
    #sim_neighbor_diff = 0

    for i in range(0, len(dict)):
        csv_res_list = []
        #mins = 1000000000
        #if i == 0:
        #    diff_neighbor = np.linalg.norm(
        #        embedding_dict[dict[i][0]].detach().numpy() - embedding_dict[0].detach().numpy())
        #    if diff_neighbor == 0.0:
        #        diff_neighbor = 1

        #for tmp in range(0, i):
        #    mins = min(np.linalg.norm(
        #        embedding_dict[dict[i][0]].detach().numpy() - embedding_dict[dict[tmp][0]].detach().numpy()), mins)

        #diff_neighbor = np.linalg.norm(
        #    embedding_dict[dict[i][0]].detach().numpy() - embedding_dict[dict[i - 1][0]].detach().numpy())
        #sim_neighbor_diff = dict[i][1] - dict[i - 1][1]

        csv_res_list.append(i)
        csv_res_list.append(dict[i][0])
        csv_res_list.append(dict[i][1])
        # csv_res_list.append(diff_neighbor)

        for patch in range(0, i):
            sim = np.linalg.norm(
                embedding_dict[dict[i][0]].detach().numpy() - embedding_dict[dict[patch][0]].detach().numpy())
            csv_res_list.append(sim)

        # print(str(n) + " " + str(dict[i]) + " " + str(mins) + " " + str(diff_neighbor) + str(sim_neighbor_diff))
        # print(csv_res_list)
        n = n + 1
        result_writer.writerow(csv_res_list)

        # result_writer.writerow([str(i), str(dict[i][0]), str(dict[i][1]), str(diff_neighbor)])

    with open(time_url, "w") as f:
        text = f.write(str(total_time) + "\n")
    f.close()
