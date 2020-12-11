import torch
import pandas as pd
import numpy as np
import warnings
from gensim.models.word2vec import Word2Vec
# from origin_model import BatchProgramCC
import os

warnings.filterwarnings('ignore')
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

PREDICT_BASE = True
USE_GPU = True if torch.cuda.is_available() else False
HIDDEN_DIM = 100
ENCODE_DIM = 128
LABELS = 1
EPOCHS = 5
BATCH_SIZE = 32
W2V_SIZE = 300000
W2V_PATH = 'all_words_embedding/all_words_w2v_' + str(W2V_SIZE)

from predict_pipeline import Pipeline

ppl = Pipeline('predict_data/', w2v_path=W2V_PATH)
ppl.run()

if PREDICT_BASE:
    from base_model import BatchProgramCC

    model_path = 'base_result/{}/base_model_{}.pth.tar'.format(str(W2V_SIZE), str(W2V_SIZE))
else:
    from unsupervised_model import BatchProgramCC

    model_path = 'unsupervised_result/unsupervised_model.pth.tar'


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
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    precision = checkpoint['precision']
    f1 = checkpoint['f1']
    print('Checkpoint Loaded!')
    print('precision = {}, f1 = {}'.format(precision, f1))
    return model


if __name__ == '__main__':

    model = load_model()

    predict_data = pd.read_pickle('predict_data/blocks.pkl').sample(frac=1)

    i = 0
    dict = {}
    while i < len(predict_data):
        batch = get_batch(predict_data, i, 1)
        i += 1
        predict1_inputs, predict2_inputs, predict_labels, id = batch
        print(predict1_inputs)
        print(predict2_inputs)

        if USE_GPU:
            predict1_inputs, predict2_inputs, predict_labels, id = predict1_inputs, predict2_inputs, predict_labels.cuda()

        model.zero_grad()
        model.batch_size = len(predict_labels)
        model.hidden = model.init_hidden()

        buggy_code_encode = model.encode(predict1_inputs)
        candidate_encode = model.encode(predict2_inputs)

        import torch.nn.functional as F

        buggy_code_encode = F.normalize(buggy_code_encode)
        candidate_encode = F.normalize(candidate_encode)

        distance = float(buggy_code_encode.mm(candidate_encode.t()))

        # distance = numpy.sqrt(numpy.sum(numpy.square(candidate_encode.detach().numpy() /
        # - buggy_code_encode.detach().numpy())))
        # distance = float(torch.cosine_similarity(buggy_code_encode, candidate_encode, dim=1)[0])
        # print(distance)

        # distance = model(predict1_inputs, predict2_inputs)
        # print(distance)
        dict[str(id[0])] = distance

    dict_result = pd.DataFrame(list(dict.items()))

    # dict_result.drop([' '])
    # dict_result.drop([0])
    # dict_result.drop(['0'])
    dict_result.to_csv('predict_data/dict_result.csv')

    print(len(dict))
    print(sorted(dict.items(), key=lambda e: e[1], reverse=True))
