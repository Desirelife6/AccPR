import pandas as pd
import torch
import time
import numpy as np
import warnings
from gensim.models.word2vec import Word2Vec
from unsupervised_model import BatchProgramCC
from torch.autograd import Variable
from sklearn.metrics import precision_recall_fscore_support
from base_pipeline import Pipeline

# from base_pipeline import Pipeline

# ppl = Pipeline('3:1:1', 'base_data/', w2v_path='base_embedding/')
# ppl.run()

warnings.filterwarnings('ignore')

HIDDEN_DIM = 100
ENCODE_DIM = 128
LABEL_SIZE = 1
EPOCHS = 5
BATCH_SIZE = 32
USE_GPU = True if torch.cuda.is_available() else False
W2V_SIZE = 30000
W2V_PATH = 'all_words_embedding/all_words_w2v_' + str(W2V_SIZE)
data_root = 'unsupervised_data'
save_dir = 'unsupervised_result/' + str(W2V_SIZE) + '/'
save_file = save_dir + 'unsupervised_model_{}.pth.tar'.format(str(W2V_SIZE))


# ppl = Pipeline('3:1:1', 'base_data/', w2v_path=W2V_PATH)
# ppl.run()


def get_batch(dataset, idx, bs):
    tmp = dataset.iloc[idx: idx + bs]
    x1, x2, labels = [], [], []
    for _, item in tmp.iterrows():
        x1.append(item['code_x'])
        x2.append(item['code_y'])
        labels.append([item['label']])
    return x1, x2, torch.FloatTensor(labels)


def init_model():
    word2vec = Word2Vec.load(W2V_PATH).wv
    max_tokens = word2vec.syn0.shape[0]
    embedding_dim = word2vec.syn0.shape[1]
    embeddings = np.zeros((max_tokens + 1, embedding_dim), dtype="float32")
    embeddings[:word2vec.syn0.shape[0]] = word2vec.syn0

    model = BatchProgramCC(embedding_dim, HIDDEN_DIM, max_tokens + 1, ENCODE_DIM, LABEL_SIZE, BATCH_SIZE,
                           USE_GPU, embeddings)
    return model


if __name__ == '__main__':

    categories = 5
    print("Begin Training")
    train_data = pd.read_pickle(data_root + 'blocks.pkl').sample(frac=1)
    test_data = pd.read_pickle(data_root + 'blocks.pkl').sample(frac=1)

    model = init_model().cuda() if USE_GPU else init_model()

    parameters = model.parameters()
    optimizer = torch.optim.Adamax(parameters)
    loss_function = torch.nn.BCELoss()

    print(train_data)
    precision, recall, f1 = 0, 0, 0
    print('Start training...')
    out = open(save_dir + 'train_out.txt', 'a+')
    for t in range(1, categories + 1):
        print("Begin training for category %d..." % t)
        out.write("Begin training for category %d..." % t)
        train_data_t = train_data[train_data['label'].isin([t, 0])]
        train_data_t.loc[train_data_t['label'] > 0, 'label'] = 1

        test_data_t = test_data[test_data['label'].isin([t, 0])]
        test_data_t.loc[test_data_t['label'] > 0, 'label'] = 1

        # training procedure
        for epoch in range(EPOCHS):
            print('Starting training for epoch: ' + str(epoch))
            start_time = time.time()
            # training epoch
            total_acc = 0.0
            total_loss = 0.0
            total = 0.0
            i = 0
            while i < len(train_data_t):
                batch = get_batch(train_data_t, i, BATCH_SIZE)
                i += BATCH_SIZE
                train1_inputs, train2_inputs, train_labels = batch
                if USE_GPU:
                    train1_inputs, train2_inputs, train_labels = train1_inputs, train2_inputs, train_labels.cuda()

                model.zero_grad()
                model.batch_size = len(train_labels)
                model.hidden = model.init_hidden()
                model.hidden_decode = model.init_hidden_decode()
                _, _, _, output = model(train1_inputs, train2_inputs)

                loss = loss_function(output, Variable(train_labels))
                loss.backward()
                optimizer.step()
            print('Finished for epoch: ' + str(epoch))

        print("Testing for category %d..." % t)
        # testing procedure
        predicts = []
        trues = []
        total_loss = 0.0
        total = 0.0
        i = 0
        while i < len(test_data_t):
            batch = get_batch(test_data_t, i, BATCH_SIZE)
            i += BATCH_SIZE
            test1_inputs, test2_inputs, test_labels = batch
            if USE_GPU:
                test_labels = test_labels.cuda()

            model.batch_size = len(test_labels)
            model.hidden = model.init_hidden()
            _, _, _, output = model(test1_inputs, test2_inputs)

            loss = loss_function(output, Variable(test_labels))

            # calc testing acc
            predicted = (output.data > 0.5).cpu().numpy()
            predicts.extend(predicted)
            trues.extend(test_labels.cpu().numpy())
            total += len(test_labels)
            total_loss += loss.item() * len(test_labels)

        weights = [0, 0.005, 0.001, 0.002, 0.010, 0.982]
        p, r, f, _ = precision_recall_fscore_support(trues, predicts, average='binary')
        precision += weights[t] * p
        recall += weights[t] * r
        f1 += weights[t] * f
        print("Type-" + str(t) + ": " + str(p) + " " + str(r) + " " + str(f))
        out.write("Type-" + str(t) + ": " + str(p) + " " + str(r) + " " + str(f))

    print("Total testing results(P,R,F1):%.3f, %.3f, %.3f" % (precision, recall, f1))
    out.write("Total testing results(P,R,F1):%.3f, %.3f, %.3f" % (precision, recall, f1))
    out.close()
    torch.save({
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'precision': precision,
        'f1': f1,
    }, save_file)
