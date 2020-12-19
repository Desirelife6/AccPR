from utils import *
import pandas as pd
import os
import sys
import warnings
from tqdm import tqdm

warnings.filterwarnings('ignore')


class OovPipeline:
    def __init__(self, root, w2v_path):
        self.root = root
        self.sources = None
        self.blocks = None
        self.pairs = None
        self.train_file_path = None
        self.dev_file_path = None
        self.test_file_path = None
        self.size = None
        self.w2v_path = w2v_path
        self.fault_ids = pd.Series()
        self.oov = 0
        self.sum_count = 0

    # parse source code
    def parse_source(self, output_file, option):

        import javalang
        source = pd.read_csv('OOV/simfix_all_words.tsv', sep='\t', header=None,
                             encoding='utf-8')
        source.columns = ['id', 'code']
        tmp = []
        for code in tqdm(source['code']):
            try:
                tokens = javalang.tokenizer.tokenize(code)
                parser = javalang.parser.Parser(tokens)
                code = parser.parse_member_declaration()
                tmp.append(code)
            # print(code)
            except:
                faulty_code_file = 'faulty_code.txt'
                out = open(faulty_code_file, 'a+')
                out.write('Code snippet failed to pass parsing')
                out.write(str(code))
                # print('Error happened while parsing')
                # print(code)
                out.close()
                code = None
                tmp.append(code)

        source['code'] = tmp
        source['code'] = source['code'].fillna('null')

        faults = source.loc[source['code'] == 'null']

        self.fault_ids = faults['id']

        source = source[~source['code'].isin(['null'])]
        # source.to_pickle(path)
        self.sources = source
        return source

    # def dictionary_and_embedding(self, size):
    #     self.size = size
    #
    #     trees = self.sources
    #
    #     def trans_to_sequences(ast):
    #         sequence = []
    #         get_sequence(ast, sequence)
    #         return sequence
    #
    #     corpus = trees['code'].apply(trans_to_sequences)
    #     str_corpus = [' '.join(c) for c in corpus]
    #     trees['code'] = pd.Series(str_corpus)
    #     # trees.to_csv(data_path+'train/programs_ns.tsv')
    #
    #     from gensim.models.word2vec import Word2Vec
    #     w2v = Word2Vec(corpus, size=size, workers=16, sg=1, max_final_vocab=30000)
    #     w2v.save('simfix_node_w2v_' + str(size))

    # generate block sequences with index representations
    def generate_block_seqs(self):

        from gensim.models.word2vec import Word2Vec

        word2vec = Word2Vec.load(self.w2v_path).wv
        vocab = word2vec.vocab
        max_token = word2vec.syn0.shape[0]

        def tree_to_index(node):
            token = node.token
            self.sum_count += 1
            if not (token in vocab):
                print(token)
                out.write(token)
                out.write('\n')
                # out.close()
                self.oov += 1
            result = [vocab[token].index if token in vocab else max_token]

            children = node.children
            for child in children:
                res = tree_to_index(child)
                result.append(res)
            return result

        trees = pd.DataFrame(self.sources, copy=True)

        temp = []
        for code in tqdm(trees['code']):
            try:
                blocks = []
                get_blocks_v1(code, blocks)
                tree = []
                for b in blocks:
                    btree = tree_to_index(b)
                    tree.append(btree)
                code = tree
                temp.append(code)
            except:
                code = None
                temp.append(code)
                print('Wooooooooooops')

        trees['code'] = temp
        print(self.sum_count)
        print(self.oov)
        print(str(self.oov / self.sum_count * 100) + '%')
        trees['code'] = trees['code'].fillna('null')
        trees = trees[~(trees['code'] == 'null')]

        if 'label' in trees.columns:
            trees.drop('label', axis=1, inplace=True)
        self.blocks = trees

    # run for processing data to train
    def run(self):
        print('parse source code...')
        self.parse_source(output_file='ast.pkl', option='existing')
        print('generate block sequences...')
        self.generate_block_seqs()


out = open('OOV/oov_res.txt', 'w')
ppl = OovPipeline('base_data/', w2v_path='base_embedding/base_node_w2v_30000')
ppl.run()
out.close()

col_listall = []
with open(r"OOV/oov_res.txt",
          'r') as f:
    for line in f:
        col_listall.append(line.strip('\n'))

res = list(set(col_listall))
print(len(res))
print(res)


def text_save(filename, data):  # filename为写入CSV文件的路径，data为要写入数据列表.
    file = open(filename, 'w')
    for i in range(len(data)):
        s = str(data[i]).replace('[', '').replace(']', '')  # 去除[],这两行按数据不同，可以选择
        s = s.replace("'", '').replace(',', '') + '\n'  # 去除单引号，逗号，每行末尾追加换行符
        file.write(s)
    file.close()
    print("保存文件成功")


text_save('OOV/oov_res.txt', res)
# from gensim.models.word2vec import Word2Vec
# from sklearn.decomposition import PCA
# from matplotlib import pyplot
#
# word2vec = Word2Vec.load('base_embedding/base_node_w2v_128').wv
# vocab = word2vec.vocab
# max_token = word2vec.syn0.shape[0]
#
# model = Word2Vec.load('base_embedding/base_node_w2v_128')
#
# model2 = Word2Vec.load('simfix_node_w2v_128')
#
# import random
#
# # 基于2d PCA拟合数据
#
# X = model[model.wv.vocab]
# X2 = model2[model2.wv.vocab]
# pca = PCA(n_components=2)
# pca2 = PCA(n_components=2)
# result = pca.fit_transform(X)
# result2 = pca2.fit_transform(X2)
# # 可视化展示
# pyplot.scatter(result[:, 0], result[:, 1], c='hotpink')
#
# # pyplot.scatter(result2[:, 0], result2[:, 1], c='r')
#
# words1 = list(model.wv.vocab)
# words2 = list(model2.wv.vocab)
# random.seed(10)
# i = int(len(list(model.wv.vocab)) / 1000)
# print(i)
# slice1 = random.sample(words1, i)
# slice2 = random.sample(words2, i)
#
# words1 = slice1
# words2 = slice2
#
# for i, word in enumerate(slice1):
#     pyplot.annotate(word, xy=(result[i, 0], result[i, 1]))
#
# for i, word in enumerate(slice2):
#     pyplot.annotate(word, xy=(result2[i, 0], result2[i, 1]))
# pyplot.show()
