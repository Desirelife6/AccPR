from utils import *
import pandas as pd
import os
import sys
import warnings
from tqdm import tqdm

warnings.filterwarnings('ignore')


class Pipeline:
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

    # parse source code
    def parse_source(self, output_file, option):

        import javalang
        # def parse_program(func):
        #     try:
        #         tokens = javalang.tokenizer.tokenize(func)
        #         parser = javalang.parser.Parser(tokens)
        #         tree = parser.parse_member_declaration()
        #         return tree
        #     except:
        #         print(str(tokens))
        #         print('Error happened while parsing')

        source = pd.read_csv(self.root + 'src.tsv', sep='\t', header=None,
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
                faulty_code_file = self.root + 'faulty_code.txt'
                out = open(faulty_code_file, 'a+')
                out.write('Code snippet failed to pass parsing')
                out.write('\n')
                out.write(str(code))
                out.write('\n')
                out.write('\n')
                print('Code snippet failed to pass parsing')
                print(str(code))
                out.close()
                code = None
                tmp.append(code)

        source['code'] = tmp
        # source['code'] = source['code'].apply(parse_program)
        source['code'] = source['code'].fillna('null')

        faults = source.loc[source['code'] == 'null']

        self.fault_ids = faults['id']

        # for fault_id in self.fault_ids:
        #     print(fault_id)

        source = source[~source['code'].isin(['null'])]
        # Files are too big for pickle to save, so I tried joblib
        # source.to_csv(self.root + '/test.csv')
        # from sklearn.externals import joblib
        # joblib.dump(source, self.root + '/pattern.pkl')
        # source.to_pickle(path)
        self.sources = source

        return source

    # create clone pairs
    def read_pairs(self, filename):
        pairs = pd.read_csv(self.root + filename)
        # pairs = pd.read_pickle(self.root + filename)
        if not self.fault_ids.empty:
            for fault_id in self.fault_ids:
                pairs = pairs[~pairs['id1'].isin([fault_id])]
                pairs = pairs[~pairs['id2'].isin([fault_id])]
        self.pairs = pairs

    # split data for training, developing and testing
    def split_data(self):
        data_path = self.root
        data = self.pairs

        # data = data.sample(frac=1)
        train = data.iloc[:]

        def check_or_create(path):
            if not os.path.exists(path):
                os.mkdir(path)

        train_path = data_path + 'train/'
        check_or_create(train_path)
        self.train_file_path = train_path + 'train_.pkl'
        train.to_pickle(self.train_file_path)

    # construct dictionary and train word embedding
    def dictionary_and_embedding(self, size):
        self.size = size

        input_file = self.train_file_path
        pairs = pd.read_pickle(input_file)
        train_ids = pairs['id1'].append(pairs['id2']).unique()

        trees = self.sources.set_index('id', drop=False).loc[train_ids]
        # trees = self.sources
        if not os.path.exists(self.w2v_path):
            os.mkdir(self.w2v_path)

        def trans_to_sequences(ast):
            sequence = []
            get_sequence(ast, sequence)
            return sequence

        corpus = trees['code'].apply(trans_to_sequences)
        str_corpus = [' '.join(c) for c in corpus]
        trees['code'] = pd.Series(str_corpus)
        # trees.to_csv(data_path+'train/programs_ns.tsv')

        from gensim.models.word2vec import Word2Vec
        w2v = Word2Vec(corpus, size=size, workers=16, sg=1, max_final_vocab=3000)
        w2v.save(self.w2v_path + 'base_node_w2v_' + str(size))

    # generate block sequences with index representations
    def generate_block_seqs(self):

        from gensim.models.word2vec import Word2Vec

        word2vec = Word2Vec.load(self.w2v_path).wv
        vocab = word2vec.vocab
        max_token = word2vec.syn0.shape[0]

        def tree_to_index(node):
            token = node.token
            children = node.children
            if node.description == 'ORIGIN':
                result = [vocab['SEGMENTATION'].index, [vocab[token].index if token in vocab else max_token]]
            else:
                result = [vocab[token].index if token in vocab else max_token]
                for child in children:
                    result.append(tree_to_index(child))
            return result

        def trans2seq(r):
            blocks = []
            get_blocks_v1(r, blocks)
            tree = []
            for b in blocks:
                btree = tree_to_index(b)
                tree.append(btree)
            return tree

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
                print(str(code))

        trees['code'] = temp
        trees['code'] = trees['code'].fillna('null')
        trees = trees[~(trees['code'] == 'null')]

        if 'label' in trees.columns:
            trees.drop('label', axis=1, inplace=True)
        self.blocks = trees

    def merge(self, data_path):
        pairs = pd.read_pickle(data_path)
        pairs['id1'] = pairs['id1'].astype(int)
        pairs['id2'] = pairs['id2'].astype(int)
        df = pd.merge(pairs, self.blocks, how='left', left_on='id1', right_on='id')
        df = pd.merge(df, self.blocks, how='left', left_on='id2', right_on='id')
        df.drop(['id_x', 'id_y'], axis=1, inplace=True)
        df.dropna(inplace=True)

        # df.to_csv(self.root + '/blocks.csv')
        # Files are too big for pickle to save, so I tried joblib
        # from sklearn.externals import joblib
        # joblib.dump(df, self.root + '/blocks.pkl')

        df.to_pickle(self.root + '/blocks.pkl')

    # run for processing data to train
    def run(self):
        print('parse source code...')
        self.parse_source(output_file='ast.pkl', option='existing')
        print('read id pairs...')
        self.read_pairs('lables.csv')
        self.split_data()
        self.dictionary_and_embedding(128)
        print('split data...')
        self.split_data()
        print('generate block sequences...')
        self.generate_block_seqs()
        print('merge pairs and blocks...')
        self.merge(self.train_file_path)


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

project_name = args.project_name
bug_id = args.bug_id
if args.predict_baseline == 'true':
    base_url = 'simfix_supervised_data/' + project_name + '/' + bug_id + '/'
else:
    base_url = 'simfix_unsupervised_data/' + project_name + '/' + bug_id + '/'

ppl = Pipeline(base_url, w2v_path='all_words_embedding/all_words_w2v_30000')
ppl.run()
