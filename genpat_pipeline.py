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
                faulty_code_file = 'parse_faulty_code.txt'
                out = open(faulty_code_file, 'a+')
                out.write('Code snippet failed to pass parsing')
                out.write('\n')
                out.write(str(code))
                out.write('\n')
                out.write('\n')
                # print('Code snippet failed to pass parsing')
                # print(str(code))
                out.close()
                code = None
                tmp.append(code)

        source['code'] = tmp
        # source['code'] = source['code'].apply(parse_program)
        source['code'] = source['code'].fillna('null')

        faults = source.loc[source['code'] == 'null']

        self.fault_ids = faults['id']
        #
        # for fault_id in self.fault_ids:
        #     print(fault_id)

        source = source[~source['code'].isin(['null'])]
        # source.to_pickle(path)
        self.sources = source

        return source

    #
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

        data = data.sample(frac=1, random_state=666)
        train = data.iloc[:]

        def check_or_create(path):
            if not os.path.exists(path):
                os.mkdir(path)

        train_path = data_path + 'train/'
        check_or_create(train_path)
        self.train_file_path = train_path + 'train_.pkl'
        train.to_pickle(self.train_file_path)

    # generate block sequences with index representations
    def generate_block_seqs(self):

        from gensim.models.word2vec import Word2Vec

        word2vec = Word2Vec.load(self.w2v_path).wv
        vocab = word2vec.vocab
        max_token = word2vec.syn0.shape[0]

        def tree_to_index(node):
            token = node.token
            result = [vocab[token].index if token in vocab else max_token]
            children = node.children
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

        df.to_pickle(self.root + '/blocks.pkl')

    # run for processing data to train
    def run(self):
        print('parse source code...')
        self.parse_source(output_file='ast.pkl', option='existing')
        print('read id pairs...')
        self.read_pairs('lables.csv')
        print('split data...')
        self.split_data()
        print('generate block sequences...')
        self.generate_block_seqs()
        print('merge pairs and blocks...')
        self.merge(self.train_file_path)


ppl = Pipeline('genpat_data/', w2v_path='all_words_embedding/all_words_w2v_30000')
ppl.run()
