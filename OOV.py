from javalang.ast import Node
import re
import warnings


def clear_text(origin_str):
    sub_str = re.sub(u"([^\u4e00-\u9fa5^a-z^A-Z^0-9^!^?^>^<^=^&^|^~^%^/^-^+^*])", "", origin_str)
    return sub_str


class BlockNode(object):
    def __init__(self, node):
        self.node = node
        self.is_str = isinstance(self.node, str)
        self.token = self.get_token(node)
        self.children = self.add_children()

    def is_leaf(self):
        if self.is_str:
            return True
        return len(self.node.children) == 0

    def get_token(self, node):
        token = ''

        if isinstance(node, str):

            # node = filter_str(node, node)
            node = clear_text(node)
            big_chars = re.findall(r'[A-Z]', node)
            if node.islower() or node.isupper() or len(big_chars) == 0 or (node[0].isupper() and len(big_chars) == 1):
                token = node
            else:
                token = 'SEGMENTATION'

        elif isinstance(node, set):
            token = 'Modifier'
        elif isinstance(node, Node):
            token = node.__class__.__name__
        else:
            token = ''
        return token

    def ori_children(self, root):
        children = []
        if isinstance(root, Node):
            if self.token in ['MethodDeclaration', 'ConstructorDeclaration']:
                children = root.children[:-1]
            else:
                children = root.children
        elif isinstance(root, set):
            children = list(root)
        elif isinstance(root, str):
            # root = filter_str(root, root)
            root = clear_text(root)
            big_chars = re.findall(r'[A-Z]', root)
            if root.islower() or root.isupper() or len(big_chars) == 0 or (root[0].isupper() and len(big_chars) == 1):
                children = []
            else:
                big_chars = re.findall(r'[A-Z]', root)

                big_chars_copy = big_chars.copy()

                for i in range(1, len(big_chars)):
                    curr_char = big_chars[i - 1]
                    next_char = big_chars[i]
                    if root.index(next_char) - root.index(curr_char) == 1:
                        if i == len(big_chars):
                            big_chars_copy.remove(curr_char)
                            big_chars_copy.remove(next_char)
                        else:
                            big_chars_copy.remove(next_char)

                big_chars = big_chars_copy

                index = []
                tmp = []
                if len(big_chars):
                    if root.index(big_chars[0]) != 0:
                        index.append(0)
                    for bigchar in big_chars:
                        index.append(root.index(bigchar))
                    index.append(len(root))
                    for i in range(len(index) - 1):
                        tmp.append(root[index[i]: index[i + 1]].lower())
                    children = list(tmp)
        else:
            children = []

        def expand(nested_list):
            for item in nested_list:
                if isinstance(item, list):
                    for sub_item in expand(item):
                        yield sub_item
                elif item:
                    yield item

        return list(expand(children))

    def add_children(self):
        # if self.is_str:
        #     return []
        logic = ['SwitchStatement', 'IfStatement', 'ForStatement', 'WhileStatement', 'DoStatement']
        children = self.ori_children(self.node)
        if self.token in logic:
            return [BlockNode(children[0])]
        elif self.token in ['MethodDeclaration', 'ConstructorDeclaration']:
            return [BlockNode(child) for child in children]
        else:
            return [BlockNode(child) for child in children if self.get_token(child) not in logic]


def get_token(node):
    token = ''
    if isinstance(node, str):
        # token = node
        # node = filter_str(node, node)
        node = clear_text(node)
        big_chars = re.findall(r'[A-Z]', node)
        if node.islower() or node.isupper() or len(big_chars) == 0 or (node[0].isupper() and len(big_chars) == 1):
            token = node
        else:
            token = 'SEGMENTATION'
    elif isinstance(node, set):
        token = 'Modifier'  # node.pop()
    elif isinstance(node, Node):
        token = node.__class__.__name__

    return token


def get_children(root):
    children = []
    if isinstance(root, Node):
        children = root.children

    elif isinstance(root, str):
        # root = filter_str(root, root)
        root = clear_text(root)
        big_chars = re.findall(r'[A-Z]', root)
        if root.islower() or root.isupper() or len(big_chars) == 0 or (root[0].isupper() and len(big_chars) == 1):
            children = []
        else:
            big_chars = re.findall(r'[A-Z]', root)

            big_chars_copy = big_chars.copy()

            for i in range(1, len(big_chars)):
                curr_char = big_chars[i - 1]
                next_char = big_chars[i]
                if root.index(next_char) - root.index(curr_char) == 1:
                    if i == len(big_chars):
                        big_chars_copy.remove(curr_char)
                        big_chars_copy.remove(next_char)
                    else:
                        big_chars_copy.remove(next_char)

            big_chars = big_chars_copy

            index = []
            tmp = []
            if len(big_chars):
                if root.index(big_chars[0]) != 0:
                    index.append(0)
                for bigchar in big_chars:
                    index.append(root.index(bigchar))
                index.append(len(root))
                for i in range(len(index) - 1):
                    tmp.append(root[index[i]: index[i + 1]].lower())
                children = list(tmp)
    elif isinstance(root, set):
        children = list(root)
    else:
        children = []

    def expand(nested_list):
        for item in nested_list:
            if isinstance(item, list):
                for sub_item in expand(item):
                    yield sub_item
            elif item:
                yield item

    return list(expand(children))


def get_sequence(node, sequence):
    token, children = get_token(node), get_children(node)
    if isinstance(token, list):
        for i in token:
            sequence.append(i)
    else:
        sequence.append(token)
    # sequence.extend()

    for child in children:
        get_sequence(child, sequence)

    if token in ['ForStatement', 'WhileStatement', 'DoStatement', 'SwitchStatement', 'IfStatement']:
        sequence.append('End')


def get_blocks_v1(node, block_seq):
    name, children = get_token(node), get_children(node)
    logic = ['SwitchStatement', 'IfStatement', 'ForStatement', 'WhileStatement', 'DoStatement']
    if name in ['MethodDeclaration', 'ConstructorDeclaration']:
        block_seq.append(BlockNode(node))
        body = node.body
        for child in body:
            if get_token(child) not in logic and not hasattr(child, 'block'):
                block_seq.append(BlockNode(child))
            else:
                get_blocks_v1(child, block_seq)
    elif name in logic:
        block_seq.append(BlockNode(node))
        for child in children[1:]:
            token = get_token(child)
            if not hasattr(node, 'block') and token not in logic + ['BlockStatement']:
                block_seq.append(BlockNode(child))
            else:
                get_blocks_v1(child, block_seq)
            block_seq.append(BlockNode('End'))
    elif name is 'BlockStatement' or hasattr(node, 'block'):
        block_seq.append(BlockNode(name))
        for child in children:
            if get_token(child) not in logic:
                block_seq.append(BlockNode(child))
            else:
                get_blocks_v1(child, block_seq)
    else:
        for child in children:
            get_blocks_v1(child, block_seq)


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
        source = pd.read_csv('simfix_all_words.tsv', sep='\t', header=None,
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
                print('Error happened while parsing')
                print(code)
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

    def dictionary_and_embedding(self, size):
        self.size = size

        trees = self.sources

        def trans_to_sequences(ast):
            sequence = []
            get_sequence(ast, sequence)
            return sequence

        corpus = trees['code'].apply(trans_to_sequences)
        str_corpus = [' '.join(c) for c in corpus]
        trees['code'] = pd.Series(str_corpus)
        # trees.to_csv(data_path+'train/programs_ns.tsv')

        from gensim.models.word2vec import Word2Vec
        w2v = Word2Vec(corpus, size=size, workers=16, sg=1, max_final_vocab=30000)
        w2v.save('simfix_node_w2v_' + str(size))

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

    # run for processing data to train
    def run(self):
        print('parse source code...')
        self.parse_source(output_file='ast.pkl', option='existing')

        self.dictionary_and_embedding(128)
        print('generate block sequences...')
        self.generate_block_seqs()


#
# ppl = OovPipeline('base_data/', w2v_path='base_embedding/base_node_w2v_128')
# ppl.run()

from gensim.models.word2vec import Word2Vec
from sklearn.decomposition import PCA
from matplotlib import pyplot

word2vec = Word2Vec.load('base_embedding/base_node_w2v_128').wv
vocab = word2vec.vocab
max_token = word2vec.syn0.shape[0]

model = Word2Vec.load('base_embedding/base_node_w2v_128')

model2 = Word2Vec.load('simfix_node_w2v_128')

import random

# 基于2d PCA拟合数据

X = model[model.wv.vocab]
X2 = model2[model2.wv.vocab]
pca = PCA(n_components=2)
pca2 = PCA(n_components=2)
result = pca.fit_transform(X)
result2 = pca2.fit_transform(X2)
# 可视化展示
pyplot.scatter(result[:, 0], result[:, 1], c='hotpink')

# pyplot.scatter(result2[:, 0], result2[:, 1], c='r')

words1 = list(model.wv.vocab)
words2 = list(model2.wv.vocab)
random.seed(10)
i = int(len(list(model.wv.vocab)) / 1000)
print(i)
slice1 = random.sample(words1, i)
slice2 = random.sample(words2, i)

words1 = slice1
words2 = slice2

for i, word in enumerate(slice1):
    pyplot.annotate(word, xy=(result[i, 0], result[i, 1]))

for i, word in enumerate(slice2):
    pyplot.annotate(word, xy=(result2[i, 0], result2[i, 1]))
pyplot.show()
