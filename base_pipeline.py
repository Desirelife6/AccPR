from javalang.ast import Node
import re
import warnings
import os
from tqdm import tqdm
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
import nltk


def clear_text(origin_str):
    sub_str = re.sub(u"([^\u4e00-\u9fa5^a-z^A-Z^!^?^>^<^=^&^|^~^%^/^+^*^_^ ^.^-^:^,^@^-])", "", origin_str)
    return sub_str


# 获取单词的词性
def get_wordnet_pos(tag):
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    else:
        return None


pattern = r',|\.|;|\'|`|\[|\]|:|"|\{|\}|@|#|\$|\(|\)|\_|，|。|、|；|‘|’|【|】|·|！| |…|（|）:| |'


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

            node = clear_text(node)
            temp = re.split(pattern, node)
            big_chars = re.findall(r'[A-Z]', node)
            if (node.islower() or node.isupper() or len(big_chars) == 0 or (
                    node[0].isupper() and len(big_chars) == 1)) and len(temp) == 1:
                token = node.lower()
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
            root = clear_text(root)
            root = re.split(pattern, root)
            root = [x for x in root if x != '']
            # print(root)
            res = []
            for x in root:
                temp = re.split(pattern, x)
                big_chars = re.findall(r'[A-Z]', x)
                if (x.islower() or x.isupper() or len(big_chars) == 0 or (
                        x[0].isupper() and len(big_chars) == 1)) and len(temp) == 1:
                    # token = x.lower()
                    children = []
                    # res.append(token)
                else:
                    big_chars_copy = big_chars.copy()

                    for i in range(1, len(big_chars)):
                        curr_char = big_chars[i - 1]
                        next_char = big_chars[i]
                        if x.index(next_char) - x.index(curr_char) == 1:
                            if x.index(next_char) == len(x) - 1:
                                try:
                                    if curr_char in big_chars_copy:
                                        big_chars_copy.remove(curr_char)
                                    big_chars_copy.remove(next_char)
                                except:
                                    print('Error happened while removing some chars')
                                    print(curr_char)
                                    print(big_chars_copy)
                            else:
                                if not x[x.index(next_char) + 1].islower():
                                    big_chars_copy.remove(next_char)

                    big_chars = big_chars_copy

                    index = []
                    tmp = []
                    if len(big_chars):
                        if x.index(big_chars[0]) != 0:
                            index.append(0)
                        for bigchar in big_chars:
                            index_list = [i.start() for i in re.finditer(bigchar, x)]
                            if len(index_list) != 1:
                                for i in index_list:
                                    if not (i in index):
                                        index.append(i)
                            else:
                                index.append(x.index(bigchar))
                        index.append(len(x))
                        index = list(set(index))
                        index.sort()
                        for i in range(len(index) - 1):
                            tmp.append(x[index[i]: index[i + 1]].lower())
                        for i in list(tmp):
                            res.append(i)
            for i in range(len(children)):
                tokens = nltk.word_tokenize(children[i])
                tag = nltk.pos_tag(tokens)
                wnl = WordNetLemmatizer()
                wordnet_pos = get_wordnet_pos(tag[0][1]) or wordnet.NOUN
                children[i] = wnl.lemmatize(tag[0][0], pos=wordnet_pos)
            children = res
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
        token = node
        node = clear_text(node)
        temp = re.split(pattern, node)
        big_chars = re.findall(r'[A-Z]', node)
        if (node.islower() or node.isupper() or len(big_chars) == 0 or (
                node[0].isupper() and len(big_chars) == 1)) and len(temp) == 1:
            token = node.lower()
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
        root = clear_text(root)
        root = re.split(pattern, root)
        root = [x for x in root if x != '']
        # print(root)
        res = []
        for x in root:
            temp = re.split(pattern, x)
            big_chars = re.findall(r'[A-Z]', x)
            if (x.islower() or x.isupper() or len(big_chars) == 0 or (
                    x[0].isupper() and len(big_chars) == 1)) and len(temp) == 1:
                # token = x.lower()
                children = []
                # res.append(token)
            else:
                big_chars_copy = big_chars.copy()

                for i in range(1, len(big_chars)):
                    curr_char = big_chars[i - 1]
                    next_char = big_chars[i]
                    if x.index(next_char) - x.index(curr_char) == 1:
                        if x.index(next_char) == len(x) - 1:
                            try:
                                if curr_char in big_chars_copy:
                                    big_chars_copy.remove(curr_char)
                                big_chars_copy.remove(next_char)
                            except:
                                print(curr_char)
                                print(big_chars_copy)
                                print(x)
                        else:
                            if not x[x.index(next_char) + 1].islower():
                                big_chars_copy.remove(next_char)

                big_chars = big_chars_copy

                index = []
                tmp = []
                if len(big_chars):
                    if x.index(big_chars[0]) != 0:
                        index.append(0)
                    for bigchar in big_chars:
                        index_list = [i.start() for i in re.finditer(bigchar, x)]
                        if len(index_list) != 1:
                            for i in index_list:
                                if not (i in index):
                                    index.append(i)
                        else:
                            index.append(x.index(bigchar))
                    index.append(len(x))
                    index = list(set(index))
                    index.sort()
                    for i in range(len(index) - 1):
                        tmp.append(x[index[i]: index[i + 1]].lower())
                    for i in list(tmp):
                        res.append(i)
        for i in range(len(children)):
            tokens = nltk.word_tokenize(children[i])
            tag = nltk.pos_tag(tokens)
            wnl = WordNetLemmatizer()
            wordnet_pos = get_wordnet_pos(tag[0][1]) or wordnet.NOUN
            children[i] = wnl.lemmatize(tag[0][0], pos=wordnet_pos)
        children = res
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


class Pipeline:
    def __init__(self, ratio, root, w2v_path):
        self.ratio = ratio
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
        path = self.root + output_file
        if os.path.exists(path) and option == 'existing':
            source = pd.read_pickle(path)
        else:
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

            source = pd.read_csv(self.root + 'bcb_funcs_all.tsv', sep='\t', header=None,
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
                    faulty_code_file = 'all_words_embedding/faulty_code.txt'
                    out = open(faulty_code_file, 'a+')
                    out.write('Code snippet failed to pass parsing')
                    out.write(str(code))
                    print('Error happened while parsing')
                    print(code)
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
            source.to_pickle(path)
        self.sources = source

        return source

    # create clone pairs
    def read_pairs(self, filename):
        pairs = pd.read_pickle(self.root + filename)
        if not self.fault_ids.empty:
            for fault_id in self.fault_ids:
                pairs = pairs[~pairs['id1'].isin([fault_id])]
                pairs = pairs[~pairs['id2'].isin([fault_id])]
        self.pairs = pairs

    # split data for training, developing and testing
    def split_data(self):
        data_path = self.root
        data = self.pairs
        data_num = len(data)
        ratios = [int(r) for r in self.ratio.split(':')]
        train_split = int(ratios[0] / sum(ratios) * data_num)
        val_split = train_split + int(ratios[1] / sum(ratios) * data_num)

        data = data.sample(frac=1, random_state=666)
        train = data.iloc[:train_split]
        dev = data.iloc[train_split:val_split]
        test = data.iloc[val_split:]

        def check_or_create(path):
            if not os.path.exists(path):
                os.mkdir(path)

        train_path = data_path + 'train/'
        check_or_create(train_path)
        self.train_file_path = train_path + 'train_.pkl'
        train.to_pickle(self.train_file_path)

        dev_path = data_path + 'dev/'
        check_or_create(dev_path)
        self.dev_file_path = dev_path + 'dev_.pkl'
        dev.to_pickle(self.dev_file_path)

        test_path = data_path + 'test/'
        check_or_create(test_path)
        self.test_file_path = test_path + 'test_.pkl'
        test.to_pickle(self.test_file_path)

    # construct dictionary and train word embedding
    # def dictionary_and_embedding(self, input_file, size):
    #     self.size = size
    #     if not input_file:
    #         input_file = self.train_file_path
    #     pairs = pd.read_pickle(input_file)
    #     train_ids = pairs['id1'].append(pairs['id2']).unique()
    #
    #     trees = self.sources.set_index('id', drop=False).loc[train_ids]
    #     if not os.path.exists(self.w2v_path):
    #         os.mkdir(self.w2v_path)
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
    #     w2v = Word2Vec(corpus, size=size, workers=16, sg=1, max_final_vocab=3000)
    #     w2v.save(self.w2v_path + 'base_node_w2v_' + str(size))

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
        trees['code'] = trees['code'].apply(trans2seq)
        if 'label' in trees.columns:
            trees.drop('label', axis=1, inplace=True)
        self.blocks = trees

    # merge pairs
    def merge(self, data_path, part):
        pairs = pd.read_pickle(data_path)
        pairs['id1'] = pairs['id1'].astype(int)
        pairs['id2'] = pairs['id2'].astype(int)
        df = pd.merge(pairs, self.blocks, how='left', left_on='id1', right_on='id')
        df = pd.merge(df, self.blocks, how='left', left_on='id2', right_on='id')
        df.drop(['id_x', 'id_y'], axis=1, inplace=True)
        df.dropna(inplace=True)

        df.to_pickle(self.root + part + '/blocks.pkl')

    # run for processing data to train
    def run(self):
        print('parse source code...')
        self.parse_source(output_file='ast.pkl', option='existing')
        print('read id pairs...')
        self.read_pairs('bcb_pair_ids.pkl')
        print('split data...')
        self.split_data()
        # print('train word embedding...')
        # self.dictionary_and_embedding(None, 128)
        print('generate block sequences...')
        self.generate_block_seqs()
        print('merge pairs and blocks...')
        self.merge(self.train_file_path, 'train')
        self.merge(self.dev_file_path, 'dev')
        self.merge(self.test_file_path, 'test')

# ppl = Pipeline('3:1:1', 'base_data/', w2v_path='all_words_embedding/all_words_w2v_300000')
# ppl.run()
