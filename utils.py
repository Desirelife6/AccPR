from javalang.ast import Node
import re
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
import nltk
from nltk.corpus import stopwords

wnl = WordNetLemmatizer()

pattern = r',|\.|;|\'|`|\[|\]|:|"|\{|\}|@|#|\$|\(|\)|\_|，|。|、|；|‘|’|【|】|·|！| |…|（|）:| |'

operators = ['<', '>', '<=', '>=', '==', '&&', '||', '%', '!', '!=', '+', '-', '*', '/', '^', '&', '|', '~', '+=', '-=',
             '*=', '/=', '|=', '&=', '^=', '>>', '<<']


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
                if node not in stopwords.words('english'):
                    if node not in operators:
                        tokens = nltk.word_tokenize(node.lower())
                        tag = nltk.pos_tag(tokens)
                        wnl = WordNetLemmatizer()
                        if len(tag) != 0:
                            wordnet_pos = get_wordnet_pos(tag[0][1]) or wordnet.NOUN
                            token = wnl.lemmatize(tag[0][0], pos=wordnet_pos)
                    else:
                        token = node.lower()
                else:
                    token = ''
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
                    children = []
                else:
                    big_chars_copy = big_chars.copy()

                    for i in range(1, len(big_chars)):
                        curr_char = big_chars[i - 1]
                        next_char = big_chars[i]
                        if x.index(next_char) - x.index(curr_char) == 1:
                            if x.index(next_char) == len(x) - 1:
                                if curr_char in big_chars_copy:
                                    big_chars_copy.remove(curr_char)
                                big_chars_copy.remove(next_char)
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
                            if (i not in stopwords.words('english')):
                                if i not in operators:
                                    tokens = nltk.word_tokenize(i)
                                    tag = nltk.pos_tag(tokens)
                                    wordnet_pos = get_wordnet_pos(tag[0][1]) or wordnet.NOUN
                                    i = wnl.lemmatize(tag[0][0], pos=wordnet_pos)
                                    res.append(i)
                                else:
                                    res.append(i)
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
        # if isinstance(children, list) and len(children) == 1 and children[0] == self.node.token:
        #     children = []
        if self.token in logic:
            return [BlockNode(children[0])]
        elif self.token in ['MethodDeclaration', 'ConstructorDeclaration']:
            return [BlockNode(child) for child in children]
        else:
            return [BlockNode(child) for child in children if self.get_token(child) not in logic]


def get_token(node):
    token = ''
    if isinstance(node, str):
        node = clear_text(node)
        temp = re.split(pattern, node)
        big_chars = re.findall(r'[A-Z]', node)
        if (node.islower() or node.isupper() or len(big_chars) == 0 or (
                node[0].isupper() and len(big_chars) == 1)) and len(temp) == 1:
            if node not in stopwords.words('english'):
                if node not in operators:
                    tokens = nltk.word_tokenize(node.lower())
                    tag = nltk.pos_tag(tokens)
                    wnl = WordNetLemmatizer()
                    if len(tag) != 0:
                        wordnet_pos = get_wordnet_pos(tag[0][1]) or wordnet.NOUN
                        token = wnl.lemmatize(tag[0][0], pos=wordnet_pos)
                else:
                    token = node.lower()
            else:
                token = ''
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
                            if curr_char in big_chars_copy:
                                big_chars_copy.remove(curr_char)
                            big_chars_copy.remove(next_char)
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
                        if (i not in stopwords.words('english')):
                            if i not in operators:
                                tokens = nltk.word_tokenize(i)
                                tag = nltk.pos_tag(tokens)
                                wordnet_pos = get_wordnet_pos(tag[0][1]) or wordnet.NOUN
                                i = wnl.lemmatize(tag[0][0], pos=wordnet_pos)
                                res.append(i)
                            else:
                                res.append(i)
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
    sequence.append(token)
    # if isinstance(token, list):
    #     for i in token:
    #         sequence.append(i)
    # else:
    #     if token != '':
    #         sequence.append(token)
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
