import re
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
import nltk
from nltk.corpus import stopwords


def clear_text(origin_str):
    sub_str = re.sub(u"([^\u4e00-\u9fa5^a-z^A-Z^!^?^>^<^=^&^|^~^%^/^+^*^_^ ^.^-^:^,^@^-])", "", origin_str)
    return sub_str


wnl = WordNetLemmatizer()

pattern = r',|\.|;|\'|`|\[|\]|:|"|\{|\}|@|#|\$|\(|\)|\_|，|。|、|；|‘|’|【|】|·|！| |…|（|）:| |'
operators = ['<', '>', '<=', '>=', '==', '&&', '||', '%', '!', '!=', '+', '-', '*', '/', '^', '&', '|', '~', '+=', '-=',
             '*=', '/=', '|=', '&=', '^=', '>>', '<<']


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


words = [
    '-=',
    '&&',
    'myListener',
    'simplified',
    'FastDateFormat',
    '-',
    '?=',
    '==',
    '>=',
    '<=',
    '||',
    '&&',
    '^',
    '*',
    '/',
    '+',
    '！=',
    'org.apache.commons.lang3.StringUtilsTest::testReplace_StringStringArrayStringArray',
    'addSWTListener',
    'createD4JCmd',
    'CatClause',
    'catClause',
    'FastDateFormat',
    'testNullBlock_FF',
    'JSError',
    'getASTNodes',
    'q1q2Me22',
    'org.apache.commons.lang3.StringUtilsTest::testReplace_StringStringArrayStringArray',
    'PlotOrientation.HORIZONTAL',
    'left-insets.right-this',
    'The month argument must be in the range 1 to 12.',
    'testIterateRangeBounds_IntervalCategoryDataset',
    'The radius cannot be negative.',
    'INFINITE_VALUE_CONVERSION',
    'testmath59',
    '_condition_replace',
    'TARGET_PROP',
    'Expecting the lower bound of the range to be around -30: ""+range.getLowerBound(),range.getLowerBound()<=-30']

for root in words:
    # root = 'length must be valid'
    root = clear_text(root)
    root = re.split(pattern, root)
    root = [x for x in root if x != '']
    # print(root)
    res = []
    for x in root:
        tmp = re.split(pattern, x)
        big_chars = re.findall(r'[A-Z]', x)
        if (x.islower() or x.isupper() or len(big_chars) == 0 or (
                x[0].isupper() and len(big_chars) == 1)) and len(tmp) == 1:
            if x not in stopwords.words('english'):
                if x not in operators:
                    tokens = nltk.word_tokenize(x.lower())
                    tag = nltk.pos_tag(tokens)
                    wnl = WordNetLemmatizer()
                    if len(tag) != 0:
                        wordnet_pos = get_wordnet_pos(tag[0][1]) or wordnet.NOUN
                        token = wnl.lemmatize(tag[0][0], pos=wordnet_pos)
                else:
                    token = x.lower()
            else:
                token = ''
            res.append(token)
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

    print(children)
