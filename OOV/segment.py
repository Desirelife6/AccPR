import re


def clear_text(origin_str):
    sub_str = re.sub(u"([^\u4e00-\u9fa5^a-z^A-Z^!^?^>^<^=^&^|^~^%^/^+^*^_^ ^.^-^:^,^@^-])", "", origin_str)
    return sub_str


pattern = r',|\.|;|\'|`|\[|\]|:|"|\{|\}|@|#|\$|\(|\)|\_|，|。|、|；|‘|’|【|】|·|！| |…|（|）:| |'

words = [
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
#
# for root in words:
root = 'length must be valid'
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
        token = x.lower()
        # children = []
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
                res.append(i)
children = res
print(children)