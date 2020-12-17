# 定义函数来显示柱状上的数值
def autolabel(max, rects):
    for rect in rects:
        tmp = rect.get_height()
        height = max - rect.get_height()
        if -1000 < tmp <= 0:
            plt.text(rect.get_x(), 1.03, 'XD')
        elif tmp <= -1000:
            plt.text(rect.get_x(), 1.03, 'Failed')
        else:
            plt.text(rect.get_x(), 1.03 * tmp, '%s' % int(height))


import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator
import numpy as np

chart_ids = [1, 3, 7, 20]
chart_baseline = 10 - np.array([7, 9, 1, 1])
chart_super_astnn = 10 - np.array([0, 388, 1, 8])

plt.ylim(0, 12)
plt.yticks([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
           [10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0])

x = list(range(len(chart_ids)))
total_width, n = 1.2, 4
width = total_width / n
a = plt.bar(x, chart_baseline, width=width, label='Baseline', tick_label=chart_ids, fc='hotpink')
for i in range(len(x)):
    x[i] = x[i] + width
b = plt.bar(x, chart_super_astnn, width=width, label='Supervised ASTNN', fc='aquamarine')

autolabel(10, a)
autolabel(10, b)

plt.legend()
plt.savefig('./visual_result/chart_result_1.jpg')
plt.clf()

# plt.ylim(0, 12)
# plt.yticks([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
#            [10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0])
#
# plt.xlabel("Id")
# plt.ylabel("Rank")
# plt.xticks(chart_ids)
# # x_major_locator=MultipleLocator(1)
# # plt.gca().xaxis.set_major_locator(x_major_locator)
# plt.plot(chart_ids, chart_baseline, linewidth=2, color='hotpink', linestyle=':',
#          label='Baseline rank')
# plt.plot(chart_ids, chart_super_astnn, linewidth=2, color='aquamarine', linestyle='-',
#          label='Supervised ASTNN rank')
# plt.legend(['Baseline', 'Supervised ASTNN'], loc='upper left')
# plt.savefig('./visual_result/chart_result_2.jpg')
# plt.clf()

closure_ids = [14, 57, 62, 73, 115]
closure_baseline = 60 - np.array([21, 58, 22, 1, 14])
closure_super_astnn = 60 - np.array([333, 26, 19, 25, 9])

tmp = np.linspace(0, 60, 61)

plt.ylim(0, 60)
plt.yticks(tmp,
           60 - tmp)
y_major_locator = MultipleLocator(10)
plt.gca().yaxis.set_major_locator(y_major_locator)
x = list(range(len(closure_ids)))
total_width, n = 1.2, 5
width = total_width / n
a = plt.bar(x, closure_baseline, width=width, label='Baseline', tick_label=closure_ids, fc='hotpink')
for i in range(len(x)):
    x[i] = x[i] + width
b = plt.bar(x, closure_super_astnn, width=width, label='Supervised ASTNN', fc='aquamarine')
autolabel(60, a)
autolabel(60, b)
plt.legend()
plt.savefig('./visual_result/closure_result_1.jpg')
plt.clf()

# tmp = np.linspace(0, 60, 61)
#
# plt.ylim(0, 60)
# plt.yticks(tmp,
#            60 - tmp)
# y_major_locator = MultipleLocator(10)
# plt.gca().yaxis.set_major_locator(y_major_locator)
#
# plt.xlabel("Id")
# plt.ylabel("Rank")
# plt.xticks(closure_ids)
# plt.plot(closure_ids, closure_baseline, linewidth=2, color='hotpink', linestyle=':',
#          label='Baseline rank')
# plt.plot(closure_ids, closure_super_astnn, linewidth=2, color='aquamarine', linestyle='-',
#          label='Supervised ASTNN rank')
# plt.legend(['Baseline', 'Supervised ASTNN'], loc='upper left')
# plt.savefig('./visual_result/closure_result_2.jpg')
# plt.clf()

lang_ids = [16, 27, 33, 39, 41, 43, 50, 58, 60]
lang_baseline = 205 - np.array([150, 200, 101, 34, 2, 3, 7, 1, 9999])
lang_super_astnn = 205 - np.array([119, 137, 32, 9999, 1, 9999, 5, 11, 9999])

tmp = np.linspace(0, 205, 206)

plt.ylim(0, 220)
plt.yticks(tmp,
           220 - tmp)

y_major_locator = MultipleLocator(20)
plt.gca().yaxis.set_major_locator(y_major_locator)
x = list(range(len(lang_ids)))
total_width, n = 2.4, 8
width = total_width / n
a = plt.bar(x, lang_baseline, width=width, label='Baseline', tick_label=lang_ids, fc='hotpink')
for i in range(len(x)):
    x[i] = x[i] + width
b = plt.bar(x, lang_super_astnn, width=width, label='Supervised ASTNN', fc='aquamarine')

autolabel(220, a)
autolabel(220, b)

plt.legend()
plt.savefig('./visual_result/lang_result_1.jpg')
plt.clf()

math_ids = [5, 33, 35, 41, 50, 53, 57, 59, 63, 70, 71, 75, 79, 98]
math_baseline = 50 - np.array([4, 9999, 9999, 15, 2, 1, 25, 7, 49, 1, 8, 26, 3, 9999])
math_super_astnn = 50 - np.array([18, 9999, 9999, 28, 54, 3, 9, 3, 8, 2, 45, 20, 1, 9999])

tmp = np.linspace(0, 50, 51)

plt.ylim(0, 52)
plt.yticks(tmp,
           52 - tmp)

y_major_locator = MultipleLocator(10)
plt.gca().yaxis.set_major_locator(y_major_locator)
x = list(range(len(math_ids)))
total_width, n = 3.2, 14
width = total_width / n
a = plt.bar(x, math_baseline, width=width, label='Baseline', tick_label=math_ids, fc='hotpink')
for i in range(len(x)):
    x[i] = x[i] + width
b = plt.bar(x, math_super_astnn, width=width, label='Supervised ASTNN', fc='aquamarine')

autolabel(50, a)
autolabel(50, b)

plt.legend()
plt.savefig('./visual_result/math_result_1.jpg')
plt.clf()
