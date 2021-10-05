import json
import argparse
import os
import sys
import pdb
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import matplotlib.pyplot as plt


def sim(list1, list2):
    # list1 = np.array(list1)
    # list2 = np.array(list2)
    # return np.mean(np.abs(list1 - list2) / np.abs(list2))
    return cosine_similarity([list1], [list2])[0][0]


parser = argparse.ArgumentParser()
parser.add_argument('--run_dir', metavar='DIR', help='run directory')
args, opts = parser.parse_known_args()
print(args.run_dir)
# with open(args.run_dir, 'r') as f:
#     record = json.load(f)
#     for kv in record.items():
#         step = int(kv[0])
#         x.append(step)
#         grad_dict = kv[1]
#         classical_grad_list = grad_dict['classical']
#         low = np.min(classical_grad_list)
#         up = np.max(classical_grad_list)
#         for grad_list in grad_dict.values():
#             low = min(low, np.min(classical_grad_list))
#             up = max(up, np.max(classical_grad_list))
        
#         fig = plt.figure()
#         ax1 = fig.add_subplot(len(shots_list) + 1, 1, 1)
#         ax1.hist(classical_grad_list, range=[low, up], density=True, bins=5)
#         tmp = 1
#         for kv2 in grad_dict.items():
#             n_shots = kv2[0]
#             if n_shots == 'classical':
#                 continue
#             else:
#                 n_shots = int(n_shots)
#             grad_list = kv2[1]
#             tmp += 1
#             ax2 = fig.add_subplot(len(shots_list) + 1, 1, tmp)
#             ax2.hist(grad_list, range=[low, up], density=True, bins=5)
#             result = sim(grad_list, classical_grad_list)
#             print(result)
#             y.append(result)
#         plt.tight_layout()
#         plt.show()
#         plt.savefig(args.run_dir+'step{0}.png'.format(step))
gl = []
mrel = []
# pdb.set_trace()
with open(args.run_dir, 'r') as f:
    record = json.load(f)
    for kv in record.items():
        step = int(kv[0])
        grad_dict = kv[1]
        classical_grad_list = np.array(grad_dict['classical'])
        mre_list = []
        for i in range(4):
            grad_list = np.array(grad_dict[str(i)])
            re_list = np.abs(grad_list - classical_grad_list) / np.abs(classical_grad_list)
            mre_list.append(re_list)
        mre_list = np.array(mre_list)
        mre_list = np.mean(mre_list, axis=0)
        gl.append(classical_grad_list)
        mrel.append(mre_list)

gl = np.array(gl).reshape(-1)
mrel = np.array(mrel).reshape(-1)
idx = np.argsort(np.abs(gl))
gl = gl[idx]
mrel = mrel[idx]
plt.plot(np.abs(gl), mrel)
plt.xlabel('gradient magnitude')
plt.ylabel('MRE')
plt.savefig(args.run_dir + 'teaser.png')

import csv
with open(args.run_dir + 'data.csv', 'w') as f:
    writer = csv.writer(f)
    writer.writerow(['gradient magnitude', 'MRE'])
    for x, y in zip(np.abs(gl), mrel):
        writer.writerow([x, y])
# x = np.array(x)
# y = np.array(y).reshape(x.shape[0], -1)
# for i, shots in enumerate(shots_list):
#     plt.plot(x, y[:, i], label='n_shots='+str(shots))
# plt.xlabel('step')
# plt.ylabel('MRE')
# plt.legend()
# plt.savefig(args.run_dir+'.png')
