import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rc
rc('font',**{'family':'sans-serif','sans-serif':['Arial'], 'size': 14})
# rc('font',**{'family':'serif','serif':['Times']})
# rc('text', usetex=True)

data_a = [[0.6968968006,
           0.675006014,
           0.4690882848,
           0.7534279529,
           0.6721193168,
           0.7163820063,
           0.6769304787,
           0.7709886938],
          [0.7914361318,
           0.7748376233,
           0.7697859033,
           0.7936011547,
           0.7421217224,
           0.7668992062,
           0.7803704595,
           0.7863844118],
          [0.8015395718,
           0.8051479432,
           0.7820543661,
           0.8058696175,
           0.802020688,
           0.8061101756,
           0.7899927833,
           0.7988934328],
          [0.8104402213,
           0.8092374308,
           0.7861438537,
           0.8099591051,
           0.8080346404,
           0.8176569642,
           0.8077940823,
           0.8191003127]
          ]

data_b = [[0.3233333333,
           0.5066666667,
           0.2933333333,
           0.4833333333,
           0.2566666667,
           0.59,
           0.2966666667,
           0.3566666667,],
          [0.1233333333,
           0.45,
           0.4633333333,
           0.3566666667,
           0.2233333333,
           0.37,
           0.4166666667,
           0.3166666667,
           ],
          [0.1833333333,
           0.2833333333,
           0.1566666667,
           0.2933333333,
           0.3233333333,
           0.2133333333,
           0.1733333333,
           0.19,],
          [
              0.2866666667,
              0.26,
              0.2266666667,
              0.2733333333,
              0.2433333333,
              0.23,
              0.26,
              0.3433333333,
          ]
          ]

hfont = {'fontname':'Arial'}
ticks = ['45', '90', '145', '180']

def set_box_color(bp, color):
    plt.setp(bp['boxes'], color=color)
    plt.setp(bp['whiskers'], color=color)
    plt.setp(bp['caps'], color=color)
    plt.setp(bp['medians'], color=color)

plt.figure()

bpl = plt.boxplot(data_a, positions=np.array(range(len(data_a)))*2.0-0.4, sym='', widths=0.6)
bpr = plt.boxplot(data_b, positions=np.array(range(len(data_b)))*2.0+0.4, sym='', widths=0.6)
set_box_color(bpl, '#00AEBB') # colors are from http://colorbrewer2.org/
set_box_color(bpr, '#F8B62D')

# draw temporary red and blue lines and use them to create a legend
plt.plot([], c='#00AEBB', label='Noise-Free Simulation', )
plt.plot([], c='#F8B62D', label='Measured on IBMQ-Yorktown')
plt.legend()

plt.xticks(range(0, len(ticks) * 2, 2), ticks)
plt.xlabel('Number of Parameters')
plt.ylabel('MNIST-4 Accuracy')
plt.xlim(-2, len(ticks)*2)
plt.ylim(0, 1)
plt.tight_layout()

plt.show()