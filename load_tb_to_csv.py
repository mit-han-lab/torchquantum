from tensorflow.python.summary.summary_iterator import summary_iterator
import os
import glob
import csv

# path_list = ['fashion_front500.four0123.4qubits.train.noaddnoise.nonorm.barren_0.n1b3.ibmq_jakarta.realqcTrainRealqcValid_grad_1_0.5_2/tensorboard',
# 'fashion_front500.four0123.4qubits.train.noaddnoise.nonorm.barren_0.n1b3.ibmq_jakarta.realqcTrainRealqcValid/tensorboard',
# 'fashion_front500.four0123.4qubits.train.noaddnoise.nonorm.barren_0.n1b3.ibmq_jakarta.clsTrainRealqcValid/tensorboard']
# path_list = ['mnist_front500.two36.4qubits.train.noaddnoise.nonorm.seth_0.n1b1.ibmq_santiago.realqcTrainRealqcValid_2/tensorboard',
# 'mnist_front500.two36.4qubits.train.noaddnoise.nonorm.seth_0.n1b1.ibmq_santiago.realqcTrainRealqcValid/tensorboard',
# 'mnist_front500.two36.4qubits.train.noaddnoise.nonorm.seth_0.n1b1.ibmq_santiago.clsTrainRealqcValid/tensorboard']
# path_list = ['vowel_front.four0516.4qubits.train.noaddnoise.nonorm.barren_0.n1b3.ibmq_lima.realqcTrainRealqcValid_grad_1_0.5_2/tensorboard',
# 'vowel_front.four0516.4qubits.train.noaddnoise.nonorm.barren_0.n1b3.ibmq_lima.realqcTrainRealqcValid/tensorboard',
# 'vowel_front.four0516.4qubits.train.noaddnoise.nonorm.barren_0.n1b3.ibmq_lima.clsTrainRealqcValid/tensorboard']
# path_list = ['mnist_front500.four0123.4qubits.train.noaddnoise.nonorm.seth_0.n1b3.ibmq_manila.realqcTrainRealqcValid_grad_1_0.3_2/tensorboard',
# 'mnist_front500.four0123.4qubits.train.noaddnoise.nonorm.seth_0.n1b3.ibmq_manila.realqcTrainRealqcValid/tensorboard',
# 'mnist_front500.four0123.4qubits.train.noaddnoise.nonorm.seth_0.n1b3.ibmq_manila.clsTrainRealqcValid/tensorboard']
path_list = ['mnist_front500.four0123.4qubits.train.noaddnoise.nonorm.seth_0.n1b3.ibmq_manila.realqcTrainRealqcValid/tensorboard',
'mnist_front500.four0123.4qubits.train.noaddnoise.nonorm.seth_0.n1b3.ibmq_manila.clsTrainClsValid/tensorboard']
value_list = []
for path in path_list:
    list_step = []
    list_value = []
    a_path = 'runs/' + path + '/*'
    list_of_files = glob.glob(a_path)
    latest_file = max(list_of_files, key=os.path.getctime)
    print(latest_file)
    for summary in summary_iterator(latest_file):
        if (len(summary.summary.value)) == 0:
            continue
        tag = summary.summary.value[0].tag
        value = summary.summary.value[0].simple_value
        if tag == 'acc/test':
            list_step.append(summary.step)
            list_value.append(value)
    value_list.append(list_step)
    value_list.append(list_value)

max_len = 0
for list in value_list:
    max_len = max(max_len, len(list))

head = ['realqcTrainRealqcValid #inference', 'realqcTrainRealqcValid accu', 'clsTrainClsValid #inference', 'clsTrainClsValid accu']
with open('fashion0123_manila.csv', 'w') as f:
    writer = csv.writer(f)
    writer.writerow(head)
    for i in range(max_len):
        row = []
        for list in value_list:
            if (i < len(list)):
                row.append(list[i])
            else:
                row.append('')
        row[0] *= 50 * 49
        row[2] *= 50
        writer.writerow(row)

