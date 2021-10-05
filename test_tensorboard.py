from tensorflow.python.summary.summary_iterator import summary_iterator
import tensorflow as tf
import pdb

pdb.set_trace()
path = '/NFS/home/zirui/repo/pytorch-quantum/runs/mnist_front500.four0123.4qubits.train.noaddnoise.nonorm.seth_0.n1b3.ibmq_santiago.realqcTrainRealqcValid/tensorboard/events.out.tfevents.1632838148.r1.mit.edu.3177147.0'
for summary in summary_iterator(path):
    print(summary.step)
    print(summary.summary.value)
    print(summary.summary.value[0].tag)
    print(summary.summary.value[0].simple_value)
