dir = 'examples/configs/mnist_front500/four0123/4qubits/train/noaddnoise/nonorm/seth_0/n1b3/ibmq_lima/'
cmd_list = []
for accumulation_window_size in [1, 2, 3, 5]:
    for sampling_ratio in [0.8, 0.7, 0.5, 0.3]:
        for sampling_window_size in [1, 2, 3, 5]:
            filename = 'clsTrainClsValid_phase_{0}_{1}_{2}.yml'.format(accumulation_window_size, sampling_ratio, sampling_window_size)
            full_filename = dir + filename
            cmd = 'python3 examples/train.py ' + full_filename
            cmd_list.append(cmd)

for i, cmd in enumerate(cmd_list):
    with open('run_on_gpu{0}.sh'.format(i % 8), 'a') as f:
        f.write(cmd + ' --gpu={0}\n'.format(i % 8))