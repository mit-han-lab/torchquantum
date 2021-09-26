dir = 'examples/configs/mnist_front500/four0123/4qubits/train/noaddnoise/nonorm/seth_0/n1b3/ibmq_lima/'
for accumulation_window_size in [1, 2, 3, 5]:
    for sampling_ratio in [0.8, 0.7, 0.5, 0.3]:
        for sampling_window_size in [1, 2, 3, 5]:
            filename = 'clsTrainClsValid_phase_{0}_{1}_{2}.yml'.format(accumulation_window_size, sampling_ratio, sampling_window_size)
            file_content = '''model:
  arch:
    sampling_method: phase_based_sampling
    accumulation_window_size: {0}
    sampling_ratio: {1}
    sampling_window_size: {2}

qiskit:
  use_qiskit_train: False
  use_qiskit_valid: False
  backend_name: ibmq_lima
  noise_model_name: ibmq_lima
            '''.format(accumulation_window_size, sampling_ratio, sampling_window_size)
            full_filename = dir + filename
            with open(full_filename, 'w') as f:
                f.write(file_content)