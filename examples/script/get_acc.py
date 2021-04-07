import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--supernet', type=str)
    parser.add_argument('--gpu', type=int)
    args = parser.parse_args()
    print(args.supernet)

    cnt = 0
    with open(
            f'logs/sfsuper/eval_subnet_noise_x2_opt2_ratio_500insuper_u3cu3_s0'
            f'.plain.blk8s1_ws1_os1.txt',
              'r') as rfid:
        for line in rfid:
            if 'Accuracy' in line:
                cnt += 1
                if cnt % 2:
                    print(eval(line.split(' ')[-1]))
