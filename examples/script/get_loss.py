import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--supernet', type=str)
    parser.add_argument('--gpu', type=int)
    args = parser.parse_args()
    print(args.supernet)

    cnt = 0
    with open(f'logs/sfsuper/eval_subnet_noise_x2_opt2_ratiorand_'
              f'insuper_{args.supernet}.txt',
              'r') as rfid:
        for line in rfid:
            if 'Loss' in line:
                cnt += 1
                if cnt % 2:
                    print(eval(line.split(' ')[-1]))
