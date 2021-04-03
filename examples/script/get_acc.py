import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--supernet', type=str)
    parser.add_argument('--gpu', type=int)
    args = parser.parse_args()
    print(args.supernet)

    cnt = 0
    with open(f'logs/super/eval_subnet_tq_ratiorand_'
              f'insuper_{args.supernet}.txt',
              'r') as rfid:
        for line in rfid:
            if 'Accuracy' in line:
                cnt += 1
                if cnt % 2:
                    print(eval(line.split(' ')[-1]))
