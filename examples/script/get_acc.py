import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('file', type=str)
    parser.add_argument('--loss', action='store_true')

    args = parser.parse_args()
    print(args.file)

    info = 'Loss' if args.loss else 'Accuracy'
    print(info)
    cnt = 0
    with open(args.file, 'r') as rfid:
        for line in rfid:
            if info in line:
                cnt += 1
                if cnt % 2:
                    print(eval(line.split(' ')[-1]))
