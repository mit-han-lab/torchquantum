if __name__ == '__main__':
    cnt = 0
    with open('logs/eval_subnet_tq_rand.txt') as rfid:
        for line in rfid:
            if 'Accuracy' in line:
                cnt += 1
                if cnt % 2:
                    print(eval(line.split(' ')[-1]))
