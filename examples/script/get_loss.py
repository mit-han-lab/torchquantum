if __name__ == '__main__':
    cnt = 0
    with open('logs/eval_subnet_tq_insuper_ldiff7.txt') as rfid:
        for line in rfid:
            if 'Loss' in line:
                cnt += 1
                if cnt % 2:
                    print(eval(line.split(' ')[-1]))
