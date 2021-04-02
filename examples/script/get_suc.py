if __name__ == '__main__':
    with open('logs/eval_subnet_suc.txt') as rfid:
        for line in rfid:
            if 'Success' in line:
                print(eval(line.split(' ')[-1]))
