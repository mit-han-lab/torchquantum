from torchquantum.measurement import find_observable_groups
from random import shuffle

def test_find_observable_groups():
    in1 = ["XXIII", "YZXXX", "YZIXX", "YZIIX", "YZIIY", 
           "YZIYI", "YZIYZ", "IZIYI", "ZZZZZ", "ZZZZI",
           "IZIIX", "XIZZX"]
    out1 = find_observable_groups(in1)
    assert out1 == {'YZIYZ': ['YZIYZ'], 'YZIYY': ['YZIIY', 'YZIYI', 'IZIYI'], 'ZZZZZ': ['ZZZZZ', 'ZZZZI'], 'YZXXX': ['YZXXX', 'YZIXX', 'YZIIX', 'IZIIX'], 'XXZZX': ['XXIII', 'XIZZX']}
    # print(out1)

def find_observable_groups_multi():
    in1 = ["XXIII", "YZXXX", "YZIXX", "YZIIX", "YZIIY", 
           "YZIYI", "YZIYZ", "IZIYI", "ZZZZZ", "ZZZZI",
           "IZIIX", "XIZZX"]
    for _ in range(100):
        shuffle(in1)
        print(find_observable_groups(in1))


if __name__ == '__main__':
    test_find_observable_groups()
    # find_observable_groups_multi()

