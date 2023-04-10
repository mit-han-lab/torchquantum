from torchquantum.measurement import find_observable_groups


def test_find_observable_groups():
    in1 = ["XXIII", "YZXXX", "YZIXX", "YZIIX", "YZIIY", 
           "YZIYI", "YZIYZ", "IZIYI", "ZZZZZ", "ZZZZI",
           "IZIIX", "XIZZX"]
    out1 = find_observable_groups(in1)
    assert out1 == {'YZIYZ': ['YZIYZ'], 'YZIYY': ['YZIIY', 'YZIYI', 'IZIYI'], 'ZZZZZ': ['ZZZZZ', 'ZZZZI'], 'YZXXX': ['YZXXX', 'YZIXX', 'YZIIX', 'IZIIX'], 'XXZZX': ['XXIII', 'XIZZX']}
    # print(out1)

if __name__ == '__main__':
    test_find_observable_groups()
