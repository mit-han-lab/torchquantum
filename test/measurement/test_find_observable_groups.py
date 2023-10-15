"""
MIT License

Copyright (c) 2020-present TorchQuantum Authors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

from torchquantum.measurement import find_observable_groups
from random import shuffle


def test_find_observable_groups():
    in1 = [
        "XXIII",
        "YZXXX",
        "YZIXX",
        "YZIIX",
        "YZIIY",
        "YZIYI",
        "YZIYZ",
        "IZIYI",
        "ZZZZZ",
        "ZZZZI",
        "IZIIX",
        "XIZZX",
    ]
    out1 = find_observable_groups(in1)
    assert out1 == {
        "YZIYZ": ["YZIYZ"],
        "YZIYY": ["YZIIY", "YZIYI", "IZIYI"],
        "ZZZZZ": ["ZZZZZ", "ZZZZI"],
        "YZXXX": ["YZXXX", "YZIXX", "YZIIX", "IZIIX"],
        "XXZZX": ["XXIII", "XIZZX"],
    }
    # print(out1)


def find_observable_groups_multi():
    in1 = [
        "XXIII",
        "YZXXX",
        "YZIXX",
        "YZIIX",
        "YZIIY",
        "YZIYI",
        "YZIYZ",
        "IZIYI",
        "ZZZZZ",
        "ZZZZI",
        "IZIIX",
        "XIZZX",
    ]
    for _ in range(100):
        shuffle(in1)
        print(find_observable_groups(in1))


if __name__ == "__main__":
    test_find_observable_groups()
    # find_observable_groups_multi()
