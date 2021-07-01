#!/usr/bin/env python3

import argparse
import random
import struct

def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument('--output', type=str, default='DNNMark_data.dat',
                        help='Name of the binary data file to generate')
    parser.add_argument('--size', type=int, default=2,
                        help='Size of binary file to generate (in GB)')
    parser.add_argument('--seed', type=int, default=777,
                        help='Seed for RNG')
    parser.add_argument('--double', action='store_true',
                        help='Store values as doubles (floats by default)')

    return parser.parse_args()

def main():
    args = parse_arguments()

    num_nums = int(args.size*1024*1024*1024/4) # Size in GB, divided by sizeof float
    print(num_nums)

    random.seed(args.seed)

    with open(args.output, "wb") as f:
        for i in range(num_nums):
            f.write(struct.pack(f'{"d" if args.double else "f"}', random.random()))

if __name__ == '__main__':
    main()
