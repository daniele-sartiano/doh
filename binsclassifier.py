#!/usr/bin/env python3

import numpy as np
from scapy.all import *
import argparse
import ast

from doh import BinsGenerator


def main():
    parser = argparse.ArgumentParser(description='DoH classifier')
    parser.add_argument('pcap', type=str, help='pcap file to classify')
    parser.add_argument('vect', type=str, help='txt file with sample vectors')

    args = parser.parse_args()

    f = 'tcp and port 443'

    vectors = []
    with open(args.vect, 'r') as fvect:
        for v in fvect:
            vectors.append(np.asarray(ast.literal_eval(v.strip())))
    
    generator = BinsGenerator(args.pcap, f)
    for host in generator.chains:
        c = BinsGenerator.getchain(generator.chains[host])
        c = np.asarray(c)
        for v in vectors:
            dst = np.linalg.norm(c-v)
            print(c, dst)
        print()
    
        
if __name__ == '__main__':
    main()
