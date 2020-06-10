#!/usr/bin/env python3

from scapy.all import *
import os
import argparse
import sys


load_layer("ip")
load_layer("tls")

def scale(len):
    if(len > 1504):
        len = 1504
        
    return(int(len/32))

maxlen = int(1504/32)
chains = {}


def newchain():
    c = [0 for x in range(maxlen+1)]
    return(c)

def updatechain(who, slot):
    if who not in chains:
        chains[who] = newchain()

    chains[who][slot] = chains[who][slot] + 1

    
def markov(src, dst, len):
    slot = scale(len) 
    updatechain(src, slot)
    updatechain(dst, slot)
        

def pkt_parser(pkt):
    if pkt.haslayer('TLS'):
        ip = pkt.getlayer(IP)
        tcp = pkt.getlayer(TCP)
        tls = pkt.getlayer(TLS)

        markov(ip.src+":"+str(tcp.sport), ip.dst+":"+str(tcp.dport), tls.len)

def printchain(a):
    #print('-->', a)
    s = sum(a)
    for x in range(maxlen):
        a[x] = int((a[x]*100)/s)

    print(a)
    

##############################################################################################################
# MAIN

def main():
    parser = argparse.ArgumentParser(description='DoH bins extractor')
    parser.add_argument('pcap', type=str, help='pcap file')
    parser.add_argument('dsn_ip_addresses', type=str, help='list of dns ip addresses separated by comma')

    args = parser.parse_args()

    f = 'tcp and port 443'
    if args.dsn_ip_addresses:
        hosts = []
        for h in args.dsn_ip_addresses.split(','):
            hosts.append('host {}'.format(h))
        if hosts:
            f += ' and ({})'.format(' or '.join(hosts))

    pkts = sniff(offline=args.pcap, prn=pkt_parser, filter=f)
    for host in chains:
        printchain(chains[host])

if __name__ == '__main__':
    main()
