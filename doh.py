#!/usr/bin/env python3

from scapy.all import *
import argparse
import sys

load_layer("ip")
load_layer("tls")

class BinsGenerator:

    MAXLEN = int(1504/32)
    
    def __init__(self, pcap, filter):
        self.chains = {}

        pkts = sniff(offline=pcap, prn=self.pkt_parser, filter=filter)

    @staticmethod
    def scale(len):
        if(len > 1504):
            len = 1504
        return(int(len/32))
        
    def newchain(self):
        c = [0 for x in range(self.MAXLEN+1)]
        return(c)

    def updatechain(self, who, slot):
        if who not in self.chains:
            self.chains[who] = self.newchain()
        self.chains[who][slot] = self.chains[who][slot] + 1

    def markov(self, src, dst, len):
        slot = self.scale(len) 
        self.updatechain(src, slot)
        self.updatechain(dst, slot)

    def pkt_parser(self, pkt):
        if pkt.haslayer('TLS'):
            ip = pkt.getlayer(IP)
            tcp = pkt.getlayer(TCP)
            tls = pkt.getlayer(TLS)

            self.markov(ip.src+":"+str(tcp.sport), ip.dst+":"+str(tcp.dport), tls.len)

    @staticmethod
    def printchain(a):
        #print('-->', a)
        s = sum(a)
        for x in range(BinsGenerator.MAXLEN):
            a[x] = int((a[x]*100)/s)

        print(a)

        

##############################################################################################################
# MAIN

def main():
    parser = argparse.ArgumentParser(description='DoH bins generator')
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

    generator = BinsGenerator(args.pcap, f)
    for host in generator.chains:
        BinsGenerator.printchain(generator.chains[host])

if __name__ == '__main__':
    main()
