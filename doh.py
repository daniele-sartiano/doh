#!/usr/bin/env python3
# 
# Prerequisite
# sudo apt-get install -y python3-pip
# pip3 install --pre scapy[basic]
#
#

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
    if (pkt.haslayer('TLS')):
        ip = pkt.getlayer(IP)
        tcp = pkt.getlayer(TCP)
        tls = pkt.getlayer(TLS)
        # print(ip.src+" -> "+ip.dst+" / "+str(tls.len))

        dns_ipaddr = ('1.1.1.1', '1.0.0.1')
        if ip.src in dns_ipaddr or ip.dst in dns_ipaddr:
            #print('calc', ip.src, ip.dst, scale(tls.len), tls.len)
            markov(ip.src+":"+str(tcp.sport), ip.dst+":"+str(tcp.dport), tls.len)


def printchain(a):
    #print('-->', a)
    s = sum(a)
    for x in range(maxlen):
        a[x] = int((a[x]*100)/s)

    print(a)
    
##############################################################################################################
# MAIN

#pcap_file="doh_single_short.pcap"
#pcap_file="doh_daniele.pcap"
#pcap_file="doh_luca.pcap"

pcap_file = sys.argv[1]

pkts = sniff(offline=pcap_file, prn=pkt_parser, bfilter='tcp and port 443') # Read pkts from pcap_file 

for host in chains:
    #print("From "+host)
    printchain(chains[host])

#print(chains)