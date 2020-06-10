
PCAP = 

DNS =

vect.txt: $(PCAP)
	python3 doh.py $< $(DNS) > $@
