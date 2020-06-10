
PCAP = 

DNS =

vect.txt: $(PCAP)
	python3 doh.py $< $(DNS) > $@


classify: vect.txt
	python3 classifier.py $(PCAP) $<
