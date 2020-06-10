
PCAP = 

vect.txt: $(PCAP)
	python3 doh.py $< > $@
