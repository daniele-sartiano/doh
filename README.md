# doh
DoH Classifier

### Deps
```
pip install -r requirements.txt
```

### Run

Set PCAP and DNS vars:
- PCAP with your pcap file
- DNS a list of DNS server separated by comma 

```
make PCAP=your_pcap_file.pcap DNS=1.1.1.1,1.0.0.1,8.8.8.8 vect.txt
```