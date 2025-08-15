import argparse
import re

from scapy.all import rdpcap
import os


def parse_label(pcap_file):
    filename = os.path.basename(pcap_file)
    filename = os.path.splitext(filename)[0]
    label = re.sub(r'(\d+[a-zA-Z]*)', '', filename)
    label = re.sub(r'(_[A-Z])', '', label)
    return label


def parse_pcap(pcap_file, output_dir=None):
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, os.path.basename(os.path.splitext(pcap_file)[0]) + '.csv')
    else:
        output_file = os.path.splitext(pcap_file)[0] + '.csv'

    label = parse_label(pcap_file)

    packets = rdpcap(pcap_file)

    start_time = packets[0].time
    with open(output_file, 'w') as f:
        f.write('date,pkt_size,src,dst,sport,dport,proto,Label\n')
        for pkt in packets:
            try:
                timestamp = pkt.time
                pkt_size = pkt.len

                # Get network layer data
                if pkt.haslayer('IP'):
                    src= pkt['IP'].src
                    dst= pkt['IP'].dst
                    proto = pkt['IP'].proto

                    # Get transport layer data
                    if pkt.haslayer('TCP') or pkt.haslayer('UDP'):
                        sport = pkt.sport
                        dport = pkt.dport
                    else:
                        sport = "N/A"
                        dport = "N/A"

                    f.write(f"{timestamp},{pkt_size},{src},{dst},{sport},{dport},{proto},{label}\n")
            except AttributeError as e:
                print(repr(e))

    print(f"save {output_file} success.")


def process_directory(input_dir, output_dir):
    pcap_files = [os.path.join(input_dir, file) for file in os.listdir(input_dir) if file.endswith('.pcap')]

    if not pcap_files:
        print("No pcap file found!")
        return

    print(f"find {len(pcap_files)} pcap files, begin parse...")
    for pcap_file in pcap_files:
        parse_pcap(pcap_file, output_dir)


def main():
    parser = argparse.ArgumentParser(description="Parse PCAP files and output packet details.")
    parser.add_argument("--pcap-file", help="Path to a single PCAP file", default=None)
    parser.add_argument("--input-dir", help="Path to a directory containing multiple PCAP files", default=None)
    parser.add_argument("--output-dir", help="Directory to save the output files", default=None)

    args = parser.parse_args()

    if args.pcap_file:
        parse_pcap(args.pcap_file, args.output_dir)
    elif args.input_dir:
        process_directory(args.input_dir, args.output_dir)
    else:
        print("please input --pcap-file or --input-dir to specify a pcap file or folder.")


if __name__ == "__main__":
    main()
