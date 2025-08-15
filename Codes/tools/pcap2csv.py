import argparse
import re

from scapy.all import rdpcap
import os


def match_label(pcap_file, label_list):
    filename = os.path.basename(pcap_file)
    filename = os.path.splitext(filename)[0].lower()

    for label in label_list:
        if label.lower() in filename:
            return label

    print(f"No matching label found for file: {pcap_file}")
    return "unknown"


def parse_pcap(pcap_file, output_dir=None, label_list=None):
    if label_list is None:
        label_list = []
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, os.path.basename(os.path.splitext(pcap_file)[0]) + '.csv')
    else:
        output_file = os.path.splitext(pcap_file)[0] + '.csv'

    label = match_label(pcap_file, label_list)

    packets = rdpcap(pcap_file)

    with open(output_file, 'w') as f:
        f.write('date,pkt_size,src,dst,sport,dport,proto,Label\n')
        for pkt in packets:
            try:
                if not pkt.haslayer("IP"):
                    continue

                timestamp = getattr(pkt, "time", "N/A")
                pkt_size = getattr(pkt, "len", "N/A")

                src = pkt["IP"].src
                dst = pkt["IP"].dst
                proto = pkt["IP"].proto

                if pkt.haslayer("TCP") or pkt.haslayer("UDP"):
                    sport = getattr(pkt, "sport", "N/A")
                    dport = getattr(pkt, "dport", "N/A")
                else:
                    sport = "N/A"
                    dport = "N/A"

                f.write(f"{timestamp},{pkt_size},{src},{dst},{sport},{dport},{proto},{label}\n")
            except AttributeError as e:
                print(f"Error processing packet: {repr(e)} | Packet summary: {pkt.summary()}")
            except Exception as e:
                print(f"Unexpected error: {repr(e)} | Packet summary: {pkt.summary()}")

    print(f"save {output_file} success.")



def process_directory(input_dir, output_dir, label_list):
    pcap_files = [os.path.join(input_dir, file) for file in os.listdir(input_dir) if file.endswith('.pcap')]

    if not pcap_files:
        print("No pcap file found!")
        return

    print(f"find {len(pcap_files)} pcap files, begin parse...")
    for pcap_file in pcap_files:
        parse_pcap(pcap_file, output_dir, label_list)



def main():
    parser = argparse.ArgumentParser(description="Parse PCAP files and output packet details.")
    parser.add_argument("--pcap-file", help="Path to a single PCAP file", default=None)
    parser.add_argument("--input-dir", help="Path to a directory containing multiple PCAP files", default=None)
    parser.add_argument("--output-dir", help="Directory to save the output files", default=None)

    args = parser.parse_args()

    # labels = ["ftps", "bittorrent", "spotify", "vimeo", "hangout", "sftp",
    #           "youtube", "icq", "netflix", "facebook", "skype", "voipbuster",
    #           "email", "aim"] #iscx-vpn-2016

    #labels = ["browsing", "file", "voip", "audio", "chat", "mail", "p2p", "video"] #iscx-tor-2016

    labels = ["Geodo", "Cridex", "Tinba", "Shifu", "Gmail", "SMB", "Weibo", "WorldOfWarcraft",
              "Zeus", "FTP", "MySQL", "BitTorrent", "Skype", "Miuref","Htbot", "Outlook", "Facetime",
              "Virut", "Nsis-ay", "Neris"]# USTC-TFC-2106




    if args.pcap_file:
        parse_pcap(args.pcap_file, args.output_dir, labels)
    elif args.input_dir:
        process_directory(args.input_dir, args.output_dir, labels)
    else:
        print("please input --pcap-file or --input-dir to specify a pcap file or folder.")



if __name__ == "__main__":
    main()
