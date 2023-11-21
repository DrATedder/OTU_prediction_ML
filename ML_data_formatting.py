import glob
import os
import sys
from Bio import Entrez

def get_taxid(accession_numbers):
    Entrez.email = "your@email.com"
    handle = Entrez.esummary(db="nucleotide", id=accession_numbers, retmode="xml")
    records = Entrez.read(handle)
    taxid_dict = {}
    for record in records:
        accession_number = record["Caption"]
        try:
            taxid = int(record["TaxId"])
            taxid_dict[accession_number] = taxid
        except KeyError:
            print(f"TaxID not found for {accession_number}")
    handle.close()
    return taxid_dict


def read_abundance_dict(abundance):
    ID_dict = {}
    with open(abundance, "r") as f_in:
        for line in f_in:
            accession, component = line.split("\t")[0].split(".")[0], line.split("\t")[1].strip()
            ID_dict[accession] = component
    return ID_dict

def total_reads(cent_report):
    total = []
    with open(cent_report, "r") as f_in:
        for line in f_in:
            if not line.startswith("name"):
                reads = line.split("\t")[4]
                total.append(int(reads))
    return sum(total)

def presence_absence_list(source_list):
    taxid_list = []
    with open(source_list, "r") as f_in:
        for line in f_in:
            taxID = line.split(",")[1]
            taxid_list.append(taxID)
    return taxid_list

def adjust_format(cent_report, source_list):
    taxRank_list = ['species', 'genus', 'subspecies', 'leaf']
    background = get_taxid(source_list)
    component_dict = {}
    for x in background:
        component_dict[background[x]] = source_list[x]
    total = total_reads(cent_report)
    line_list = []
    with open(cent_report, "r") as f_in:
        for line in f_in:
            if not line.split("\t")[0] == "name":
                line = line.replace(",","")
                tmp_list = []
                reads = int(line.split("\t")[4])
                abundance = float(reads/total)
                taxID = line.split("\t")[1].strip()
                line = line.split("\t")
                taxRank = line[2].strip()
                if taxRank in taxRank_list:
                    genus = line[0].split(" ")[0]
                else:
                    genus = "NA"
                if int(taxID) in background.values():
                    presence = 1
                    sim_abundance = component_dict[int(taxID)]
                else:
                    presence = 0
                    sim_abundance = 0
                for x in range(0, len(line)-1, 1):
                    tmp_list.append(line[x])
                line_list.append(f"{','.join(tmp_list)},{abundance},{genus},{presence},{sim_abundance}")
            else:
                line = line.replace("\t", ",").strip()
                line_list.append(f"{line},genus,presence,sim_abundance")
    return line_list

directories = sys.argv[1]
processed_inputs = sys.argv[2]

for run in glob.glob(directories + "*/"):
    print(run)
    cent_report = f"{run}{run.split('/')[-2]}_centrifugeReport.txt"
    abundance = f"{run}{run.split('/')[-2].lstrip('a')}_abundance.txt"
    with open(f'{processed_inputs}{run.split("/")[-2]}_data.txt', 'w') as f_out:
        for items in adjust_format(cent_report, read_abundance_dict(abundance)):
            f_out.write(f'{items}\n')
