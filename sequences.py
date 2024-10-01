import csv

inp_seq = ""

output_seq = []
bases = ["A", "C", "G", "T"]
for i in range(0, len(inp_seq)):
    for base in bases:
        curr_str = inp_seq[0:i] + base + inp_seq[i+1:]
        output_seq.append(curr_str)

# Let us print the list in a clean format
for element in output_seq:
    print(element)
    print()

# Let us output our list into a csv file for a cleaner view
with open("sequences.csv", 'w') as f:   # Opening file in "w" or write mode
    fc = csv.writer(f, lineterminator='\n')
    fc.writerows(output_seq)
