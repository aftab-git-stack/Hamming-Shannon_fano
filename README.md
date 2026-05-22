# Huffman-Shannon_fano
# Aim:
Consider a discrete memoryless source with symbols and statistics {0.125, 0.0625, 0.25, 0.0625, 0.125, 0.125, 0.25} for its output. 
Apply the Huffman and Shannon-Fano to this source. 
Show that by drawing the tree diagram, and 
Calculate the average code word length, entropy, variance, redundancy, and efficiency.
# Tools Required:
# Program:
```
# ============================================
# EXPERIMENT 2
# Huffman Coding and Shannon-Fano Coding
# ============================================
# Google Colab Compatible Code
# ============================================

import math
from collections import Counter
import heapq
import pandas as pd

# ------------------------------------------------
# SOURCE SYMBOLS AND PROBABILITIES
# ------------------------------------------------

symbols = ['A', 'B', 'C', 'D', 'E', 'F', 'G']
probabilities = [0.125, 0.0625, 0.25, 0.0625, 0.125, 0.125, 0.25]

# ------------------------------------------------
# ENTROPY CALCULATION
# ------------------------------------------------

def calculate_entropy(probs):
    return -sum(p * math.log2(p) for p in probs)

entropy = calculate_entropy(probabilities)

print("Entropy H(X) =", round(entropy, 4), "bits\n")

# =========================================================
# HUFFMAN CODING
# =========================================================

class Node:
    def __init__(self, prob, symbol, left=None, right=None):
        self.prob = prob
        self.symbol = symbol
        self.left = left
        self.right = right
        self.code = ''

    def __lt__(self, nxt):
        return self.prob < nxt.prob

# Build Huffman Tree
def build_huffman(symbols, probs):

    heap = []

    for s, p in zip(symbols, probs):
        heapq.heappush(heap, Node(p, s))

    while len(heap) > 1:
        left = heapq.heappop(heap)
        right = heapq.heappop(heap)

        left.code = '0'
        right.code = '1'

        new_node = Node(left.prob + right.prob,
                        left.symbol + right.symbol,
                        left,
                        right)

        heapq.heappush(heap, new_node)

    return heap[0]

# Generate Huffman Codes
def generate_huffman_codes(node, val='', codes={}):
    new_val = val + str(node.code)

    if node.left:
        generate_huffman_codes(node.left, new_val, codes)

    if node.right:
        generate_huffman_codes(node.right, new_val, codes)

    if not node.left and not node.right:
        codes[node.symbol] = new_val

    return codes

# Print Tree
def print_tree(node, level=0):
    if node:
        print(' ' * level * 4 + f'[{node.symbol}:{round(node.prob,4)}]')
        print_tree(node.left, level + 1)
        print_tree(node.right, level + 1)

huffman_root = build_huffman(symbols, probabilities)

print("========== HUFFMAN TREE ==========\n")
print_tree(huffman_root)

huffman_codes = generate_huffman_codes(huffman_root)

print("\n========== HUFFMAN CODES ==========\n")
for k, v in huffman_codes.items():
    print(k, ":", v)

# Average Code Length
def average_length(codes, probs, symbols):
    return sum(len(codes[s]) * p for s, p in zip(symbols, probs))

L_huffman = average_length(huffman_codes, probabilities, symbols)

# Variance
def variance(codes, probs, symbols, avg_len):
    return sum(p * ((len(codes[s]) - avg_len) ** 2)
               for s, p in zip(symbols, probs))

V_huffman = variance(huffman_codes, probabilities, symbols, L_huffman)

# Efficiency and Redundancy
eff_huffman = entropy / L_huffman
red_huffman = 1 - eff_huffman

# =========================================================
# SHANNON-FANO CODING
# =========================================================

sf_codes = {}

# Sort symbols based on probabilities
sorted_data = sorted(zip(symbols, probabilities),
                     key=lambda x: x[1],
                     reverse=True)

def shannon_fano(data, code=''):

    if len(data) == 1:
        symbol = data[0][0]
        sf_codes[symbol] = code
        return

    total = sum(p for _, p in data)

    split = 0
    cumulative = 0

    for i in range(len(data)):
        cumulative += data[i][1]

        if cumulative >= total / 2:
            split = i
            break

    left = data[:split+1]
    right = data[split+1:]

    shannon_fano(left, code + '0')

    if right:
        shannon_fano(right, code + '1')

print("\n========== SHANNON-FANO TREE ==========\n")

def print_sf_tree(data, level=0):

    total = sum(p for _, p in data)

    print(' ' * level * 4 + f'Total={round(total,4)} -> {data}')

    if len(data) == 1:
        return

    cumulative = 0
    split = 0

    for i in range(len(data)):
        cumulative += data[i][1]

        if cumulative >= total / 2:
            split = i
            break

    left = data[:split+1]
    right = data[split+1:]

    print_sf_tree(left, level + 1)

    if right:
        print_sf_tree(right, level + 1)

print_sf_tree(sorted_data)

shannon_fano(sorted_data)

print("\n========== SHANNON-FANO CODES ==========\n")
for k, v in sf_codes.items():
    print(k, ":", v)

L_sf = average_length(sf_codes, probabilities, symbols)

V_sf = variance(sf_codes, probabilities, symbols, L_sf)

eff_sf = entropy / L_sf
red_sf = 1 - eff_sf

# =========================================================
# RESULT TABLES
# =========================================================

print("\n========== HUFFMAN RESULT ==========\n")

huffman_table = pd.DataFrame({
    "Symbol": symbols,
    "Probability": probabilities,
    "Code": [huffman_codes[s] for s in symbols],
    "Length": [len(huffman_codes[s]) for s in symbols]
})

print(huffman_table)

print("\nAverage Code Length =", round(L_huffman, 4))
print("Entropy =", round(entropy, 4))
print("Variance =", round(V_huffman, 4))
print("Efficiency =", round(eff_huffman * 100, 2), "%")
print("Redundancy =", round(red_huffman * 100, 2), "%")

# ---------------------------------------------------------

print("\n========== SHANNON-FANO RESULT ==========\n")

sf_table = pd.DataFrame({
    "Symbol": symbols,
    "Probability": probabilities,
    "Code": [sf_codes[s] for s in symbols],
    "Length": [len(sf_codes[s]) for s in symbols]
})

print(sf_table)

print("\nAverage Code Length =", round(L_sf, 4))
print("Entropy =", round(entropy, 4))
print("Variance =", round(V_sf, 4))
print("Efficiency =", round(eff_sf * 100, 2), "%")
print("Redundancy =", round(red_sf * 100, 2), "%")

comparison = pd.DataFrame({
    "Method": ["Huffman", "Shannon-Fano"],
    "Average Length": [round(L_huffman,4), round(L_sf,4)],
    "Entropy": [round(entropy,4), round(entropy,4)],
    "Variance": [round(V_huffman,4), round(V_sf,4)],
    "Efficiency (%)": [round(eff_huffman*100,2), round(eff_sf*100,2)],
    "Redundancy (%)": [round(red_huffman*100,2), round(red_sf*100,2)]
})

print("\n========== COMPARISON ==========\n")
print(comparison)

```
# Calculation:
```
Entropy H(X) = 2.625 bits

[CFBDGEA:1.0]
    [CFBD:0.5]
        [C:0.25]
        [FBD:0.25]
            [F:0.125]
            [BD:0.125]
                [B:0.0625]
                [D:0.0625]
    [GEA:0.5]
        [G:0.25]
        [EA:0.25]
            [E:0.125]
            [A:0.125]

========== HUFFMAN CODES ==========

C : 00
F : 010
B : 0110
D : 0111
G : 10
E : 110
A : 111

========== SHANNON-FANO TREE ==========

Total=1.0 -> [('C', 0.25), ('G', 0.25), ('A', 0.125), ('E', 0.125), ('F', 0.125), ('B', 0.0625), ('D', 0.0625)]
    Total=0.5 -> [('C', 0.25), ('G', 0.25)]
        Total=0.25 -> [('C', 0.25)]
        Total=0.25 -> [('G', 0.25)]
    Total=0.5 -> [('A', 0.125), ('E', 0.125), ('F', 0.125), ('B', 0.0625), ('D', 0.0625)]
        Total=0.25 -> [('A', 0.125), ('E', 0.125)]
            Total=0.125 -> [('A', 0.125)]
            Total=0.125 -> [('E', 0.125)]
        Total=0.25 -> [('F', 0.125), ('B', 0.0625), ('D', 0.0625)]
            Total=0.125 -> [('F', 0.125)]
            Total=0.125 -> [('B', 0.0625), ('D', 0.0625)]
                Total=0.0625 -> [('B', 0.0625)]
                Total=0.0625 -> [('D', 0.0625)]

========== SHANNON-FANO CODES ==========

C : 00
G : 01
A : 100
E : 101
F : 110
B : 1110
D : 1111

========== HUFFMAN RESULT ==========

  Symbol  Probability  Code  Length
0      A       0.1250   111       3
1      B       0.0625  0110       4
2      C       0.2500    00       2
3      D       0.0625  0111       4
4      E       0.1250   110       3
5      F       0.1250   010       3
6      G       0.2500    10       2

Average Code Length = 2.625
Entropy = 2.625
Variance = 0.4844
Efficiency = 100.0 %
Redundancy = 0.0 %

========== SHANNON-FANO RESULT ==========

  Symbol  Probability  Code  Length
0      A       0.1250   100       3
1      B       0.0625  1110       4
2      C       0.2500    00       2
3      D       0.0625  1111       4
4      E       0.1250   101       3
5      F       0.1250   110       3
6      G       0.2500    01       2

Average Code Length = 2.625
Entropy = 2.625
Variance = 0.4844
Efficiency = 100.0 %
Redundancy = 0.0 %

```
# Output
```
        Method  Average Length  Entropy  Variance  Efficiency (%)  \
0       Huffman           2.625    2.625    0.4844           100.0   
1  Shannon-Fano           2.625    2.625    0.4844           100.0   

   Redundancy (%)  
0             0.0  
1             0.0  

``` 
# Results:
```
Hence the program is executed and the output is verified.
```
