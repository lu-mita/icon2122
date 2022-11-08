from queue import Queue
from collections import namedtuple
import pandas as pd
import os

class Node:
    def __init__(self):
        self.level = None
        self.profit = None
        self.bound = None
        self.weight = None

    def __str__(self):
        return "Level: %s Profit: %s Bound: %s Weight: %s" % (self.level, self.profit, self.bound, self.weight)


def bound(node, n, W, items):
    if node.weight >= W:
        return 0

    profit_bound = int(node.profit)
    j = node.level + 1
    totweight = int(node.weight)

    while (j < n) and (totweight + items[j].weight) <= W:
        totweight += items[j].weight
        profit_bound += items[j].value
        j += 1

    if(j < n):
        profit_bound += (W - totweight) * items[j].value / float(items[j].weight)

    return profit_bound

Q = Queue()

def KnapSackBranchNBound(weight, items, total_items):
    items = sorted(items, key=lambda x: x.value/float(x.weight), reverse=True)

    u = Node()
    v = Node()

    u.level = -1
    u.profit = 0
    u.weight = 0

    Q.put(u)
    maxProfit = 0

    while not Q.empty():
        u = Q.get()
        v = Node()  # Added line
        if u.level == -1:
            v.level = 0

        if u.level == total_items - 1:
            continue

        v.level = u.level + 1
        v.weight = u.weight + items[v.level].weight
        v.profit = u.profit + items[v.level].value
        if v.weight <= weight and v.profit > maxProfit:
            maxProfit = v.profit

        v.bound = bound(v, total_items, weight, items)
        if v.bound > maxProfit:
            Q.put(v)

        v = Node()  # Added line
        v.level = u.level + 1  # Added line
        v.weight = u.weight
        v.profit = u.profit
        v.bound = bound(v, total_items, weight, items)
        if v.bound > maxProfit:
            # print(items[v.level])
            Q.put(v)

    return maxProfit

def execute(filename):
    capacity = int(input("Insert truck capacity:"))
    item_count = 100
    Item = namedtuple("Item", ['index', 'value', 'weight'])
    file = pd.read_csv(filename)
    weights = list(file["PESO VOLUMETRICO"])
    values = list(file["UTILE"])

    items = []

    for i in range(1, item_count + 1):
        items.append(Item(i - 1, values[i - 1], weights[i - 1]))
    kbb = KnapSackBranchNBound(capacity, items, item_count)
    print(f"Utile massimo:{kbb}€\n")

execute(os.path.join('data', 'branch_and_bound', 'Products.csv'))







