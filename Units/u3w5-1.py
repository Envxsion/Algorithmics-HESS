import networkx as nx
import matplotlib.pyplot as plt
import random
from collections import deque

class Node:
    def __init__(self, id):
        self.id = id
        self.visited = False
        self.neighbors = []

    def __str__(self):
        return str(self.id)
        
    def add_neighbor(self, neighbor):
        if neighbor not in self.neighbors:
            self.neighbors.append(neighbor)
            self.sort_neighbors()
        
    def sort_neighbors(self):
        self.neighbors.sort(key=lambda x: x.id)

class Edge:
    def __init__(self, source, target, weight=1):
        self.source = source
        self.target = target
        self.weight = weight

G = nx.Graph()

nodes = [Node(i) for i in range(1, 11)]
for node in nodes:
    G.add_node(node)

for node in nodes:
    neighbors = random.sample(nodes, 3)
    for neighbor in neighbors:
        if node != neighbor:
            edge = Edge(node, neighbor)
            node.add_neighbor(neighbor)
            neighbor.add_neighbor(node)
            G.add_edge(edge.source, edge.target)


pos = nx.spring_layout(G, seed=42)
nx.draw(G, pos, with_labels=True, node_size=500, node_color='lightblue', font_size=12, font_color='black', font_weight='bold')

plt.title("Rand Graph - 10 nodes (3 Connection/Node)")
plt.axis('off')
plt.show()
stack = []
stack.append(nodes[0])
path = []

while stack:
    current = stack.pop()
    if not current.visited:
        path.append(current.id)
        current.visited = True
        for neighbor in current.neighbors:
            if not neighbor.visited:
                stack.append(neighbor)

print(f"Depth-first search: {path}")

for node in nodes:
    node.visited = False

path = []
queue = deque()
queue.append(nodes[0])

while queue:
    current = queue.popleft()
    if not current.visited:
        path.append(current.id)
        current.visited = True
        for neighbor in current.neighbors:
            if not neighbor.visited:
                queue.append(neighbor)

print(f"Breadth-first search: {path}")
