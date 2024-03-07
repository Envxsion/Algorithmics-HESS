import networkx as nx
import matplotlib.pyplot as plt
import random

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
      # This method is needed to return answers in numerical order
      sorted = []
      while len(self.neighbors) > 0:
        smallest = self.neighbors[0]
        for node in self.neighbors:
          if node.id < smallest.id:
            smallest = node
        sorted.append(smallest)
        self.neighbors.remove(smallest)
      self.neighbors = sorted

class Edge:
    def __init__(self, source, target, weight = 1):
        self.source = source
        self.target = target
        self.weight = weight

# Create an empty graph
G = nx.Graph()

# Create 10 nodes and add them to the graph
nodes = [Node(i) for i in range(1, 11)]
for node in nodes:
    G.add_node(node)

# Connect each node to 3 random nodes
for node in nodes:
    neighbors = random.sample(nodes, 3)
    for neighbor in neighbors:
        if node != neighbor:
            edge = Edge(node, neighbor)
            node.add_neighbor(neighbor)
            neighbor.add_neighbor(node)
            G.add_edge(edge.source, edge.target)

# Draw the graph
pos = nx.spring_layout(G, seed=42)
nx.draw(G, pos, with_labels=True, node_size=500, node_color='lightblue', font_size=12, font_color='black', font_weight='bold')

# Show the graph
plt.title("Random Graph with 10 Nodes (3 Edges per Node)")
plt.axis('off')
plt.show()


# Traverse using DFS
# Read this example carefully.

stack = []
stack.append(nodes[0])
path = []

while len(stack) > 0:
  current = stack.pop()
  if current.visited == False:
    path.append(current.id)
    current.visited = True
    for neighbor in current.neighbors:
      #if neighbor.visited == False:
      stack.append(neighbor)

print(f"Depth first search: {path}")

# Set all nodes back to unvisited

for node in nodes:
  node.visited = False

# Now add your code to carry out a BFS traversal.