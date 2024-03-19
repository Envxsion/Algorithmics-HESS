import networkx as nx
import matplotlib.pyplot as plt
import random

class Node:
    def __init__(self, id):
        self.id = id
        self.visited = False
        self.neighbours = []

    def __str__(self):
        return str(self.id)
        
    def add_neighbour(self, neighbour):
      if neighbour not in self.neighbours:
        self.neighbours.append(neighbour)
        self.sort_neighbours()
        
        
    def sort_neighbours(self):
      sorted = []
      while len(self.neighbours) > 0:
        smallest = self.neighbours[0]
        for node in self.neighbours:
          if node.id < smallest.id:
            smallest = node
        sorted.append(smallest)
        self.neighbours.remove(smallest)
      self.neighbours = sorted

class Edge:
    def __init__(self, source, target, weight):
        self.source = source
        self.target = target
        self.weight = weight
        self.id1 = source.id
        self.id2 = target.id
        self.color = 'black'

class RandomGraph:

  def __init__(self,V_N,E_N):
    self.edges = []
    self.nodes = [Node(i) for i in range(1,V_N+1)]
    
    for node in self.nodes:
      neighbours = random.choices(self.nodes, k=3)
      for neighbour in neighbours:
          if node != neighbour and self.get_weight(node.id, neighbour.id) == -1:
              rw = random.randint(1, 7)
              edge = Edge(node, neighbour, rw)
              self.edges.append(edge)
              node.add_neighbour(neighbour)
              neighbour.add_neighbour(node)


  def get_weight(self, node1, node2):
    for edge in self.edges:
      if edge.id1 == node1 and edge.id2 == node2:
        return edge.weight
      if edge.id2 == node1 and edge.id1 == node2:
        return edge.weight
    return -1

  def set_colour(self, id1, id2, colour):
    for edge in self.edges:
      if edge.id1 == id1 and edge.id2 == id2:
        edge.color = colour
      if edge.id1 == id2 and edge.id2 == id1:
        edge.color = colour  

  def display_graph(self):
    self.G = nx.Graph()
    for node in self.nodes:
        self.G.add_node(node)
    
    edge_colors = []
    for edge in self.edges:
        self.G.add_edge(edge.source, edge.target, weight=edge.weight)
        edge_colors.append(edge.color)

    pos = nx.spring_layout(self.G, seed=42)
    nx.draw(self.G, pos, with_labels=True, node_size=500, node_color='lightblue',
            font_size=12, font_color='black', font_weight='bold', edge_color=edge_colors, width=2.0)

    # Draw edge labels
    edge_labels = {(edge.source, edge.target): edge.weight for edge in self.edges}
    nx.draw_networkx_edge_labels(self.G, pos, edge_labels=edge_labels)

    plt.title("Prims")
    plt.axis('off')
    plt.show()


  def list_nodes_and_edges(self):
    for node in self.nodes:
      print(f"Node: {node.id}")
    for edge in self.edges:
      print(f"Edge from {edge.id1} to {edge.id2} with weight {edge.weight} and colour {edge.color}")


  def reset_visits(self):
    for node in self.nodes:
      node.visited = False

  def Prims(self):
    self.reset_visits()  
    visited_nodes = set()  
    mst_edges = [] 
    start_node = random.choice(self.nodes)
    start_node.visited = True
    visited_nodes.add(start_node)
    
    while len(visited_nodes) < len(self.nodes):
        min_edge = None
        min_weight = float('inf')
        for node in visited_nodes:
            for neighbor in node.neighbours:
                if neighbor not in visited_nodes:
                    weight = self.get_weight(node.id, neighbor.id)
                    if weight < min_weight:
                        min_weight = weight
                        min_edge = (node, neighbor, weight)

        if min_edge:
            mst_edges.append(min_edge)
            neighbor = min_edge[1]
            neighbor.visited = True
            visited_nodes.add(neighbor)

    
    for edge in mst_edges:
        edge[0].add_neighbour(edge[1])  
        self.set_colour(edge[0].id, edge[1].id, 'red')




      
RG = RandomGraph(10,3)
RG.Prims()
RG.list_nodes_and_edges()
RG.display_graph()
