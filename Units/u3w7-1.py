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

    def __init__(self, V_N, E_N):
        self.edges = []
        self.nodes = [Node(i) for i in range(1,V_N+1)]
        
        for node in self.nodes:
        
          neighbours = random.sample(self.nodes,3)
          for neighbour in neighbours:
            if node != neighbour and self.get_weight(node.id, neighbour.id) == -1:
              rw = random.randint(1,7)
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
        for edge in self.edges:
          self.G.add_edge(edge.source, edge.target, weight = edge.weight)
        pos = nx.spring_layout(self.G)
        nx.draw(self.G, pos, with_labels=True, node_size=500, node_color='lightblue', font_size=12, font_color='black', font_weight='bold')
          
        edge_labels = {}
        edge_colors = []
        for edge in self.edges:
          edge_colors.append(edge.color)  
          edge_labels[(edge.source, edge.target)]= edge.weight
        
        
        
        nx.draw(self.G, pos, with_labels=True, node_size=500, node_color='lightblue', font_size=12, font_color='black', font_weight='bold', edge_color=edge_colors, width=2.0)
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

    def Dijkstra(self, source, target):
      distances = {node: float('inf') for node in self.nodes}
      predecessors = {node: None for node in self.nodes}
      distances[source] = 0

      while True:
          unvisited_nodes = [node for node in self.nodes if not node.visited]
          if not unvisited_nodes:  #addcheck
              print("No path found from source to target.")
              return []

          min_node = min(unvisited_nodes, key=lambda node: distances[node])
          min_distance = distances[min_node]

          min_node.visited = True

          for neighbor in min_node.neighbours:
              if not neighbor.visited:
                  edge_weight = self.get_weight(min_node.id, neighbor.id)
                  if min_distance + edge_weight < distances[neighbor]:
                      distances[neighbor] = min_distance + edge_weight
                      predecessors[neighbor] = min_node

          if min_node == target:
              break

      shortest_path = []
      current_node = target
      while current_node is not None:
          shortest_path.insert(0, current_node)
          current_node = predecessors[current_node]

      print("Shortest Path:")
      for node in shortest_path:
          print(node.id, end=" -> ")
      print("\nTotal Distance:", distances[target])

      for i in range(len(shortest_path) - 1):
          self.set_colour(shortest_path[i].id, shortest_path[i + 1].id, 'red')

      return shortest_path


RG = RandomGraph(10, 3)
shortest_path = RG.Dijkstra(RG.nodes[0], RG.nodes[-1])
RG.list_nodes_and_edges()
RG.display_graph()
