import networkx as nx
import matplotlib.pyplot as plt
import csv
import random
import math
from itertools import permutations

class Node:
    def __init__(self, name, pop, income, lat, long):
        self.name = name
        self.lat = lat
        self.long = long
        self.pop = pop
        self.income = income
        self.colour = 1
        self.neighbours = []

   
        
    def add_neighbour(self, neighbour):
      if neighbour not in self.neighbours:
        self.neighbours.append(neighbour)
        #self.sort_neighbours()
        
        
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
  def __init__(self, place1, place2, dist, time):
    self.place1 = place1
    self.place2 = place2
    self.dist = dist
    self.time = time
    self.colour = 2

class Graph:
  def __init__(self):
    self.edges = []
    self.nodes = []
    self.colour_dict = {0: "black", 1: "blue", 2: "red", 3: "green", 4: "yellow", 5:"lightblue"}

  def load_data(self):

    with open("nodes.csv", 'r', encoding='utf-8-sig') as csvfile:
      csv_reader = csv.reader(csvfile)
      for row in csv_reader:
        name = row[0]
        pop = row[1]
        income = row[2]
        lat = float(row[3])
        long = float(row[4])
        node = Node(name, pop, income, lat, long)
        self.nodes.append(node)
        

    with open("edges.csv", "r", encoding='utf-8-sig') as csvfile:
      csv_reader = csv.reader(csvfile)
      for row in csv_reader:
        place1 = row[0]
        place2 = row[1]
        dist = int(row[2])
        time = int(row[3])
        edge = Edge(place1, place2, dist, time)
        self.edges.append(edge)
        
        for node in self.nodes:
          if node.name == place1:
            node.add_neighbour(place2)
          if node.name == place2:
            node.add_neighbour(place1)
    
       

  def get_dist(self,place1,place2):
  
    for edge in self.edges:
      if edge.place1 == place1 and edge.place2 == place2:
        return edge.dist
      if edge.place1 == place2 and edge.place2 == place1:
        return edge.dist
    return -1

  def get_dist_FW(self,place1,place2):
  
    for edge in self.FW_edges:
      if edge.place1 == place1 and edge.place2 == place2:
        return edge.dist
      if edge.place1 == place2 and edge.place2 == place1:
        return edge.dist
    return -1

  def change_dist_FW(self,place1,place2,dist):
    for edge in self.FW_edges:
      if edge.place1 == place1 and edge.place2 == place2:
        edge.dist = dist
      if edge.place1 == place2 and edge.place2 == place1:
        edge.dist = dist

  def display(self):
    edge_labels = {}
    edge_colours = []
    G = nx.Graph()
    node_colour_list = []
    for node in self.nodes:
      G.add_node(node.name, pos=(node.long, node.lat))
      node_colour_list.append(self.colour_dict[node.colour])
    for edge in self.edges:
      G.add_edge(edge.place1, edge.place2)
      edge_labels[(edge.place1, edge.place2)] = edge.dist
      edge_colours.append(self.colour_dict[edge.colour])
    node_positions = nx.get_node_attributes(G, 'pos')


    plt.figure(figsize=(10, 8))
    nx.draw(G, node_positions, with_labels=True, node_size=50, node_color=node_colour_list, font_size=8, font_color='black', font_weight='bold', edge_color=edge_colours)
    nx.draw_networkx_edge_labels(G, node_positions, edge_labels=edge_labels)
    plt.title('')
    plt.savefig("map.png")
    plt.show()

  def deploy_team(self,target):
    ##### PART OF THE SOLUTION
    num_finalised = 1
    num_nodes = len(self.nodes)
    distance = {}
    finalised = {}
    neighbours = {}

    # Initialise all distances to 999999, set finalised to False and create a dictionary of neighbours

    for node in self.nodes:
      distance[node.name] = 999999
      finalised[node.name] = False
      neighbours[node.name] = node.neighbours
    current = "Bendigo"
    distance[current] = 0
    finalised[current] = True

    while num_finalised < num_nodes:
      # Check to see if a shorter path exists between the neighbours of the current node
      for neighbour in neighbours[current]:
        if finalised[neighbour] == False and distance[current] + self.get_dist(current,neighbour) < distance[neighbour]:
          distance[neighbour] = distance[current] + self.get_dist(current,neighbour)
      # Find the smallest node
      smallest_name = None
      smallest_dist = 999999
      for node in self.nodes:
        if finalised[node.name] == False and distance[node.name] < smallest_dist:
          smallest_name = node.name
          smallest_dist = distance[node.name]
      # Finalise that node
      finalised[smallest_name] = True
      num_finalised += 1
      current = smallest_name
    # Now backtrack to find the path
    path = [target]
    current = target
    while current != "Bendigo":
      for neighbour in neighbours[current]:
        if distance[current] - distance[neighbour] == self.get_dist(current,neighbour):
          current = neighbour
          path.append(current)
          break
    path.reverse()
    print(f"The shortest path from Bendigo to {target} is {path} with a distance of {distance[target]}")

  def haversine(self, lat1, lon1, lat2, lon2):
  #### PART OF SOLUTION
  
      # Radius of the Earth in kilometers
      R = 6371.0

      # Convert latitude and longitude from degrees to radians
      lat1_rad = math.radians(lat1)
      lon1_rad = math.radians(lon1)
      lat2_rad = math.radians(lat2)
      lon2_rad = math.radians(lon2)

      # Differences in coordinates
      dlat = lat2_rad - lat1_rad
      dlon = lon2_rad - lon1_rad

      # Haversine formula
      a = math.sin(dlat / 2)**2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon / 2)**2
      c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

      # Calculate the distance
      distance = R * c

      return distance
  

  def vaccinate(self,target,radius):
  #### PART OF SOLUTION
  # Look up lat and long of target
    for node in self.nodes:
      if node.name == target:
        tlat = node.lat
        tlong = node.long
    # First find all towns within the radius
    to_vaccinate = []
    for node in self.nodes:
      if self.haversine(tlat, tlong, node.lat, node.long) <= radius and node.name != target:
        to_vaccinate.append(node.name) 

    #print(f"Vaccinating towns: {to_vaccinate}")

    # Now create all possible paths between these nodes
    pathtuples = permutations(to_vaccinate)
    paths = [list(pathtuple) for pathtuple in pathtuples]

    # Add target to start and end of each path
    for path in paths:
      path.append(target)
      path.insert(0,target)

    #print("Paths to consider:")
    #for path in paths:
    #  print(path)
    # (removed following debugging)

    # Create a FW matrix to represent distances

    to_vaccinate.append(target)

    V_n = len(to_vaccinate) # Number of places to vaccinate

    matrix = [[9999 for i in range(V_n)] for j in range(V_n)]

    #print(f"Matrix: {to_vaccinate}")

    # Set leading diagonal to zero
    for i in range(V_n):
      matrix[i][i] = 0

    # Look up edges
    for i in range(V_n):
      for j in range(V_n):
        if i != j:
          dist = self.get_dist(to_vaccinate[i],to_vaccinate[j])
          if dist != -1:
            matrix[i][j] = dist
    
    # Print original matrix
    #print("Original matrix:") 
    #for row in matrix:
    #  print(row)

    # Run FW

    for k in range(V_n):
      for i in range(V_n):
        for j in range(V_n):
          if i != j and i != k and j != k and matrix[i][k] + matrix[k][j] < matrix[i][j]:
            matrix[i][j] = matrix[i][k] + matrix[k][j]

    # Print new matrix
    #print("New matrix:")
    #for row in matrix:
    #  print(row)

    # Create dictionary to look up matrix index
    matidx = {}
    for i in range(V_n):
      matidx[to_vaccinate[i]] = i
    


    best_dist = 99999999
    best_path = None
    for path in paths:
      dist = 0
      for i in range(len(path)-1):
        dist += matrix[matidx[path[i]]][matidx[path[i+1]]]
      #print(f"Path {path} has a distance of {dist}")
      if dist < best_dist:
        best_dist = dist
        best_path = path
    # Print the best path
    print(f"Best path is to vaccinate all towns within {radius} of {target} is {best_path} with distance of {best_dist}")
    print(f"Considered {len(paths)} paths in total.")
    





 
    

 
    




original = Graph()
original.load_data()
#original.display()
#original.searchteam("Orbost", 100)
original.deploy_team("Shepparton")
original.vaccinate("Shepparton", 100)
