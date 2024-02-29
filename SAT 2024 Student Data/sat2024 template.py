import networkx as nx
import matplotlib.pyplot as plt
import csv
import random
import math
from itertools import permutations

class Node:
    def __init__(self, name, pop, income, lat, long):
        # Name, latitude, longitude, population, weekly household income, default colour (1-5), empty list of neighbours
        self.name = name
        self.lat = lat
        self.long = long
        self.pop = pop
        self.income = income
        self.colour = 1
        self.neighbours = []

   
        
    def add_neighbour(self, neighbour):
      # Adds a neighbour (node object) after checking to see if it was there already
      if neighbour not in self.neighbours:
        self.neighbours.append(neighbour)
        
    
        
    

class Edge:
  def __init__(self, place1, place2, dist, time):
    # Two places (order unimportant), distance in km, time in mins, default colour (1-5)
    self.place1 = place1
    self.place2 = place2
    self.dist = dist
    self.time = time
    self.colour = 2

class Graph:
  def __init__(self):
    # List of edge objects and node objects
    self.edges = []
    self.nodes = []
    self.colour_dict = {0: "black", 1: "blue", 2: "red", 3: "green", 4: "yellow", 5:"lightblue"}

  def load_data(self):
    # Reads the CSV files you are provided with and creates node/edges accordingly.
    # You should not need to change this function.

    # Read the nodes, create node objects and add them to the node list.
    with open("SAT 2024 Student Data/nodes.csv", 'r', encoding='utf-8-sig') as csvfile:
      csv_reader = csv.reader(csvfile)
      for row in csv_reader:
        name = row[0]
        pop = row[1]
        income = row[2]
        lat = float(row[3])
        long = float(row[4])
        node = Node(name, pop, income, lat, long)
        self.nodes.append(node)
        
    # Read the edges, create edge objects and add them to the edge list.
    with open("SAT 2024 Student Data/nodes.csv", "r", encoding='utf-8-sig') as csvfile:
      csv_reader = csv.reader(csvfile)
      for row in csv_reader:
        place1 = row[0]
        place2 = row[1]
        dist = int(row[2])
        time = int(float(row[3]))
        edge = Edge(place1, place2, dist, time)
        self.edges.append(edge)
        
    for node in self.nodes:
      if node.name == place1:
        node.add_neighbour(place2)
      if node.name == place2:
        node.add_neighbour(place1)
    
       

  def get_dist(self,place1,place2):
    # Returns the distance between two place names (strings) if an edge exists,
    # otherwise returns -1.
  
    for edge in self.edges:
      if edge.place1 == place1 and edge.place2 == place2:
        return edge.dist
      if edge.place1 == place2 and edge.place2 == place1:
        return edge.dist
    return -1

  
  def display(self, filename = "map.png"):
    # Displays the object on screen and also saves it to a PNG named in the argument.
    
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
    plt.savefig(filename)
    plt.show()

  

  def haversine(self, lat1, lon1, lat2, lon2):
  # Returns the distance in km between two places with given latitudes and longitudes.
  
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
  
  # This is where you will write your algorithms. You don't have to use
  # these names/parameters but they will probably steer you in the right
  # direction.

  def searchteam(self,target,radius):
    pass

  def vaccinate(self,target,radius):
    pass

  def findpath(self,target):
    pass



# These commands run the code.

# Create a new graph object called 'original'
original = Graph()

# Load data into that object.
original.load_data()

# Display the object, also saving to map.png
original.display("map.png")

# You will add your own functions under the Graph object and call them in this way:
#original.findpath("Alexandra")