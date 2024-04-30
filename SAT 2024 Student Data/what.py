import heapq
import random
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
from networkx import spring_layout

class Graph:
    def __init__(self):
        self.nodes = {}
        self.edges = {}
        self.degrees = {}  # Keep track of node degrees for the Chinese Postman Problem

    def add_node(self, name, population, income, lat, lon, age):
        self.nodes[name] = (population, income, lat, lon, age)
        self.degrees[name] = 0

    def add_edge(self, node1, node2, weight):
        if node1 not in self.edges:
            self.edges[node1] = {}
        self.edges[node1][node2] = weight
        if node2 not in self.edges:
            self.edges[node2] = {}
        self.edges[node2][node1] = weight
        self.degrees[node1] += 1
        self.degrees[node2] += 1

# Load nodes and edges data with custom column names
nodes_df = pd.read_csv('SAT 2024 Student Data/nodes.csv', header=None, names=['name', 'population', 'income', 'lat', 'lon', 'age', 'node'])
edges_df = pd.read_csv('SAT 2024 Student Data/edges.csv', header=None, names=['from', 'to', 'distance', 'time'])

# Create a graph
G = Graph()

# Add nodes to the graph
for _, row in nodes_df.iterrows():
    G.add_node(row['name'], row['population'], row['income'], row['lat'], row['lon'], row['age'])

# Add edges to the graph with weights
for _, row in edges_df.iterrows():
    G.add_edge(row['from'], row['to'], row['distance'])

# ---------- Task 1: Shortest Path from Bendigo (Dijkstra's Algorithm) ----------

def dijkstra(graph, source):
    distances = {node: float('inf') for node in graph.nodes}
    distances[source] = 0
    heap = [(0, source)]

    while heap:
        current_distance, current_node = heapq.heappop(heap)

        if current_distance > distances[current_node]:
            continue

        for neighbor, weight in graph.edges[current_node].items():
            distance = current_distance + weight
            if distance < distances[neighbor]:
                distances[neighbor] = distance
                heapq.heappush(heap, (distance, neighbor))

    return distances

def shortest_path_from_bendigo(target):
    bendigo_distances = dijkstra(G, 'Bendigo')
    path = [target]
    while path[-1] != 'Bendigo':
        path.append(next(neighbor for neighbor, weight in G.edges[path[-1]].items() if bendigo_distances[path[-1]] == bendigo_distances[neighbor] + weight))
    path.reverse()
    return path


# Find the shortest path from Bendigo to the target location
target_node = 'Melbourne'
shortest_path = shortest_path_from_bendigo(target_node)
print(f"AllTeams: Going to Target Location ({target_node}) from Bendigo")
print("AllTeams: Going to Target Location using path:", ' -> '.join(shortest_path))

# Calculate the distance and travel time to the target location
bendigo_distance = sum(G.edges[shortest_path[i]][shortest_path[i+1]] for i in range(len(shortest_path)-1))
bendigo_travel_time = sum(G.edges[shortest_path[i]][shortest_path[i+1]] for i in range(len(shortest_path)-1))
print(f"AllTeams: Distance to Target Location: {bendigo_distance:.2f} km, Travel Time: {bendigo_travel_time:.2f} minutes")

# ---------- Task 2: Medical Team Deployment (Simplified Traveling Salesman Problem) ----------

def random_greedy_tour(graph, start_node):
    tour = [start_node]
    remaining_nodes = list(graph.nodes)
    remaining_nodes.remove(start_node)
    
    while remaining_nodes:
        current_node = tour[-1]
        next_node = min(remaining_nodes, key=lambda node: graph.edges[current_node].get(node, float('inf')))
        tour.append(next_node)
        remaining_nodes.remove(next_node)
    
    return tour

def tour_distance(graph, tour):
    total_distance = 0
    for i in range(len(tour) - 1):
        current_node = tour[i]
        next_node = tour[i + 1]
        total_distance += graph.edges[current_node].get(next_node, 0)
    
    # Close the tour
    total_distance += graph.edges[tour[-1]].get(tour[0], 0)
    
    return total_distance

# Find a random tour for the medical team starting from Bendigo
medical_team_start = 'Bendigo'
medical_team_tour = random_greedy_tour(G, medical_team_start)
medical_team_distance = tour_distance(G, medical_team_tour)
print("Medical Team Tour:", medical_team_tour)
print("Medical Team Total Distance:", medical_team_distance)

# Print verbose statements for medical team deployment
print("MedicalTeam: At Target Location (Melbourne), recalculating towns within radius:")
for i in range(len(medical_team_tour) - 1):
    current_node = medical_team_tour[i]
    next_node = medical_team_tour[i + 1]
    distance = G.edges[current_node].get(next_node, 0)
    print(f"MedicalTeam: Vaccinating {next_node} (distance: {distance:.2f} km, travel time: {distance:.2f} minutes)")
    path = shortest_path_from_bendigo(next_node)
    print("MedicalTeam: Going to", next_node, "using path:", ' -> '.join(path))
print("MedicalTeam: Vaccinations Finished, returning to Bendigo")

# ---------- Task 3: Search Team Deployment (Graph Traversal) ----------

def breadth_first_search(graph, source, target):
    visited = {node: False for node in graph.nodes}
    queue = [source]
    visited[source] = True

    while queue:
        current_node = queue.pop(0)

        if current_node == target:
            return True

        for neighbor in graph.edges[current_node].keys():
            if not visited[neighbor]:
                queue.append(neighbor)
                visited[neighbor] = True

    return False

# Find the shortest path from Bendigo to the target location
search_team_start = 'Bendigo'
search_team_target = 'Melbourne'
search_team_path = []

current_node = search_team_start
while current_node != search_team_target:
    for neighbor in G.edges[current_node].keys():
        if neighbor not in search_team_path:
            search_team_path.append(neighbor)
            current_node = neighbor
            break

search_team_path.append(search_team_target)
print("Search Team Path:", search_team_path)

# ---------- Task 4: Sanitation Team Deployment (Simplified Chinese Postman Problem) ----------

def eulerian_cycle(graph):
    odd_degree_nodes = [node for node, degree in graph.degrees.items() if degree % 2 != 0]
    if len(odd_degree_nodes) == 0:
        return list(graph.nodes)
    
    if len(odd_degree_nodes) > 2:
        return None
    
    start_node = odd_degree_nodes[0]
    end_node = odd_degree_nodes[1] if len(odd_degree_nodes) > 1 else start_node
    
    stack = [start_node]
    cycle = []
    
    while stack:
        current_node = stack[-1]
        if graph.edges[current_node]:
            neighbor = next(iter(graph.edges[current_node]))
            stack.append(neighbor)
            graph.edges[current_node].pop(neighbor)
            graph.edges[neighbor].pop(current_node)
        else:
            cycle.append(stack.pop())
    
    if len(cycle) < len(graph.nodes):
        return None
    
    return cycle + [end_node] if start_node != end_node else cycle

def sanitation_team_path(graph):
    # Create a copy of the graph
    modified_graph = Graph()
    for node, data in graph.nodes.items():
        modified_graph.add_node(node, *data)
    
    for node1, neighbors in graph.edges.items():
        for node2, weight in neighbors.items():
            modified_graph.add_edge(node1, node2, weight)
            modified_graph.add_edge(node2, node1, weight)
    
    # Find a Eulerian cycle
    cycle = eulerian_cycle(modified_graph)
    
    return cycle

sanitation_team_path = sanitation_team_path(G)
print("Sanitation Team Path:", sanitation_team_path)

# Visualize the graph with nodes and edges
plt.figure(figsize=(12, 8))

# Create a layout for the nodes
pos = nx.spring_layout(G)

# Draw the nodes and edges
nx.draw_networkx_nodes(G, pos, node_size=500, node_color='lightblue', edgecolors='black')
nx.draw_networkx_edges(G, pos, width=1.5, edge_color='gray')

# Label the nodes
nx.draw_networkx_labels(G, pos)

plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.title('Graph Visualization')
plt.show()