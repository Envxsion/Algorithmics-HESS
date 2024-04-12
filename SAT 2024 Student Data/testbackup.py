import csv
import math
import networkx as nx
import matplotlib.pyplot as plt
from collections import defaultdict
from heapq import heappush, heappop

class PangobatResponseForce:
    def __init__(self, nodes_file, edges_file):
        self.nodes = self.load_nodes(nodes_file)
        self.edges = self.load_edges(edges_file)
        self.graph = self.build_graph()
        self.medical_team = MedicalTeam(self, self.nodes, self.edges, self.graph)
        self.search_team = SearchTeam(self, self.nodes, self.edges, self.graph)
        self.sanitation_team = SanitationTeam(self, self.nodes, self.edges, self.graph)

    def load_nodes(self, filename):
        nodes = {}
        with open(filename, 'r', encoding='utf-8-sig') as f:
            reader = csv.reader(f)
            for row in reader:
                name = row[0]
                pop = row[1]
                income = row[2]
                lat = float(row[3])
                long = float(row[4])
                age = row[5]
                node = {
                    'name': name,
                    'population': int(pop),
                    'income': int(income),
                    'latitude': lat,
                    'longitude': long,
                    'average_age': float(age)
                }
                nodes[name] = node
        return nodes

    def load_edges(self, filename):
        edges = []
        with open(filename, 'r', encoding='utf-8-sig') as f:
            reader = csv.reader(f)
            for row in reader:
                place1 = row[0]
                place2 = row[1]
                dist = float(row[2])
                time = float(row[3])
                edge = {
                    'place1': place1,
                    'place2': place2,
                    'distance': dist,
                    'travel_time': time
                }
                edges.append(edge)
        return edges

    def build_graph(self):
        G = nx.Graph()
        for node in self.nodes.values():
            G.add_node(node['name'], **node)
        for edge in self.edges:
            G.add_edge(edge['place1'], edge['place2'], weight=edge['distance'], travel_time=edge['travel_time'])
        return G

    def haversine(self, lat1, lon1, lat2, lon2):
        R = 6371.0
        lat1_rad = math.radians(lat1)
        lon1_rad = math.radians(lon1)
        lat2_rad = math.radians(lat2)
        lon2_rad = math.radians(lon2)

        dlat = lat2_rad - lat1_rad
        dlon = lon2_rad - lon1_rad

        a = math.sin(dlat / 2) ** 2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon / 2) ** 2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

        distance = R * c

        return distance

    def dijkstra_shortest_path(self, start, end):
        distances = {}
        prev_nodes = {}
        heap = [(0, start)]
        distances[start] = 0

        while heap:
            curr_dist, curr_node = heappop(heap)

            if curr_node == end:
                path = [end]
                while curr_node in prev_nodes:
                    curr_node = prev_nodes[curr_node]
                    path.insert(0, curr_node)
                return path, curr_dist, self.graph[path[0]][path[1]]['travel_time']

            if curr_dist > distances.get(curr_node, float('inf')):
                continue

            for neighbor, data in self.graph[curr_node].items():
                distance = curr_dist + data['weight']
                if distance < distances.get(neighbor, float('inf')):
                    distances[neighbor] = distance
                    prev_nodes[neighbor] = curr_node
                    heappush(heap, (distance, neighbor))

        return None, float('inf'), float('inf')

class MedicalTeam:
    def __init__(self, response_force, nodes, edges, graph):
        self.response_force = response_force
        self.nodes = nodes
        self.edges = edges
        self.graph = graph

    def vaccinate_population(self, target_location, radius):
        print(f"MedicalTeam: Vaccinating population within {radius} of {target_location}")
        for node in self.nodes.values():
            if self.response_force.haversine(node['latitude'], node['longitude'], self.nodes[target_location]['latitude'], self.nodes[target_location]['longitude']) <= radius:
                print(f"MedicalTeam: Vaccinating {node['name']} with population {node['population']}")

class SearchTeam:
    def __init__(self, response_force, nodes, edges, graph):
        self.response_force = response_force
        self.nodes = nodes
        self.edges = edges
        self.graph = graph

    def search_for_pangobat(self, target_location, radius):
        print(f"SearchTeam: Searching for Pangobat within {radius} of {target_location}")
        for node in self.nodes.values():
            if self.response_force.haversine(node['latitude'], node['longitude'], self.nodes[target_location]['latitude'], self.nodes[target_location]['longitude']) <= radius:
                print(f"SearchTeam: Searching {node['name']} for Pangobat")

class SanitationTeam:
    def __init__(self, response_force, nodes, edges, graph):
        self.response_force = response_force
        self.nodes = nodes
        self.edges = edges
        self.graph = graph

    def sanitize_roads(self, target_location, radius):
        print(f"SanitationTeam: Sanitizing roads within {radius} of {target_location}")
        for edge in self.edges:
            if self.response_force.haversine(self.nodes[edge['place1']]['latitude'], self.nodes[edge['place1']]['longitude'], self.nodes[target_location]['latitude'], self.nodes[target_location]['longitude']) <= radius or \
               self.response_force.haversine(self.nodes[edge['place2']]['latitude'], self.nodes[edge['place2']]['longitude'], self.nodes[target_location]['latitude'], self.nodes[target_location]['longitude']) <= radius:
                print(f"SanitationTeam: Spraying road between {edge['place1']} and {edge['place2']}")

class PangobatResponseManager:
    def __init__(self, nodes_file, edges_file):
        self.response_force = PangobatResponseForce(nodes_file, edges_file)

    def respond_to_pangobat_sighting(self, target_location, radius):
        print(f"PangobatResponseManager: Responding to Pangobat sighting in {target_location}")

        # Deploy the medical team
        self.response_force.medical_team.vaccinate_population(target_location, radius)

        # Deploy the search team
        self.response_force.search_team.search_for_pangobat(target_location, radius)

        # Deploy the sanitation team
        self.response_force.sanitation_team.sanitize_roads(target_location, radius)

        # Print the shortest path to the target location
        path, distance, travel_time = self.response_force.dijkstra_shortest_path('Bendigo', target_location)
        print(f"Shortest path from Bendigo to {target_location}: {' -> '.join(path)}")
        print(f"Distance: {distance:.2f} km")
        print(f"Travel time: {travel_time:.2f} minutes")

# Example usage
pangobat_response_manager = PangobatResponseManager('SAT 2024 Student Data/nodes.csv', 'SAT 2024 Student Data/edges.csv')
pangobat_response_manager.respond_to_pangobat_sighting('Ouyen', 100)