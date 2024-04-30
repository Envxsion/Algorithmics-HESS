import csv
import math
import networkx as nx
import matplotlib.pyplot as plt
from heapq import heappush, heappop
import random

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
                    'average_age': float(age),
                    'pangobat_virus': False 
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

        # No path found, return None
        return None, float('inf'), float('inf')

    def get_distance_and_travel_time(self, start, end):
        path, distance, travel_time = self.dijkstra_shortest_path(start, end)
        if len(path) < 2:
            return float('inf'), float('inf')
        else:
            return distance, travel_time

class MedicalTeam:
    def __init__(self, response_force, nodes, edges, graph):
        self.response_force = response_force
        self.nodes = nodes
        self.edges = edges
        self.graph = graph

    def vaccinate_population(self, target_location, radius):
        print(f"AllTeams: Going to Target Location ({target_location}) from Bendigo")
        path, distance, travel_time = self.response_force.dijkstra_shortest_path('Bendigo', target_location)
        if path is None:
            print(f"AllTeams: No path found to Target Location ({target_location}), unable to proceed.")
            return
        print(f"AllTeams: Going to Target Location using path: {' -> '.join(path)}")
        print(f"AllTeams: Distance to Target Location: {distance:.2f} km, Travel Time: {travel_time:.2f} minutes"+"\n")

        print(f"MedicalTeam: At Target Location ({target_location}), recalculating towns within radius:")
        towns_to_vaccinate = []
        for node_name, node in self.nodes.items():
            if self.response_force.haversine(node['latitude'], node['longitude'], self.nodes[target_location]['latitude'], self.nodes[target_location]['longitude']) <= radius and node_name != target_location:
                towns_to_vaccinate.append(node_name)

        towns_to_vaccinate.sort(key=lambda x: self.response_force.get_distance_and_travel_time(target_location, x)[0])  # Sort by distance from target location
        for town in towns_to_vaccinate:
            distance, travel_time = self.response_force.get_distance_and_travel_time(target_location, town)
            if distance == float('inf'):
                print(f"MedicalTeam: No path found to {town}, skipping vaccination.")
                continue
            print(f"MedicalTeam: Vaccinating {town} (distance: {distance:.2f} km, travel time: {travel_time:.2f} minutes)")
            path, _, _ = self.response_force.dijkstra_shortest_path(target_location, town)
            print(f"MedicalTeam: Going to {town} using path: {' -> '.join(path)}")

        print(f"MedicalTeam: Vaccinations Finished, returning to Bendigo")
        path, distance, travel_time = self.response_force.dijkstra_shortest_path(target_location, 'Bendigo')
        if path is None:
            print(f"MedicalTeam: No path found from Target Location to Bendigo, unable to return.")
            return
        print(f"MedicalTeam: Returning to Bendigo using path: {' -> '.join(path)}")
        print(f"MedicalTeam: Distance to Bendigo: {distance:.2f} km, Travel Time: {travel_time:.2f} minutes"+"\n")

class SearchTeam:
    def __init__(self, response_force, nodes, edges, graph):
        self.response_force = response_force
        self.nodes = nodes
        self.edges = edges
        self.graph = graph

    def tsp(self, start):
        unvisited = set(self.nodes.keys())
        unvisited.remove(start)
        current = start
        path = [start]
        total_distance = 0
        total_travel_time = 0

        while unvisited:
            next_node = min(unvisited, key=lambda x: self.response_force.get_distance_and_travel_time(current, x)[0])
            distance, travel_time = self.response_force.get_distance_and_travel_time(current, next_node)
            total_distance += distance
            total_travel_time += travel_time
            path.append(next_node)
            unvisited.remove(next_node)
            current = next_node

        # Return to start
        distance, travel_time = self.response_force.get_distance_and_travel_time(current, start)
        total_distance += distance
        total_travel_time += travel_time
        path.append(start)

        return path, total_distance, total_travel_time

    def search_for_pangobat(self, radius):
        print(f"SearchTeam: Searching for Pangobat within {radius} of all towns")
        start_location = 'Bendigo'

        # Use the TSP approach to find the optimal path for searching
        path, total_distance, total_travel_time = self.tsp(start_location)

        # Output the path and details
        search_path_msg = f"SearchTeam: Searching towns using path: {' -> '.join(path)}\n"
        for town in path[:-1]:  # Exclude the last town (Bendigo) since it's already printed in the path
            if self.nodes[town]['pangobat_virus']:
                search_path_msg += f"{town}: Pangobat Virus Found\n"
            else:
                search_path_msg += f"{town}: No Pangobat Virus found\n"

        print(search_path_msg)
        print(f"SearchTeam: Total distance: {total_distance:.2f} km, Total travel time: {total_travel_time:.2f} minutes")

        print("SearchTeam: Search process complete")

class SanitationTeam:
    def __init__(self, response_force, nodes, edges, graph):
        self.response_force = response_force
        self.nodes = nodes
        self.edges = edges
        self.graph = graph

    def nearest_neighbor_route(self, start):
        unvisited = set(self.nodes.keys())
        unvisited.remove(start)
        current = start
        path = [start]
        total_distance = 0
        total_travel_time = 0

        while unvisited:
            next_node = min(unvisited, key=lambda x: self.response_force.get_distance_and_travel_time(current, x)[0])
            distance, travel_time = self.response_force.get_distance_and_travel_time(current, next_node)
            total_distance += distance
            total_travel_time += travel_time
            path.append(next_node)
            unvisited.remove(next_node)
            current = next_node

        # Add the last edge to return to the starting point
        distance, travel_time = self.response_force.get_distance_and_travel_time(path[-1], start)
        total_distance += distance
        total_travel_time += travel_time

        return path, total_distance, total_travel_time

    def sanitize_roads(self):
        print("\n"+"SanitationTeam: Sanitizing all roads on the map")

        # Choose a starting location (e.g., Bendigo)
        start_location = 'Bendigo'

        # Use the Nearest Neighbor Algorithm to find the optimal path for sanitizing roads
        path, total_distance, total_travel_time = self.nearest_neighbor_route(start_location)

        # Output the path and details
        sanitized_path_msg = "Sanitation Route:\n"
        for i in range(len(path) - 1):
            place1 = path[i]
            place2 = path[i + 1]
            distance, travel_time = self.response_force.get_distance_and_travel_time(place1, place2)
            sanitized_path_msg += f"{place1} -> {place2} (distance: {distance:.2f} km, travel time: {travel_time:.2f} minutes)\n"

        print(sanitized_path_msg)
        print(f"SanitationTeam: Total distance: {total_distance:.2f} km, Total travel time: {total_travel_time:.2f} minutes")

        print("SanitationTeam: Sanitation process complete")



class PangobatResponseManager:
    def __init__(self, nodes_file, edges_file):
        self.response_force = PangobatResponseForce(nodes_file, edges_file)

    def respond_to_pangobat_sighting(self, target_location, radius):
        print(f"PangobatResponseManager: Responding to Pangobat sighting in {target_location}")
        # Randomly tag some towns with the Pangobat virus tag within the radius
        infected_towns = []
        for node in self.response_force.nodes.values():
            if self.response_force.haversine(node['latitude'], node['longitude'], self.response_force.nodes[target_location]['latitude'], self.response_force.nodes[target_location]['longitude']) <= radius:
                if random.random() < 0.5:  # Adjust the probability as needed
                    node['pangobat_virus'] = True
                    infected_towns.append(node['name'])
        if infected_towns:
            print(f"Pangobat virus detected in the following towns: {', '.join(infected_towns)}"+"\n")
        # Deploy the medical team
        self.response_force.medical_team.vaccinate_population(target_location, radius)

        
        # Deploy the search team
        self.response_force.search_team.search_for_pangobat(radius)

        # Deploy the sanitation team
        self.response_force.sanitation_team.sanitize_roads()  # Removed the parameters
        
        # Visualize the network
        self.visualize_network()

    def visualize_network(self):
        plt.figure(figsize=(20,16))
        pos = {node['name']: (node['longitude'], node['latitude']) for node in self.response_force.nodes.values()}
        nx.draw(self.response_force.graph, pos, with_labels=True, node_size=100, font_size=10)
        plt.show()


# Example usage
pangobat_response_manager = PangobatResponseManager('SAT 2024 Student Data/nodes.csv', 'SAT 2024 Student Data/edges.csv')
pangobat_response_manager.respond_to_pangobat_sighting('Rye', 110)
