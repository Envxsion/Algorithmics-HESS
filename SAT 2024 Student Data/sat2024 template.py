import networkx as nx
import matplotlib.pyplot as plt
import csv
import math
from queue import PriorityQueue
import random

class Node:
    def __init__(self, name, pop, income, age, lat, long):
        self.name = name
        self.lat = lat
        self.long = long
        self.pop = int(pop)
        self.income = int(income)
        self.age = float(age)
        self.colour = 1
        self.neighbours = []

    def add_neighbour(self, neighbour):
        if neighbour not in self.neighbours:
            self.neighbours.append(neighbour)

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
        self.colour_dict = {0: "black", 1: "blue", 2: "red", 3: "green", 4: "yellow", 5: "lightblue"}

    def load_data(self):
        with open("SAT 2024 Student Data/nodes.csv", 'r', encoding='utf-8-sig') as csvfile:
            csv_reader = csv.reader(csvfile)
            for row in csv_reader:
                name = row[0]
                pop = row[1]
                income = row[2]
                age = row[5]
                lat = float(row[3])
                long = float(row[4])
                node = Node(name, pop, income, age, lat, long)
                self.nodes.append(node)

        with open("SAT 2024 Student Data/edges.csv", "r", encoding='utf-8-sig') as csvfile:
            csv_reader = csv.reader(csvfile)
            for row in csv_reader:
                place1 = row[0]
                place2 = row[1]
                dist = int(row[2])
                time = int(row[3])

                #Add a small random variation to the time or distance, tie between Melton and Sunbury
                time += random.uniform(-0.5, 0.5)  

                edge = Edge(place1, place2, dist, time)
                self.edges.append(edge)

                for node in self.nodes:
                    if node.name == place1:
                        node.add_neighbour(place2)
                    if node.name == place2:
                        node.add_neighbour(place1)

    def get_dist(self, place1, place2):
        for edge in self.edges:
            if edge.place1 == place1 and edge.place2 == place2:
                return edge.dist
            if edge.place1 == place2 and edge.place2 == place1:
                return edge.dist
        return float('inf') #negative loop if I return -1, cause problems with distacne between Melton and Sunbury

    def display(self, filename="map.png", figsize=(20, 16), scale_factor=2):
        edge_labels = {}
        edge_colours = []
        G = nx.Graph()
        node_colour_list = []
        for node in self.nodes:
            G.add_node(node.name, pos=(node.long * scale_factor, node.lat * scale_factor))
            node_colour_list.append(self.colour_dict[node.colour])
        for edge in self.edges:
            G.add_edge(edge.place1, edge.place2)
            edge_labels[(edge.place1, edge.place2)] = edge.dist
            edge_colours.append(self.colour_dict[edge.colour])
        node_positions = nx.get_node_attributes(G, 'pos')

        plt.figure(figsize=figsize)
        nx.draw(G, node_positions, with_labels=True, node_size=100, node_color=node_colour_list, font_size=10,
                font_color='black', font_weight='bold', edge_color=edge_colours)
        nx.draw_networkx_edge_labels(G, node_positions, edge_labels=edge_labels)
        plt.title('')
        plt.axis('off')
        plt.savefig(filename, bbox_inches='tight')
        plt.show()

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

    def dijkstra(self, start, end):
        distances = {node: float('inf') for node in self.nodes}
        distances[start] = 0
        pq = PriorityQueue()
        pq.put((0, start))

        while not pq.empty():
            (dist, current_node) = pq.get()
            if dist > distances[current_node]:
                continue

            for neighbor in current_node.neighbours:
                distance = self.get_dist(current_node.name, neighbor)
                if distance == float('inf'):
                    continue
                new_dist = distances[current_node] + distance
                if new_dist < distances[neighbor]:
                    distances[neighbor] = new_dist
                    pq.put((new_dist, neighbor))

        return distances[end]

    def searchteam(self, target, radius):
        start_node = self.nodes[0]  # Assuming Bendigo is the first node
        towns_in_radius = [node for node in self.nodes if self.haversine(start_node.lat, start_node.long, node.lat, node.long) <= radius]
        search_order = sorted(towns_in_radius, key=lambda node: self.dijkstra(start_node, node))
        search_team_size = len(search_order)
        print(f"The search team will require {search_team_size} people, and will visit (in time order):")
        for town in search_order:
            print(town.name)

    def vaccinate(self, target, radius):
        start_node = self.nodes[0]  # Assuming Bendigo is the first node
        towns_in_radius = [node for node in self.nodes if self.haversine(start_node.lat, start_node.long, node.lat, node.long) <= radius]
        vaccination_order = sorted(towns_in_radius, key=lambda node: (node.age, -node.pop), reverse=True)
        total_distance = 0
        total_time = 0
        current_node = start_node
        print("The medical team should visit towns in this order:")
        for town in vaccination_order:
            distance = self.dijkstra(current_node, town)  # Changed this line
            total_distance += distance
            time = self.get_dist(current_node.name, town.name)
            total_time += time
            print(f"{town.name} (Distance: {distance} km, Time: {time} mins)")
            current_node = town  # Changed this line
        print(f"This will take a total of {total_time} mins to cover {total_distance} km.")

    def findpath(self, target):
        start_node = self.nodes[8]  # Bendigo
        print("Start Node:", start_node.name)
        target_node = next((node for node in self.nodes if node.name == target), None)
        if target_node is None:
            print(f"Target town '{target}' not found.")
            return

        path = []
        visited = set()
        current_node = start_node
        total_distance = 0
        total_time = 0

        def find_next_node(current_node, visited, path):
            unvisited_neighbors = [neighbor for neighbor in current_node.neighbours if neighbor not in visited and neighbor not in path]
            if not unvisited_neighbors:
                return None
            return min(unvisited_neighbors, key=lambda neighbor: self.get_dist(current_node.name, neighbor))

        while current_node != target_node:
            visited.add(current_node)
            try:
                print("Visited Neighbors:", ', '.join(neighbor.name for neighbor in visited))
            except:
                pass

            print("Current Node: ", current_node.name)
            unvisited_neighbors = [neighbor for neighbor in current_node.neighbours if neighbor not in visited and neighbor not in path]
            print("Unvisited Neighbors:", unvisited_neighbors)
            if not unvisited_neighbors:
                if not path:
                    print(f"Unable to reach {target_node.name} from {start_node.name}.")
                    return
                # Backtrack
                while path and path[-1] in visited:
                    previous_node = path.pop()
                    print("Previous Node:", previous_node.name)
                    print("Backtracking from", current_node.name, "to", previous_node)
                    total_distance -= self.get_dist(previous_node, current_node.name)
                    current_node = next((node for node in self.nodes if node.name == previous_node), None)

                if not path:
                    print(f"Unable to reach {target_node.name} from {start_node.name}.")
                    return

                continue

            next_node = find_next_node(current_node, visited, path)
            print("Next Node:", next_node)
            if next_node is None:
                print(f"Unable to reach {target_node.name} from {start_node.name}.")
                return

            next_node_obj = next((node for node in self.nodes if node.name == next_node), None)

            if next_node_obj is None:
                print(f"Node '{next_node}' not found.")
                return

            path.append(next_node)
            distance = self.get_dist(current_node.name, next_node)
            total_distance += distance
            time = self.get_dist(current_node.name, next_node)
            total_time += time
            current_node = next_node_obj

            # Reset visited set within the current path
            visited.clear()
            visited.update(path)

        print(f"Travel to {target_node.name} via {', '.join(path)}, taking {total_time} mins to drive {total_distance} km.")






original = Graph()
original.load_data()
original.display("map.png")

target_town = "Sunbury"
radius = 100

print(f"Input: {target_town}, {radius}")
print()

original.findpath(target_town)
print()

towns_in_radius = [node for node in original.nodes if original.haversine(original.nodes[0].lat, original.nodes[0].long, node.lat, node.long) <= radius]
town_names = [town.name for town in towns_in_radius]
print(f"Towns within {radius} km of {target_town} are: {', '.join(town_names)}.")
print()

original.vaccinate(target_town, radius)
print()
original.searchteam(target_town, radius)