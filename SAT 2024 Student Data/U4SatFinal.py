from matplotlib import pyplot as plt
import os
import pandas as pd
import networkx as nx
import math
import time
import pygame
import random
import heapq
from collections import defaultdict
from os import environ
from win32gui import SetWindowPos
import tkinter as tk
import numpy as np
import itertools

#bring window to front forcefully (PyGame)
from os import environ
environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
import pygame  # import after disabling environ prompt
from win32gui import SetWindowPos
import tkinter as tk
root = tk.Tk() 
root.withdraw()
screen_w, screen_h = root.winfo_screenwidth(), root.winfo_screenheight()
BLACK = '\033[30m'
RED = '\033[31m'
GREEN = '\033[32m'
YELLOW = '\033[33m'  # orange on some systems
BLUE = '\033[34m'
MAGENTA = '\033[35m'
CYAN = '\033[36m'
LIGHT_GRAY = '\033[37m'
DARK_GRAY = '\033[90m'
BRIGHT_RED = '\033[91m'
BRIGHT_GREEN = '\033[92m'
BRIGHT_YELLOW = '\033[93m'
BRIGHT_BLUE = '\033[94m'
BRIGHT_MAGENTA = '\033[95m'
BRIGHT_CYAN = '\033[96m'
WHITE = '\033[97m'
BOLD = '\033[1m'
UNDERLINE = '\033[4m'
RESET = '\033[0m'  # called to return to standard terminal text color

class PangobatResponseManager:
    def __init__(self, edges_file, nodes_file):
        self.edges_file = edges_file
        self.nodes_file = nodes_file
        self.G = None
        self.edges = None
        self.nodes = None
        self.infected_towns = []
        self.last_best_file = 'SAT 2024 Student Data/lastbest.csv'
        self.ts = None
        self.tt = None
        self.tp = None
        self.tc = None

    def load_data(self):
        self.edges = pd.read_csv(self.edges_file, header=0, names=['Town1', 'Town2', 'Distance', 'Time'])
        self.nodes = pd.read_csv(self.nodes_file, header=0, names=['Town', 'Population', 'Income', 'Latitude', 'Longitude', 'Age'])
        self.G = nx.Graph()
        for i, row in self.edges.iterrows():
            self.G.add_edge(row['Town1'], row['Town2'], distance=row['Distance'], time=row['Time'])
        pos = {town: (lon, lat) for town, lon, lat in zip(self.nodes['Town'], self.nodes['Longitude'], self.nodes['Latitude'])}
        nx.set_node_attributes(self.G, pos, 'pos')

    def haversine_distance(self, lon1, lat1, lon2, lat2):
        lon1, lat1, lon2, lat2 = map(math.radians, [lon1, lat1, lon2, lat2])
        dlon = lon2 - lon1
        dlat = lat2 - lat1
        a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
        c = 2 * math.asin(math.sqrt(a))
        r = 6371
        return c * r

    def heuristic(self, node1, node2):
        lon1, lat1 = self.nodes.loc[self.nodes['Town'] == node1, ['Longitude', 'Latitude']].values[0]
        lon2, lat2 = self.nodes.loc[self.nodes['Town'] == node2, ['Longitude', 'Latitude']].values[0]
        return self.haversine_distance(lon1, lat1, lon2, lat2)

    def identify_infected_towns(self):
        random.seed(42)
        infection_probability = 0.3
        for index, row in self.nodes.iterrows():
            town = row['Town']
            if random.random() < infection_probability:
                if town not in self.infected_towns:
                    self.infected_towns.append(town)
        print(f"{BRIGHT_RED}{UNDERLINE}Infected Towns:{RESET} {RED}{self.infected_towns}{RESET}")
        
    def time_algorithm(self, algorithm_name, func, *args, **kwargs):
        if algorithm_name == 'Held-Karp':
            start_time = time.time()
            path, distance = func(*args, **kwargs)
            end_time = time.time()
            print(f"Vaccination Path: {' -> '.join(path)}")
            print(f"Total Distance: {distance:.2f} km")
        else:
            start_time = time.time()
            func(*args, **kwargs)
            end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"{algorithm_name} took {elapsed_time:.4f} seconds.")

        # Read or create the lastbest.csv file
        if not os.path.exists(self.last_best_file):
            # Create a new DataFrame with the current algorithm time
            df = pd.DataFrame({
                'Algorithm': [algorithm_name],
                'Time': [elapsed_time]
            })
            df.to_csv(self.last_best_file, index=False)
            print(f"Recorded {BRIGHT_YELLOW}{UNDERLINE}{elapsed_time:.4f} seconds{RESET} for {MAGENTA}{algorithm_name}{RESET} in a new file.")
        else:
            try:
                df = pd.read_csv(self.last_best_file)
                if df.empty:
                    # File is empty, create a new DataFrame with the current algorithm time
                    df = pd.DataFrame({
                        'Algorithm': [algorithm_name],
                        'Time': [elapsed_time]
                    })
                else:
                    if algorithm_name in df['Algorithm'].values:
                        best_time = df.loc[df['Algorithm'] == algorithm_name, 'Time'].values[0]
                        if elapsed_time < best_time:
                            df.loc[df['Algorithm'] == algorithm_name, 'Time'] = elapsed_time
                            print(f"{YELLOW}Previous best {RESET} time was {BRIGHT_YELLOW}{UNDERLINE}{best_time} seconds{RESET}. Updated best time for {UNDERLINE}{algorithm_name}{RESET} to {BRIGHT_GREEN}{elapsed_time:.4f} seconds{RESET}.")
                        else:
                            print(f"{RED}No improvement{RESET} for {YELLOW}{algorithm_name}{RESET}. Best time remains {BRIGHT_YELLOW}{UNDERLINE}{best_time:.4f} seconds{RESET}.")
                    else:
                        new_entry = pd.DataFrame({
                            'Algorithm': [algorithm_name],
                            'Time': [elapsed_time]
                        })
                        df = pd.concat([df, new_entry], ignore_index=True)
                        print(f"Recorded {BRIGHT_YELLOW}{UNDERLINE}{elapsed_time:.4f} seconds{RESET} for {MAGENTA}{algorithm_name}{RESET} in the file.")

                df.to_csv(self.last_best_file, index=False)
            except pd.errors.EmptyDataError:
                # File exists but is empty, create a new DataFrame with the current algorithm time
                df = pd.DataFrame({
                    'Algorithm': [algorithm_name],
                    'Time': [elapsed_time]
                })
                df.to_csv(self.last_best_file, index=False)
                print(f"Recorded {elapsed_time:.4f} seconds for {algorithm_name} in a new file.")
        
    def a_star_timed(self, start, target, use_bidirectional=False):
        if use_bidirectional:
            self.time_algorithm('Bidirectional A*', self.bidirectional_a_star, start, target, self.heuristic)
        else:
            self.time_algorithm('A*', self.a_star_algorithm_internal, start, target)
        self.visualize_path(self.ts,self.tt,self.tp,self.tc)
        
    def visualize_graph(self):
        plt.figure(figsize=(12, 8))
        pos = nx.get_node_attributes(self.G, 'pos')
        nx.draw_networkx_nodes(self.G, pos, node_color='skyblue', node_size=300, alpha=0.8)
        nx.draw_networkx_edges(self.G, pos, edge_color='gray', width=1.0, alpha=0.5)
        labels = {town: town for town in self.G.nodes}
        nx.draw_networkx_labels(self.G, pos, labels, font_size=10, font_color='black')
        plt.title('Pangobat Response Network')
        plt.axis('off')
        plt.tight_layout()
        plt.show()

    def visualize_path(self, start, target, path, came_from):
        def draw_grid():
            screen.fill((0, 0, 0))
            for edge in self.G.edges:
                pygame.draw.line(screen, (128, 128, 128), pos_scaled[edge[0]], pos_scaled[edge[1]], 1)
                # Labels
                edge_midpoint = ((pos_scaled[edge[0]][0] + pos_scaled[edge[1]][0]) // 2, 
                                 (pos_scaled[edge[0]][1] + pos_scaled[edge[1]][1]) // 2)
                distance = self.G[edge[0]][edge[1]]['distance']
                text_surface = font.render(str(distance), True, (255, 255, 255))
                screen.blit(text_surface, edge_midpoint)
            for node in self.G.nodes:
                color = (255, 255, 255)
                if node == start:
                    color = (255, 0, 0)
                elif node == target:
                    color = (0, 255, 0)
                elif node in came_from:
                    color = (0, 0, 255)
                pygame.draw.circle(screen, color, pos_scaled[node], node_radius)
                # Draw node labels
                text_surface = font.render(node, True, (255, 255, 255))
                screen.blit(text_surface, (pos_scaled[node][0] + node_radius, pos_scaled[node][1] + node_radius))
            pygame.display.flip()
    
        def draw_path():
            for current in path:
                if current in came_from:
                    pygame.draw.line(screen, (0, 255, 0), pos_scaled[current], pos_scaled[came_from[current]], 3)
                    pygame.display.flip()
                    time.sleep(0.5)
    
        # Grid and Pygame settings
        grid_size = 28
        node_radius = 14
        screen_width = 1600
        screen_height = 1200
    
        pygame.init()
        screen = pygame.display.set_mode((screen_width, screen_height))
        x = round((screen_w - screen_width) / 2)
        y = round((screen_h - screen_height) / 2 * 0.8)
        SetWindowPos(pygame.display.get_wm_info()['window'], -1, x, y, 0, 0, 1)
        pygame.display.set_caption("A* Algorithm Visualization")
        font = pygame.font.SysFont(None, 20)
    
        pos = nx.get_node_attributes(self.G, 'pos')
        min_lon = min(pos[node][0] for node in pos)
        max_lon = max(pos[node][0] for node in pos)
        min_lat = min(pos[node][1] for node in pos)
        max_lat = max(pos[node][1] for node in pos)
    
        pos_scaled = {node: (int((lon - min_lon) / (max_lon - min_lon) * (screen_width - 40) + 20), 
                             int((lat - min_lat) / (max_lat - min_lat) * (screen_height - 40) + 20)) 
                      for node, (lon, lat) in pos.items()}
    
        draw_grid()
        draw_path()
    
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return

    def a_star_algorithm_internal(self, start, target):
        def heuristic(node1, node2):
            lon1, lat1 = self.nodes.loc[self.nodes['Town'] == node1, ['Longitude', 'Latitude']].values[0]
            lon2, lat2 = self.nodes.loc[self.nodes['Town'] == node2, ['Longitude', 'Latitude']].values[0]
            return self.haversine_distance(lon1, lat1, lon2, lat2)
        
        frontier = [(0, start)]
        heapq.heapify(frontier)
        came_from = {}
        cost_so_far = {start: 0}
        visited = set()

        while frontier:
            current_priority, current_node = heapq.heappop(frontier)

            if current_node == target:
                path = []
                while current_node != start:
                    path.append(current_node)
                    current_node = came_from[current_node]
                path.append(start)
                path.reverse()
                self.ts = start
                self.tt = target
                self.tp = path
                self.tc = came_from
                print(f"{BRIGHT_GREEN}A* Path from {start} to {target}: {' -> '.join(path)}{RESET}")
                return path

            visited.add(current_node)

            for neighbor in self.G.neighbors(current_node):
                new_cost = cost_so_far[current_node] + self.G[current_node][neighbor]['distance']

                if neighbor not in cost_so_far or new_cost < cost_so_far[neighbor]:
                    cost_so_far[neighbor] = new_cost
                    priority = new_cost + heuristic(neighbor, target)
                    heapq.heappush(frontier, (priority, neighbor))
                    came_from[neighbor] = current_node

        return None

    def bidirectional_a_star(self, start, target, heuristic):
        def reconstruct_path(came_from, current):
            path = [current]
            while current in came_from:
                current = came_from[current]
                path.append(current)
            return path[::-1]

        open_set_start = [(0, start)]
        open_set_target = [(0, target)]
        heapq.heapify(open_set_start)
        heapq.heapify(open_set_target)

        came_from_start = {}
        came_from_target = {}
        g_score_start = {node: float('inf') for node in self.G.nodes}
        g_score_target = {node: float('inf') for node in self.G.nodes}
        g_score_start[start] = 0
        g_score_target[target] = 0
        f_score_start = {node: float('inf') for node in self.G.nodes}
        f_score_target = {node: float('inf') for node in self.G.nodes}
        f_score_start[start] = heuristic(start, target)
        f_score_target[target] = heuristic(target, start)

        visited_start = set()
        visited_target = set()

        meeting_node = None

        while open_set_start and open_set_target:
            current_start = heapq.heappop(open_set_start)[1]
            current_target = heapq.heappop(open_set_target)[1]

            visited_start.add(current_start)
            visited_target.add(current_target)

            if current_start in came_from_target:
                meeting_node = current_start
                break

            if current_target in came_from_start:
                meeting_node = current_target
                break

            for neighbor in self.G.neighbors(current_start):
                tentative_g_score = g_score_start[current_start] + self.G[current_start][neighbor]['distance']
                if tentative_g_score < g_score_start[neighbor]:
                    came_from_start[neighbor] = current_start
                    g_score_start[neighbor] = tentative_g_score
                    f_score_start[neighbor] = g_score_start[neighbor] + heuristic(neighbor, target)
                    heapq.heappush(open_set_start, (f_score_start[neighbor], neighbor))

            for neighbor in self.G.neighbors(current_target):
                tentative_g_score = g_score_target[current_target] + self.G[current_target][neighbor]['distance']
                if tentative_g_score < g_score_target[neighbor]:
                    came_from_target[neighbor] = current_target
                    g_score_target[neighbor] = tentative_g_score
                    f_score_target[neighbor] = g_score_target[neighbor] + heuristic(neighbor, start)
                    heapq.heappush(open_set_target, (f_score_target[neighbor], neighbor))

        if meeting_node is None:
            print(f"No path found between {start} and {target}.")
            return

        path_start = reconstruct_path(came_from_start, meeting_node)
        path_target = reconstruct_path(came_from_target, meeting_node)
        path_target.reverse()

        path = path_start + path_target[1:]

        # Print the nodes visited by each A* search
        print(f"{BRIGHT_GREEN}Nodes visited by Bi-A* search from {start}: {', '.join(visited_start)}{RESET}")
        print(f"{BRIGHT_GREEN}Nodes visited by Bi-A* search from {target}: {', '.join(visited_target)}{RESET}")

        # Print the combined path
        print(f"{BRIGHT_GREEN}Combined Path from {start} to {target}: {' -> '.join(path)}{RESET}")
        self.ts = start
        self.tt = target
        self.tp = path
        self.tc = {**came_from_start, **came_from_target}
    def held_karp(self, nodes_within_radius, start):
        n = len(nodes_within_radius)
        all_nodes = range(n)
        start_index = nodes_within_radius.index(start)
        dist = lambda i, j: self.haversine_distance(
            self.nodes.loc[self.nodes['Town'] == nodes_within_radius[i], 'Longitude'].values[0],
            self.nodes.loc[self.nodes['Town'] == nodes_within_radius[i], 'Latitude'].values[0],
            self.nodes.loc[self.nodes['Town'] == nodes_within_radius[j], 'Longitude'].values[0],
            self.nodes.loc[self.nodes['Town'] == nodes_within_radius[j], 'Latitude'].values[0]
        )
        
        memo = {}
        for k in all_nodes:
            if k != start_index:
                memo[(1 << k, k)] = dist(start_index, k)
        
        for subset_size in range(2, n):
            for subset in itertools.combinations([x for x in all_nodes if x != start_index], subset_size):
                bits = 1 << start_index
                for bit in subset:
                    bits |= 1 << bit
                for k in subset:
                    prev_bits = bits & ~(1 << k)
                    result = float('inf')
                    for m in subset:
                        if m == k:
                            continue
                        new_distance = memo[(prev_bits, m)] + dist(m, k)
                        if new_distance < result:
                            result = new_distance
                    memo[(bits, k)] = result
        
        bits = (1 << n) - 1
        result = float('inf')
        for k in all_nodes:
            if k == start_index:
                continue
            new_distance = memo[(bits, k)] + dist(k, start_index)
            if new_distance < result:
                result = new_distance
        return result

    def simulated_annealing(self, nodes_within_radius, start):
        def distance(route):
            return sum(self.haversine_distance(
                self.nodes.loc[self.nodes['Town'] == route[i], 'Longitude'].values[0],
                self.nodes.loc[self.nodes['Town'] == route[i], 'Latitude'].values[0],
                self.nodes.loc[self.nodes['Town'] == route[i + 1], 'Longitude'].values[0],
                self.nodes.loc[self.nodes['Town'] == route[i + 1], 'Latitude'].values[0]
            ) for i in range(len(route) - 1))
        
        current_route = nodes_within_radius[:]
        current_route.remove(start)
        current_route.insert(0, start)
        current_distance = distance(current_route)
        T = 1.0
        T_min = 0.00001
        alpha = 0.995
        
        while T > T_min:
            i, j = random.sample(range(1, len(current_route)), 2)
            new_route = current_route[:]
            new_route[i], new_route[j] = new_route[j], new_route[i]
            new_distance = distance(new_route)
            
            if new_distance < current_distance or random.random() < math.exp((current_distance - new_distance) / T):
                current_route, current_distance = new_route, new_distance
            
            T *= alpha
        
        return current_route, current_distance

    def find_nodes_within_radius(self, target, radius):
        target_lon, target_lat = self.nodes.loc[self.nodes['Town'] == target, ['Longitude', 'Latitude']].values[0]
        nodes_within_radius = [
            row['Town'] for index, row in self.nodes.iterrows()
            if self.haversine_distance(target_lon, target_lat, row['Longitude'], row['Latitude']) <= float(radius)
        ]
        return nodes_within_radius

    def visualize_vaccination_path(self, path):
        def draw_grid():
            screen.fill((0, 0, 0))
            for edge in self.G.edges:
                pygame.draw.line(screen, (128, 128, 128), pos_scaled[edge[0]], pos_scaled[edge[1]], 1)
            for node in self.G.nodes:
                color = (255, 255, 255)
                pygame.draw.circle(screen, color, pos_scaled[node], node_radius)
                text_surface = font.render(node, True, (255, 255, 255))
                screen.blit(text_surface, (pos_scaled[node][0] + node_radius, pos_scaled[node][1] + node_radius))
            pygame.display.flip()

        def draw_path(path):
            for i in range(len(path) - 1):
                pygame.draw.line(screen, (0, 255, 0), pos_scaled[path[i]], pos_scaled[path[i + 1]], 3)
                pygame.display.flip()
                time.sleep(0.5)

        grid_size = 28
        node_radius = 14
        screen_width = 1600
        screen_height = 1200

        pygame.init()
        screen = pygame.display.set_mode((screen_width, screen_height))
        x = round((screen_w - screen_width) / 2)
        y = round((screen_h - screen_height) / 2 * 0.8)
        SetWindowPos(pygame.display.get_wm_info()['window'], -1, x, y, 0, 0, 1)
        pygame.display.set_caption("Vaccination Path Visualization")
        font = pygame.font.SysFont(None, 20)

        pos = nx.get_node_attributes(self.G, 'pos')
        min_lon = min(pos[node][0] for node in pos)
        max_lon = max(pos[node][0] for node in pos)
        min_lat = min(pos[node][1] for node in pos)
        max_lat = max(pos[node][1] for node in pos)

        pos_scaled = {node: (int((lon - min_lon) / (max_lon - min_lon) * (screen_width - 40) + 20),
                             int((lat - min_lat) / (max_lat - min_lat) * (screen_height - 40) + 20))
                      for node, (lon, lat) in pos.items()}

        draw_grid()
        draw_path(path)

        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return

    def run_vaccination(self, target_town, radius):
        nodes_within_radius = self.find_nodes_within_radius(target_town, radius)
        if target_town not in nodes_within_radius:
            nodes_within_radius.insert(0, target_town)

        print(f"Nodes within radius: {nodes_within_radius}")
        response_manager.time_algorithm('Held-Karp', self.simulated_annealing, nodes_within_radius, target_town)


if __name__ == "__main__":
    input_prompt = input(f"Would you like to {BRIGHT_BLUE}{UNDERLINE}load edges and nodes?{RESET}{BLUE} [y/n]{RESET} ")
    if input_prompt.lower() == 'y':
        edges_file = 'SAT 2024 Student Data/edges.csv'
        nodes_file = 'SAT 2024 Student Data/nodes.csv'
        last_best = 'SAT 2024 Student Data/lastbest.csv'
        response_manager = PangobatResponseManager(edges_file, nodes_file)
        response_manager.load_data()
        print_prompt = input(f"Would you like to print {BRIGHT_RED}{UNDERLINE}all loaded data?{RESET}{RED}[y/n]{RESET} ")
        if print_prompt.lower() == 'y':
            print("Graph Data:", response_manager.G.nodes.data(), response_manager.G.edges.data(), "\nNodes Dataframe:", response_manager.nodes, "\nEdges Dataframe:", response_manager.edges)
        vis_map = input(f"{BRIGHT_YELLOW}Display map? [y/n]{RESET} ")
        if vis_map.lower() == 'y':
            print(f"\n{BRIGHT_RED}{BOLD}LOADING GRAPH...{RESET}")
            time.sleep(1)
            response_manager.visualize_graph()
        target_town = input(f"Input the {UNDERLINE}target town{RESET} to setup {MAGENTA}base camp{RESET} in, {YELLOW}[?]{RESET} [Town]: ")
        if target_town == "?" or target_town not in response_manager.G.nodes:
            all_towns = list(response_manager.G.nodes)
            print(f"{GREEN}All towns: {', '.join(all_towns)}{RESET}")
            target_town = input(f"{UNDERLINE}Enter Choice:{RESET} ")
        radius = input(f"{YELLOW}Enter search radius:{RESET} ")
        response_manager.identify_infected_towns()
        start_town = "Bendigo"

        bidirectional_flag = input(f"Would you like to use {BRIGHT_BLUE}{UNDERLINE}bidirectional A*{RESET}{BLUE}? [y/n]{RESET} ")
        bidirectional = bidirectional_flag.lower() == 'y'
        
        response_manager.a_star_timed(start_town, target_town, bidirectional)
        response_manager.run_vaccination(target_town, radius)
    else:
        print(RED + "Exiting..." + RESET)
