import streamlit as st
import pandas as pd
import numpy as np
import pandas as pd
from datetime import date
import random
import networkx as nx
import matplotlib.pyplot as plt
import requests

api_key = '5ae2e3f221c38a28845f05b6daacc6d880fc4ea844546eeaca216f24'

def fetch_attractions(api_key, city):
    url = f"https://api.opentripmap.com/0.1/en/places/geoname?name={city}&apikey={api_key}"
    response = requests.get(url)
    data = response.json()
    if "error" in data:
        print("Error:", data["error"])
        return None
    else:
        attractions = data["places"]
        return attractions

# Placeholder for retrieved data
attractions = []  # List of dictionaries containing attraction details

def find_shortest_path(starting_point):
  # Replace with your actual pathfinding implementation
  return [], 0, 0

def main():
  st.title("Algorithmics Project: Tourist City Exploration")

  with st.sidebar:
    # City search input
    city_name = st.text_input("Enter a city name:")

    # Map visualization settings
    map_type = st.selectbox("Map type", ["Standard", "Satellite", "Terrain"])
    zoom_level = st.slider("Zoom level", min_value=5, max_value=18, value=15)

    # Pathfinding settings
    travel_mode = st.selectbox("Travel mode", ["Walking", "Public transport", "Both"])
    consider_walking_distance = st.checkbox("Consider walking distance between attractions")

  # Placeholder for map visualization
  st.empty()  # Replace with map visualization code upon data retrieval

  # Button to trigger calculations
  if st.button("Find shortest path"):
    if city_name:
      # Implement logic to retrieve attraction data and build the graph
      # based on city_name (replace with your actual implementation)
      # Update attractions list and create the graph here
      # ...

      # Find shortest path based on selected settings
      starting_point = attractions[random.randint(0, len(attractions) - 1)]
      shortest_path, total_distance, total_time = find_shortest_path(starting_point, travel_mode, consider_walking_distance)

      # Display results
      st.write(f"Starting point: {starting_point['name']}")
      st.write(f"Shortest path: {', '.join([a['name'] for a in shortest_path])}")
      st.write(f"Total distance: {total_distance}")
      st.write(f"Total travel time: {total_time}")
    else:
      st.error("Please enter a city name")

if __name__ == "__main__":
  main()