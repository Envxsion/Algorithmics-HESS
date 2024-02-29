from bokeh.plotting import figure, show
from bokeh.models import HoverTool, LabelSet, ColumnDataSource
from bokeh.palettes import Viridis256

# Define user data with profiles
users = {
    "Alice": "gay"
}

# Define connections and their strength
edges = [
    ("Alice", "Bob", 2),
    ("Alice", "Charlie", 4),
    ("Bob", "Charlie", 3),
    ("Bob", "Diana", 1),
    ("Charlie", "Diana", 2),
    ("Diana", "Eve", 3),
]

# Convert data to dictionary for Bokeh
data = {"users": users, "edges": edges}

# Create a ColumnDataSource for user data
source = ColumnDataSource(data)

# Define a figure for the plot
p = figure(
    title="Social Network with Connections, Interests, and Recommendations",
    x_range=[min(p.x_range.start, min(x for x, _ in source.data["source"].values())),
             max(p.x_range.end, max(x for x, _ in source.data["source"].values()))],
    y_range=[min(p.y_range.start, min(y for _, y in source.data["source"].values())),
             max(p.y_range.end, max(y for _, y in source.data["source"].values()))],
    tools="",  # Remove default zoom and pan tools
)

# Create circles for nodes with colors and sizes based on connections
p.circle(
    x="x",
    y="y",
    source=source,
    size="connections",
    fill_color="color",
    line_color="black",
    hover_fill_alpha=0.7,  # Reduce fill opacity on hover
)

# Label nodes with user names
labels = LabelSet(
    x="x",
    y="y",
    text="name",
    level="glyph",
    x_offset=5,
    y_offset=5,
    source=source,
    text_align="left",
    text_baseline="middle",
)
p.add_layout(labels)

# Draw edges
p.edge(
    x0="x",
    y0="y",
    x1="x",
    y1="y",
    source=source,
    line_width=0.5,
    line_color="gray",
    alpha=0.7,
)

# Hover tool to display user information
hover = HoverTool()

hover.tooltips = [
    ("Name", "@name"),
    ("Interests", "@interests"),
    ("Location", "@location"),
    ("Connections", "@connections"),
]

p.add_tools(hover)

# Display the interactive plot
show(p)
