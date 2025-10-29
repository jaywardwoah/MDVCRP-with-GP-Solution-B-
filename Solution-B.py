"""
This script solves the Multi-Depot Vehicle Routing Problem (MDVRP)
with Split Deliveries for the last-mile distribution of major vegetables.

This version is configured to find "Solution A":
The solution that strictly minimizes total distance (cost),
with the fairness_coefficient set to 0.
"""
import math
import random
import folium
import networkx as nx
import matplotlib.pyplot as plt
import graphviz
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp

VEHICLE_CAPACITY = 1215  # Capacity of one L300 truck in kg

def create_base_data_model():
    """Stores the original, unsplit data for the problem."""
    data = {}
    data['location_names'] = [
        'Divisoria (Depot)', 'Balintawak (Depot)', 'Muñoz Market',
        'Farmers Market', 'Marikina Public Market', 'Pasig Mega Market',
        'Pritil Public Market', 'Quinta Market', 'Pasay Public Market',
        'Guadalupe Public Market', 'Las Piñas Public Market','Alabang Public Market'
    ]
    data['locations'] = [
        (14.6155, 120.9780), (14.6565, 121.0125), (14.6542, 121.0158),
        (14.6214, 121.0544), (14.6300, 121.0968), (14.5772, 121.0854),
        (14.6291, 120.9722), (14.5960, 120.9774), (14.5498, 120.9954),
        (14.5786, 121.0583), (14.4504, 120.9882), (14.4239, 121.0428),
    ]
    # ---  DISTANCE MATRIX ---
    data['distance_matrix'] = [
        [0, 8900, 10400, 10400, 15900, 15500, 3100, 2500, 8300, 13000, 18900, 24000],
        [11400, 0, 2700, 7600, 13300, 16400, 9800, 10600, 18100, 14600, 28000, 31000],
        [11000, 2700, 0, 6100, 12000, 15000, 11000, 9300, 18000, 13000, 28000, 32000],
        [11000, 9400, 7000, 0, 6700, 10000, 11000, 9900, 14000, 7000, 25000, 27000],
        [15900, 14000, 12000, 5900, 0, 12000, 16000, 15000, 20000, 14000, 30000, 30000],
        [17000, 19000, 16000, 10000, 11000, 0, 17000, 16000, 13000, 6200, 24000, 21000],
        [3200, 7500, 10000, 10000, 15000, 18000, 0, 4000, 11000, 19000, 21000, 27000],
        [2600, 10100, 9100, 10100, 15000, 14000, 4300, 0, 7300, 10000, 18000, 25000],
        [8800, 21100, 18200, 14700, 18000, 13000, 11000, 6900, 0, 8100, 12000, 18000],
        [13900, 14800, 13100, 8100, 12600, 7900, 12500, 10600, 7900, 0, 18900, 19900],
        [20000, 26000, 29000, 25000, 28000, 23000, 23000, 19000, 13000, 17000, 0, 14000],
        [25300, 33500, 33300, 26300, 30500, 21500, 27200, 24800, 19000, 20400, 10700, 0],
    ]
    data['demands'] = [0, 0, 2309, 5609, 1769, 526, 4850, 2620, 3785, 7131, 3019, 2635]
    data['vehicle_capacity'] = VEHICLE_CAPACITY
    data['depots'] = [0, 1]
    return data

def create_split_delivery_data_model(base_data):
    """Creates the expanded data model for split deliveries."""
    new_location_names = base_data['location_names'][:2]
    new_demands = [0, 0]
    index_map = list(range(2))
    for i, demand in enumerate(base_data['demands'][2:], 2):
        original_location_name = base_data['location_names'][i]
        if demand > base_data['vehicle_capacity']:
            num_full_trucks = demand // base_data['vehicle_capacity']
            remainder = demand % base_data['vehicle_capacity']
            for j in range(num_full_trucks):
                new_location_names.append(f"{original_location_name} (Job {j+1})")
                new_demands.append(base_data['vehicle_capacity'])
                index_map.append(i)
            if remainder > 0:
                new_location_names.append(f"{original_location_name} (Job {num_full_trucks+1})")
                new_demands.append(remainder)
                index_map.append(i)
        elif demand > 0:
            new_location_names.append(original_location_name)
            new_demands.append(demand)
            index_map.append(i)
    num_new_locations = len(new_location_names)
    new_distance_matrix = [[0] * num_new_locations for _ in range(num_new_locations)]
    for i in range(num_new_locations):
        for j in range(num_new_locations):
            original_from_index = index_map[i]
            original_to_index = index_map[j]
            new_distance_matrix[i][j] = base_data['distance_matrix'][original_from_index][original_to_index]
    split_data = {}
    split_data['location_names'] = new_location_names
    split_data['distance_matrix'] = new_distance_matrix
    split_data['demands'] = new_demands
    total_demand = sum(new_demands)
    num_vehicles_needed = math.ceil(total_demand / base_data['vehicle_capacity'])
    split_data['num_vehicles'] = num_vehicles_needed
    split_data['vehicle_capacities'] = [base_data['vehicle_capacity']] * num_vehicles_needed
    split_data['depots'] = base_data['depots']
    return split_data

def print_solution(data, manager, routing, solution):
    """Prints a high-level summary of the optimal solution."""
    print("\n" + "="*80)
    print(" VRP Solution Summary")
    print("="*80)
    
    # Use the helper function to get all trip details
    all_trips = _get_all_trip_details(data, manager, routing, solution)
    num_vehicles_used = len(all_trips)
    
    # ### MODIFICATION ###
    # The *true* total distance must now be summed from the routes,
    # as the objective value includes the (span * coefficient) penalty.
    total_route_distance = sum(trip['distance'] for trip in all_trips) # This is in km
    
    # Find the workload span
    if all_trips:
        min_route = min(trip['distance'] for trip in all_trips)
        max_route = max(trip['distance'] for trip in all_trips)
        workload_span = max_route - min_route
    else:
        min_route, max_route, workload_span = 0, 0, 0
    # ### END MODIFICATION ###

    print(f'Objective Value (Cost Function): {solution.ObjectiveValue()}')
    print(f'Total Distance of All Routes: {total_route_distance:.2f} km')
    print(f'Total load delivered: {sum(data["demands"]):.0f} kg')
    print(f'Number of vehicles used: {num_vehicles_used}')
    print("-" * 80)
    print("Fairness Metrics (for Solution A):")
    print(f'  Longest Route: {max_route:.2f} km')
    print(f'  Shortest Route: {min_route:.2f} km')
    print(f'  Workload Span (Unfairness): {workload_span:.2f} km')
    print("="*80)

def _get_all_trip_details(data, manager, routing, solution):
    """Helper function to collect all trip details for tables/visualizations."""
    all_trips = []
    for vehicle_id in range(data['num_vehicles']):
        index = routing.Start(vehicle_id)
        if routing.IsEnd(solution.Value(routing.NextVar(index))):
            continue
        route_str_list, customer_nodes_in_trip = [], []
        route_distance, route_load = 0, 0
        temp_index = index
        while not routing.IsEnd(temp_index):
            from_node = manager.IndexToNode(temp_index)
            route_str_list.append(data['location_names'][from_node])
            if from_node not in data['depots']:
                customer_nodes_in_trip.append(from_node)
                route_load += data['demands'][from_node]
            previous_index = temp_index
            temp_index = solution.Value(routing.NextVar(temp_index))
            to_node = manager.IndexToNode(temp_index)
            route_distance += data['distance_matrix'][from_node][to_node]
        final_node_index = manager.IndexToNode(temp_index)
        route_str_list.append(data['location_names'][final_node_index])
        all_trips.append({'vehicle_id': vehicle_id, 'distance': route_distance / 1000, 'load': route_load,
                          'customer_nodes': customer_nodes_in_trip, 'route_str': ' -> '.join(route_str_list)})
    return all_trips

def print_trip_summary_table(data, manager, routing, solution):
    """Prints a detailed summary table of all trips, sorted by Vehicle ID."""
    print("\n" + "="*80)
    print(" VRP Detailed Trip Summary")
    print("="*80)
    all_trips = _get_all_trip_details(data, manager, routing, solution)
    print(f"{'Vehicle ID':<12} {'Load (kg)':<12} {'Distance (km)':<15} {'Route'}")
    print("-" * 120)
    for trip in sorted(all_trips, key=lambda x: x['vehicle_id']):
        print(f"{trip['vehicle_id']:<12} {trip['load']:<12} {trip['distance']:<15.2f} {trip['route_str']}")
    print("\n" + "="*80 + "\n")

def get_original_location_index(split_data, base_data, job_index):
    """Finds the original location index from the base_data corresponding to a job_index in the split_data."""
    job_name = split_data['location_names'][job_index]
    if "(Depot)" in job_name:
        return base_data['location_names'].index(job_name)
    if "(Job" not in job_name:
        try:
            return base_data['location_names'].index(job_name)
        except ValueError:
            return -1
    original_name = job_name.split(" (Job")[0]
    return base_data['location_names'].index(original_name)


def visualize_solution_on_map(base_data, data, manager, routing, solution):
    """Saves a visualization of the solution to an HTML file with distances and a legend."""
    print("Generating enhanced map visualization... 🗺️")
    map_center = [14.6091, 121.0223]
    m = folium.Map(location=map_center, zoom_start=12, tiles="cartodbpositron")
    
    # Add markers for depots and customers
    for i, loc in enumerate(base_data['locations']):
        name = base_data['location_names'][i]
        is_depot = i in base_data['depots']
        icon_type = 'truck' if is_depot else 'shopping-bag'
        color = 'darkred' if is_depot else 'blue'
        popup_html = f"<b>{name}</b>"
        if not is_depot:
            popup_html += f"<br>Demand: {base_data['demands'][i]} kg"
        folium.Marker(
            location=loc, 
            popup=popup_html, 
            icon=folium.Icon(color=color, icon=icon_type, prefix='fa')
        ).add_to(m)

    # Add route lines with distances in popups
    all_trips = _get_all_trip_details(data, manager, routing, solution)
    for trip in all_trips:
        route_color = "#{:06x}".format(random.randint(0, 0xFFFFFF))
        route_points = []
        for name in trip['route_str'].split(' -> '):
            try:
                job_index = data['location_names'].index(name)
                original_loc_index = get_original_location_index(data, base_data, job_index)
                if original_loc_index != -1:
                    route_points.append(base_data['locations'][original_loc_index])
            except ValueError:
                continue
        popup_html = f"<b>Vehicle {trip['vehicle_id']}</b><br>Distance: {trip['distance']:.2f} km"
        folium.PolyLine(
            locations=route_points, 
            color=route_color, 
            weight=4, 
            opacity=0.8, 
            popup=popup_html
        ).add_to(m)

    # Add an HTML legend to the map
    legend_html = """
    <div style="position: fixed; 
                bottom: 50px; left: 50px; width: 150px; height: 90px; 
                border:2px solid grey; z-index:9999; font-size:14px;
                background-color:white; padding: 5px;">
        &nbsp; <b>Legend</b> <br>
        &nbsp; <i class="fa fa-truck" style="color:darkred"></i> &nbsp; Depot <br>
        &nbsp; <i class="fa fa-shopping-bag" style="color:blue"></i> &nbsp; Market <br>
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))

    map_filename = "vrp_routes_map.html"
    m.save(map_filename)
    print(f"...SUCCESS: Enhanced map saved to '{map_filename}'.")

def visualize_solution_with_graphviz(base_data, data, manager, routing, solution):
    """
    Creates a high-quality, simple visualization using Graphviz.
    """
    print("Generating final Graphviz network visualization... 🎨")
    dot = graphviz.Digraph(comment='VRP Solution', engine='neato')
    dot.attr('graph',
             label='VRP Solution Network Model',
             labelloc='t',
             fontsize='20',
             splines='true',
             overlap='false',
             bgcolor='white'
            )
    dot.attr('node', shape='circle', style='filled', fontname='Helvetica', fontsize='10')
    dot.attr('edge', fontsize='8', fontname='Helvetica', color='black')

    nodes_in_solution = set()
    all_trips = _get_all_trip_details(data, manager, routing, solution)
    for trip in all_trips:
        for node in trip['customer_nodes']:
            nodes_in_solution.add(node)
    for depot in data['depots']:
        nodes_in_solution.add(depot)

    market_colors = {}
    color_palette = plt.colormaps.get('tab10')

    for node_idx in nodes_in_solution:
        if node_idx in data['depots']:
            original_name = base_data['location_names'][node_idx]
            dot.node(str(node_idx), original_name, shape='doublecircle', fillcolor='orangered', fontcolor='white')
        else:
            original_loc_index = get_original_location_index(data, base_data, node_idx)
            market_name = base_data['location_names'][original_loc_index]
            if market_name not in market_colors:
                market_colors[market_name] = color_palette(len(market_colors) % 10)

            rgba_color = market_colors[market_name]
            hex_color = '#{:02x}{:02x}{:02x}'.format(int(rgba_color[0]*255), int(rgba_color[1]*255), int(rgba_color[2]*255))
            dot.node(str(node_idx), str(node_idx), fillcolor=hex_color, fontcolor='white')

    G = nx.DiGraph([ (data['location_names'].index(u), data['location_names'].index(v))
                     for trip in all_trips for u, v in zip(trip['route_str'].split(' -> ')[:-1], trip['route_str'].split(' -> ')[1:]) ])

    processed_round_trips = set()
    for u, v in G.edges():
        edge_pair = tuple(sorted((u, v)))
        if G.has_edge(v, u) and edge_pair not in processed_round_trips:
            total_dist = (data['distance_matrix'][u][v] + data['distance_matrix'][v][u]) / 1000
            dot.edge(str(u), str(v), label=f'{total_dist:.1f}', dir='both')
            processed_round_trips.add(edge_pair)
        elif not G.has_edge(v, u):
            dist = data['distance_matrix'][u][v] / 1000
            dot.edge(str(u), str(v), label=f'{dist:.1f}')

    legend_html = '<<TABLE BORDER="0" CELLBORDER="1" CELLSPACING="0" BGCOLOR="white">'
    legend_html += '<TR><TD COLSPAN="2"><B>Legend (Distances in km)</B></TD></TR>'
    legend_html += '<TR><TD BGCOLOR="orangered"> </TD><TD ALIGN="LEFT">Depot</TD></TR>'
    for market_name, rgba_color in sorted(market_colors.items()):
        hex_color = '#{:02x}{:02x}{:02x}'.format(int(rgba_color[0]*255), int(rgba_color[1]*255), int(rgba_color[2]*255))
        legend_html += f'<TR><TD BGCOLOR="{hex_color}"> </TD><TD ALIGN="LEFT">{market_name}</TD></TR>'
    legend_html += '</TABLE>>'

    dot.node('legend', label=legend_html, shape='plaintext')

    try:
        dot.render('vrp_graphviz_solution', format='png', view=False, cleanup=True)
        print("...SUCCESS: Graphviz visualization saved to 'vrp_graphviz_solution.png'.")
    except graphviz.backend.execute.ExecutableNotFound:
        print("\n--- GRAPHVIZ ERROR: Graphviz executable not found. Skipping visualization. ---")

def visualize_market_summary_table(base_data, data, manager, routing, solution):
    """Creates a Matplotlib figure displaying the trip summary table with adjusted columns."""
    print("Generating visualized trip summary table... 📈")
    all_trips = _get_all_trip_details(data, manager, routing, solution)
    headers = ["Vehicle ID", "Destination(s)", "Load (kg)", "Distance (km)", "Route"]
    table_data = []
    for trip in sorted(all_trips, key=lambda x: x['vehicle_id']):
        original_markets = {base_data['location_names'][get_original_location_index(data, base_data, node_idx)] for node_idx in trip['customer_nodes']}
        if len(original_markets) == 1:
            destination = list(original_markets)[0]
        elif len(original_markets) > 1:
            destination = "Multi-Stop"
        else:
            destination = "N/A"
        table_data.append([f"{trip['vehicle_id']}", destination, f"{trip['load']}", f"{trip['distance']:.2f}", trip['route_str']])
    fig, ax = plt.subplots(figsize=(16, 12))
    fig.canvas.manager.set_window_title('Trip Summary Table')
    ax.axis('off')
    col_widths = [0.07, 0.12, 0.07, 0.09, 0.65]
    table = ax.table(cellText=table_data, colLabels=headers, colWidths=col_widths, loc='center', cellLoc='left')
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_fontsize(10)
            cell.set_facecolor("#DCDCDC")
            cell.set_text_props(weight='bold')
        cell.set_height(0.04)
    fig.suptitle('VRP Trip Summary', fontsize=16)
    plt.tight_layout()
    plt.savefig("vrp_trip_summary_table.png")
    print("...SUCCESS: Visualized trip summary table saved to 'vrp_trip_summary_table.png'.")
    plt.close(fig)

def create_kpi_charts(base_data, data, manager, routing, solution, starts):
    """Creates and saves KPI charts for the VRP solution."""
    print("Generating KPI charts... 📈")
    all_trips = _get_all_trip_details(data, manager, routing, solution)
    used_vehicle_ids = [trip['vehicle_id'] for trip in all_trips]
    vehicle_distances = [trip['distance'] for trip in all_trips]
    vehicle_loads = [trip['load'] for trip in all_trips]
    depot_loads = {depot_id: 0 for depot_id in base_data['depots']}
    for trip in all_trips:
        start_depot_node = starts[trip['vehicle_id']]
        depot_loads[start_depot_node] += trip['load']
    fig, axs = plt.subplots(2, 2, figsize=(16, 12), constrained_layout=True)
    fig.canvas.manager.set_window_title('KPIs')
    fig.suptitle('VRP Solution KPIs', fontsize=20)
    axs[0, 0].bar(range(len(used_vehicle_ids)), vehicle_distances, color='cyan', edgecolor='black')
    axs[0, 0].set_title('Distance per Vehicle')
    axs[0, 0].set_ylabel('Distance (km)')
    axs[0, 0].set_xlabel('Vehicle ID')
    axs[0, 0].set_xticks(range(len(used_vehicle_ids)))
    axs[0, 0].set_xticklabels(used_vehicle_ids, rotation=45, ha="right")
    axs[0, 1].bar(range(len(used_vehicle_ids)), vehicle_loads, color='orange', edgecolor='black')
    axs[0, 1].axhline(y=VEHICLE_CAPACITY, color='r', linestyle='--', label=f'Max Capacity ({VEHICLE_CAPACITY} kg)')
    axs[0, 1].set_title('Load per Vehicle')
    axs[0, 1].set_ylabel('Load (kg)')
    axs[0, 1].set_xlabel('Vehicle ID')
    axs[0, 1].set_xticks(range(len(used_vehicle_ids)))
    axs[0, 1].set_xticklabels(used_vehicle_ids, rotation=45, ha="right")
    axs[0, 1].legend()
    depot_names = [base_data['location_names'][i] for i in depot_loads.keys()]
    loads = list(depot_loads.values())
    axs[1, 0].pie(loads, labels=depot_names, autopct='%1.1f%%', startangle=90, colors=['lightblue', 'lightgreen'])
    axs[1, 0].set_title('Demand Fulfilled by Depot')
    axs[1, 1].axis('off')
    plt.savefig("vrp_kpi_charts.png")
    print("...SUCCESS: KPI charts saved to 'vrp_kpi_charts.png'.")
    plt.close(fig)

def create_demand_bar_graph(base_data):
    """Creates a Matplotlib bar graph of the original market demands."""
    print("Generating demand bar graph... 📊")
    market_names = sorted([name for name in base_data['location_names'] if "(Depot)" not in name])
    demand_map = {name: demand for name, demand in zip(base_data['location_names'], base_data['demands'])}
    market_demands = [demand_map[name] for name in market_names]
    fig, ax = plt.subplots(figsize=(12, 8))
    fig.canvas.manager.set_window_title('Market Demands')
    ax.bar(market_names, market_demands, color='skyblue', edgecolor='black')
    ax.set_ylabel('Demand (kg)')
    ax.set_title('Total Demand per Market')
    ax.tick_params(axis='x', rotation=45)
    plt.tight_layout()
    plt.savefig("vrp_demand_bar_graph.png")
    print("...SUCCESS: Demand bar graph saved to 'vrp_demand_bar_graph.png'.")
    plt.close(fig)

def visualize_solution_as_scatter(base_data, data, manager, routing, solution, starts):
    """Creates a new, enhanced visualization of the solution as a scatter plot."""
    print("Generating enhanced scatter plot visualization... 📈")
    fig, ax = plt.subplots(figsize=(12, 12))
    fig.canvas.manager.set_window_title('Scatter Plot (Enhanced)')
    cmap = plt.colormaps.get('tab10')
    market_colors, depot_handle, market_handles = {}, None, []
    for i, loc in enumerate(base_data['locations']):
        lon, lat = loc[1], loc[0]
        if i in base_data['depots']:
            handle = ax.scatter(lon, lat, c='red', marker='s', s=200, zorder=10)
            if depot_handle is None: depot_handle = handle
            ax.text(lon + 0.001, lat, f' {base_data["location_names"][i]}', fontsize=9, va='center')
        else:
            market_name = base_data['location_names'][i]
            color_index = i - 2
            color = cmap(color_index / 10.0)
            if market_name not in market_colors:
                market_colors[market_name] = color
                handle = ax.scatter(lon, lat, c=[color], s=100, zorder=10)
                market_handles.append((market_name, handle))
            else:
                ax.scatter(lon, lat, c=[market_colors[market_name]], s=100, zorder=10)
    all_trips = _get_all_trip_details(data, manager, routing, solution)
    depot_route_colors = ['#0000FF', '#008000']
    for trip in all_trips:
        start_node = starts[trip['vehicle_id']]
        color = depot_route_colors[start_node % len(depot_route_colors)]
        route_points = []
        for name in trip['route_str'].split(' -> '):
            job_index = data['location_names'].index(name)
            original_loc_index = get_original_location_index(data, base_data, job_index)
            if original_loc_index != -1:
                route_points.append(base_data['locations'][original_loc_index])
        for i in range(len(route_points) - 1):
            start_point, end_point = route_points[i], route_points[i+1]
            ax.arrow(start_point[1], start_point[0], end_point[1] - start_point[1], end_point[0] - start_point[0],
                     color=color, length_includes_head=True, head_width=0.0015, head_length=0.002, alpha=0.6, zorder=5)
    ax.set_title('VRP Solution - Scatter Plot with Routes')
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    legend_handles_sorted = sorted(market_handles, key=lambda x: x[0])
    legend_labels = ["Depot"] + [mh[0] for mh in legend_handles_sorted]
    legend_handles = [depot_handle] + [mh[1] for mh in legend_handles_sorted]
    ax.legend(legend_handles, legend_labels, title="Locations", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True)
    plt.tight_layout(rect=[0, 0, 0.85, 1])
    plt.savefig("vrp_scatter_plot_enhanced.png")
    print("...SUCCESS: Enhanced scatter plot saved to 'vrp_scatter_plot_enhanced.png'.")
    plt.close(fig)

def main():
    """The main entry point for the VRP solver script."""
    base_data = create_base_data_model()
    data = create_split_delivery_data_model(base_data)

    print("--- STARTING DATA VALIDATION ---")
    print(f"Number of locations (from distance matrix): {len(data['distance_matrix'])}")
    print(f"Number of demands: {len(data['demands'])}")
    print(f"Number of vehicles: {data['num_vehicles']}")
    print(f"Number of vehicle capacities: {len(data['vehicle_capacities'])}")

    assert len(data['distance_matrix']) == len(data['demands']), "FATAL: Mismatch!"
    assert data['num_vehicles'] == len(data['vehicle_capacities']), "FATAL: Mismatch!"
    print("--- DATA VALIDATION PASSED ---\n")

    try:
        print("1. Creating RoutingIndexManager...")

        num_vehicles = data['num_vehicles']
        num_depots = len(data['depots'])
        starts = [data['depots'][i % num_depots] for i in range(num_vehicles)]
        ends = [data['depots'][i % num_depots] for i in range(num_vehicles)]

        manager = pywrapcp.RoutingIndexManager(len(data['distance_matrix']), num_vehicles, starts, ends)
        print("...SUCCESS: RoutingIndexManager created.\n")

        print("2. Creating RoutingModel...")
        routing = pywrapcp.RoutingModel(manager)
        print("...SUCCESS: RoutingModel created.\n")

        print("3. Registering transit (distance) callback...")
        def distance_callback(from_index, to_index):
            from_node = manager.IndexToNode(from_index)
            to_node = manager.IndexToNode(to_index)
            return data['distance_matrix'][from_node][to_node]

        transit_callback_index = routing.RegisterTransitCallback(distance_callback)
        print("...SUCCESS: Transit callback registered.\n")

        print("4. Setting arc cost (This is Goal 1: Minimize Total Cost)...")
        routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)
        print("...SUCCESS: Arc cost set.\n")

        # ### MODIFICATION 1 (FOR GOAL PROGRAMMING) ###
        print("4a. Adding Distance dimension to track route lengths...")
        # We must add a dimension to track the cumulative distance of each
        # route, otherwise, we can't calculate the "span" (max - min).
        routing.AddDimension(
            transit_callback_index,
            0,          # no slack
            9999999,    # vehicle maximum travel distance (a large number)
            True,       # start cumul to zero
            'Distance'
        )
        distance_dimension = routing.GetDimensionOrDie('Distance')
        print("...SUCCESS: Distance dimension added.\n")

        print("4b. Setting Goal 2: Minimize 'Unfairness' (Workload Span)...")
        # This coefficient is the 'weight' for the Goal Programming.
        
        # --- FOR SOLUTION A (Min Cost) ---
        # purely cost-based solution.
        fairness_coefficient = 1000
        
        distance_dimension.SetGlobalSpanCostCoefficient(fairness_coefficient)
        print(f"...SUCCESS: Set global span cost coefficient (fairness weight) to {fairness_coefficient}.\n")
        # ### END MODIFICATION ###


        print("5. Adding a fixed cost to each vehicle to minimize fleet size...")
        routing.SetFixedCostOfAllVehicles(100000)
        print("...SUCCESS: Vehicle cost set.\n")

        print("6. Registering demand callback...")
        def demand_callback(from_index):
            from_node = manager.IndexToNode(from_index)
            return data['demands'][from_node]

        demand_callback_index = routing.RegisterUnaryTransitCallback(demand_callback)
        print("...SUCCESS: Demand callback registered.\n")

        print("7. Adding capacity dimension...")
        routing.AddDimensionWithVehicleCapacity(demand_callback_index, 0, data['vehicle_capacities'], True, 'Capacity')
        print("...SUCCESS: Capacity dimension added.\n")

        print("8. Setting search parameters...")
        search_parameters = pywrapcp.DefaultRoutingSearchParameters()
        search_parameters.first_solution_strategy = (
            routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC)
        search_parameters.local_search_metaheuristic = (
            routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH)
        search_parameters.time_limit.FromSeconds(30)
        print("...SUCCESS: Search parameters set.\n")

        print('Solving the VRP with Split Deliveries...\n')
        solution = routing.SolveWithParameters(search_parameters)
        print("...SUCCESS: Solver finished.\n")

        if solution:
            # --- Text Outputs ---
            print_solution(data, manager, routing, solution)
            print_trip_summary_table(data, manager, routing, solution)

            # --- File-based Visualizations ---
            visualize_solution_on_map(base_data, data, manager, routing, solution)
            visualize_solution_with_graphviz(base_data, data, manager, routing, solution)

            # --- Additional Matplotlib Visualizations ---
            print("\n--- Generating Additional Matplotlib Visualizations ---")
            visualize_market_summary_table(base_data, data, manager, routing, solution)
            create_kpi_charts(base_data, data, manager, routing, solution, starts)
            create_demand_bar_graph(base_data)
            visualize_solution_as_scatter(base_data, data, manager, routing, solution, starts)
            
            # If you want to see the plots pop up after they are saved, uncomment the line below
            # plt.show()
            print("\n--- All visualizations have been saved as files. ---")

        else:
            print('No solution found !')

    except Exception as e:
        print(f"A PYTHON EXCEPTION OCCURRED: {e}")


if __name__ == '__main__':
    main()