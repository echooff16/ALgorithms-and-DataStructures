"""
Assignment
Question 1:
"""
import heapq


def dijkstra(L, adj, start_location):
    """
    Perform Dijkstra's algorithm from the start node with edge relaxation.

    Time Complexity: O(|L| + |R| log |L|)
        - |L| = number of locations (nodes)
        - |R| = number of roads (undirected edges; counted as two directed edges in the adjacency list)

    Detailed Analysis:
        1) Initialisation: O(|L|)
           - Build the distance array of size |L|; set source distance to 0.

        2) Priority queue setup: O(1)
           - Heap initialised with a single (0, start_location) entry.

        3) Main loop (heap pops/pushes): O(|R| log |L|)
           - Dijkstra pushes a new heap entry on every successful relaxation
             (no inplace decrease key). Over the whole run there are <= |R| pushes,
             hence <= |R| pops. Each heap operation costs O(log |L|).

        4) Edge scans (relaxations): O(|R|)
           - Each directed adjacency entry is examined once overall; arithmetic and
             comparisons per check are O(1). This term is dominated by heap costs.

    Space: O(|L|) (dist array) + up to O(|R|) transient heap entries
        - Summarised as O(|L| + |R|) extra space beyond the input graph.

    Args:
        L (int): Number of locations (nodes), IDs in [0, L-1].
        adj (list[list[tuple[int, int]]]): Adjacency list; for each u, adj[u] is a list of
            (v, w) pairs where v is a neighbour and w > 0 is the road length.
        start_location (int): Source node ID.

    Returns:
        list[float]: min_distance where min_distance[l] is the shortest distance
            from start_location to l (float('inf') if unreachable).

    Notes:
         This implementation does not perform in-place decrease-key on the heap. When a shorter
         distance to v is found, it pushes a new (dist, v) entry; later, if an older entry for v is
         popped, it's identified as stale (its distance > dist[v]) and skipped. With at most one
         successful relaxation per edge, the total number of heap pushes/pops is O(|R|), each
         costing O(log |L|), plus O(|L|) for initialisation. Hence, the running time is
         O(|L| + |R| log |L|)  for a decrease-key implementation.
    """

    # From our test case [Test case 1 from brief] (Refer to assign())
    # L = 16
    # adj = city_adj_list populated from preprocess_distances():
    # start_loc = 0, 6, 15 (per iteration)

    # Priority Queue: stores tuples of (distance, location_id)
    pq = [(0,
           start_location)]  # Priority queue initialized with start node: Time Comp: O(1) # pq[(0,6)] for first instance from our case

    # Array to store the minimum distance from start_location
    min_distance = []

    for _ in range(L):  # O(L) # till 16
        min_distance.append(float('inf'))  # [ inf, inf, inf.....]

    min_distance[start_location] = 0

    while pq:  # Process nodes in order of current best distance: O(R)

        # initialisation (here pickup loc 0 is being demonstrated)
        # pq = [(0, 0)]   # start node 0 at distance 0
        # min_distance = [0, inf, inf, inf, inf, inf, inf, inf,
        #                 inf, inf, inf, inf, inf, inf, inf, inf]   # every other node unreachable yet
        # pop the start node
        # current_dist, current_location = heapq.heappop(pq)       # pop (0, 0)
        # pq is now []                                            # heap empty after popping source
        # scan neighbour of node 0
        # adj[0] = [(1,3), (2,5), (3,10), (8,1), (9,1), (10,1), (11,1)]  # all roads from 0

        # Get the location with the smallest known distance
        current_dist, current_location = heapq.heappop(
            pq)  # Pop the node with the smallest distance: Time Comp: O(log |L|)

        # Skip if this is not the shortest path found: Time Comp: O(1)
        if current_dist != min_distance[current_location]:
            continue

        # Iterate over neighbors
        # adj[current_location] is a list of (neighbor_location, road_length)

        # Here: iterates over (1,3),(2,5),(3,10),(8,1),(9,1),(10,1),(11,1)
        for road_edge in adj[current_location]:  # Iterate over adjacent nodes (edges): Time Comp: O(|R|)

            neighbor_location, road_length = road_edge  # 1st pass: neighbor=1, w=3
            # new_dist = 0 + 3 = 3
            new_dist = min_distance[
                           current_location] + road_length  # Calculate new possible shortest distance: Time Comp: O(1)

            # Relaxation step: if a shorter path to the neighbor is found
            # 3 < inf -> True (relax)
            if new_dist < min_distance[neighbor_location]:  # Check if the new distance is shorter: Time Comp: O(1)
                # min_distance[1] = 3
                min_distance[neighbor_location] = new_dist  # Update distance to the shorter one: Time Comp: O(1)
                # Add/update the neighbor in the priority queue
                # push(3,1) and so on....
                heapq.heappush(pq, (
                new_dist, neighbor_location))  # Push the updated distance and node: Time Comp: O(log |L|)

    return min_distance  # Returns the distance: O(1)


def dfs(graph, u, sink, flow, parent, visited, numNodes):
    """
    Depth-First Search (DFS) on a residual network to find an augmenting path and
    return its bottleneck capacity.

    From the current node 'u', the function follows residual edges toward 'sink', only using
    edges with positive remaining capacity and skipping nodes already visited. If it reaches
    'sink', it returns the smallest capacity seen along that path (the bottleneck). As it
    returns from recursion, it stores who led to each node and which edge was used, so the
    caller can later update the residual capacities along the found path.

    Approach:
    - The function operates recursively. The base case for the recursion is when the current node 'u' is the 'sink'.
    - It uses a 'visited' list to keep track of nodes visited in the current DFS traversal, preventing cycles and redundant exploration.
    - For each neighbor 'v' of the current node 'u', it checks two conditions:
      1. Has 'v' not been visited in this search ('not visited[v]')
      2. Is there available capacity on the edge from 'u' to 'v' ('cap > 0')
    - If both are true, it recursively calls 'dfs' on the neighbor 'v'. The 'flow' passed to the recursion is the minimum of the current path's
     bottleneck and the capacity of the new edge ('u', 'v').
    - If a recursive call successfully finds a path to the sink (returns 'new_flow > 0'), it means an augmenting path has been found.
    - On the way back up the call stack (after a successful path is found), it populates the 'parent' array. 'parent[v] = u' records the predecessor, and
    'parent[numNodes + v] = i' records which edge was taken. This information is essential for the calling function ('ford_fulkerson') to backtrack and update the flow.
    - If the loop finishes without finding a path from 'u' to the sink, it returns 0.

    Detailed Time Complexity Analysis:
    1. Base Case:
       - The check 'if u == sink:' is a simple comparison, taking O(1) time.

    2. Node Visitation:
       - 'visited[u] = True' is an array assignment, which is O(1).

    3. Loop through Neighbors:
       - The for loop iterates through all outgoing edges from node 'u'. The number of iterations is equal
         to the degree of 'u'.
       - Inside the loop, all checks ('not visited[v]', 'cap > 0') and calculations ('min(flow, cap)') are O(1) per edge.

    4. Recursive Call:
       - The function calls 'dfs' on its neighbors. Because of the 'visited' array, each node and edge in the graph will be visited at most once during a
       single top-level call to 'dfs'.

    5. Path Recording:
       - If a path is found ('new_flow > 0'), the assignments to the 'parent' array are O(1) operations.

    Combined Time Complexity:
    - A single, top-level call to this 'dfs' function performs a standard Depth-First Search. It explores each node (vertex, V) and
      each edge (E) at most one time.
    - Therefore, the generic time complexity for one complete search is O(V + E).
    - In the context of the problem:
      - The number of nodes 'V' (or 'numNodes') is O(S + B + B_u), where S is students, B is buses, and B_u is unique locations.
        Since B_u is a constant and B <= S, this simplifies to O(S).
      - The number of edges 'E' is O(S*B_u + B). Since B_u is a constant and B <= S, this also simplifies to O(S).
    - So, for our specific problem, the time complexity of a single 'dfs' run is O(S + S), which simplifies to O(S).

    Detailed Space Complexity Analysis:
    1. Input Graph:
       - The residual network (graph) is stored as an adjacency list.
         It occupies Θ(V + E) space, where V is the number of nodes and E is the number of edges.

    2. Recursive Call Stack:
       - In the worst case (a linear chain of nodes), the recursion depth equals 'numNodes'.
         This adds O(V) space on the call stack.

    3. 'parent' Array:
       - Allocated once by the calling function with size 2 * numNodes (O(V)).
         Passed by reference to this function, so it does not add extra space per call.

    4. 'visited' Array:
       - Also allocated once by the caller with size O(V) and passed by reference.

    5. Local Variables:
       - Variables such as 'i', 'v', 'cap', and 'new_flow' consume O(1) space per recursive call.

    Explanation of Space Complexity:
        - The total space complexity is dominated by the residual graph, which requires Θ(V + E).
        - The algorithm itself adds O(V) working memory due to recursion depth and supporting arrays.
        - Hence, total space complexity = Θ(V + E), with O(V) additional working space.
        - In the context of the main 'assign' problem, where V and E are proportional to S,
          the total space complexity is Θ(S), with O(S) additional working memory.

    Args:
        graph (list of tuples): The residual graph, represented as an adjacency list. For each node 'u', 'graph[u]' contains tuples '(v, capacity, reverse_idx)'
        for each outgoing edge.
        u (int): The current node being explored in the DFS.
        sink (int): The target (sink) node of the flow network.
        flow (float): The bottleneck capacity found on the path from the source to node 'u'. Initially called with 'float('inf')'.
        parent (list): An array of size '2 * numNodes' used to reconstruct the path after it's found. It is populated by this function on the recursive return.
        visited (list): A boolean array of size 'numNodes' to track visited nodes for the current DFS search. It is reset by the caller for each new path search.
        numNodes (int): The total number of nodes (vertices) in the graph.

    Returns:
        float: The bottleneck capacity of the augmenting path found, or 0 if no path exists from 'u' to 'sink'.
    """

    # Tests whether the current node is the sink; if true, a path is completed.
    if u == sink:
        return flow

    # Marks the current node as visited to prevent revisiting in this DFS call tree.
    visited[u] = True
    # Explore neighbors of u using DFS
    for i in range(len(graph[u])):
        # Unpack the edge tuple
        v, cap, _ = graph[u][i]

        # Check if the neighbor is unvisited and has residual capacity
        if not visited[v] and cap > 0:
            bottleneck_limit = min(flow, cap)

            # Recursively find the bottleneck capacity on the remaining path
            new_flow = dfs(graph, v, sink, bottleneck_limit, parent, visited, numNodes)

            # If flow was successfully pushed through the rest of the path
            if new_flow > 0:
                # Record the path for flow update (predecessor and edge index)
                parent[v] = u
                parent[numNodes + v] = i
                return new_flow

    # If no augmenting path is found, return 0
    return 0


def ford_fulkerson(graph, source, sink, T_limit=float('inf')):
    """
    Computes the maximum flow from a source to a sink using the Ford-Fulkerson method.

    This function repeatedly finds augmenting paths (paths from source to sink with
    available capacity) in the residual graph using a Depth-First Search (DFS). For each
    path found, it pushes the bottleneck capacity of that path, updates the total flow,
    and modifies the residual graph. The process continues until no more augmenting
    paths can be found or an optional total flow limit (T_limit) is reached.

    Approach:
    - The algorithm operates in a loop that continues as long as augmenting paths exist.
    - In each iteration, it initializes a 'visited' array for the DFS and calls the 'dfs'
      helper function to find one augmenting path.
    - The 'dfs' function returns the bottleneck capacity of the path it found ('path_flow').
      If it returns 0, no path was found, and the main loop terminates.
    - The 'path_flow' is potentially reduced to not exceed the optional 'T_limit'.
    - The 'max_flow' is increased by this 'path_flow'.
    - The function then backtracks along the path found by DFS (using the 'parent'
      array) to update the residual capacities of the edges. It decreases the
      capacity of forward edges and increases the capacity of backward (residual) edges.
    - Once the loop terminates, the accumulated 'max_flow' is returned.
     Detailed Time Complexity Analysis:
    1. Initialization:
       - 'numNodes = len(graph)': O(1).
       - The loop to initialize the 'parent' array runs '2 * numNodes' times. This takes O(V) time, where V is numNodes which comes down to O(S).
       - 'max_flow = 0': O(1).

    2. Main Augmentation Loop ('while max_flow < T_limit'):
       - The number of iterations of this loop is determined by the number of augmentations.
       - For integer capacities, each augmenting path increases the total flow by at least 1.
       - Therefore, the loop runs at most F times, where F is the value of the maximum flow (or T_limit).

    3. Work Inside the Loop:
       - 'visited = [False] * numNodes': Takes O(V) time to create the list.
       - 'path_flow = dfs(...)': The DFS search for a path takes O(V + E) time, where E is the number of edges.
       - 'path_flow = min(...)', 'max_flow += ...': These are O(1) operations.
       - Backtracking loop ('while v != source'): This loop traverses the found path, which has at most V-1 edges. The work inside is O(1).
         So, this step takes O(V) time.
       - The dominant work in each iteration is the DFS call, making each iteration O(V + E).

    Combined Time Complexity:
    - The total time complexity is the product of the number of augmentations and the cost of finding each augmentation:
      Total Time = (Number of Augmentations) * (Cost per Augmentation).
    - The "Number of Augmentations" is bounded by the total flow, F. With integer capacities where each augmentation is at least 1, this is O(F).
    - The "Cost per Augmentation" is the time to find one path, which is dominated by the DFS call, O(V + E).
    - This gives a general time complexity of O(F * (V + E)).
    - In the context of our 'assign' method:
      - The total flow 'F' is the target number of students, T.
      - The number of vertices 'V' and edges 'E' in our specific flow graph are both proportional to the number of students, S. (detailed explanation is provided in dfs())
        So, the cost of one 'dfs' run becomes O(S).
      - Plugging these in gives: Total Time = O(T) * O(S), which results in O(ST).

    Detailed Space Complexity Analysis:
    1. 'parent' Array:
       - This array is created once with a size of '2 * numNodes'. It requires O(V) space.
    2. 'visited' Array:
       - This array is created in each iteration of the main loop with a size of 'numNodes'. It requires O(V) space.
    3. 'dfs' Call Stack:
       - The recursive 'dfs' function can have a recursion depth of up to 'numNodes' in the worst case (a single long path), requiring O(V + E),
         O(S) for our case, space on the call stack.
    4. Other Variables:
       - Variables like 'max_flow', 'path_flow', 'u', 'v' take up a constant, O(1), amount of space.

    Space Complexity: O(S)

    Args:
        graph (list of tuples): The flow network, represented as an adjacency list.
        source (int): The index of the source node.
        sink (int): The index of the sink node.
        T_limit (float): The maximum total flow to find. The algorithm will
            stop once this flow is reached. Defaults to infinity, which finds the
            absolute maximum flow.

    Returns:
        max_flow: The calculated maximum flow, which will be at most T_limit.

    """

    numNodes = len(graph)  # O(1)  ;  = 36  (nodes 0..35)

    # parent array: parent[v]=u, parent[num_nodes+v]=edge_idx
    parent = []

    # = 72 entries
    for _ in range(
            2 * numNodes):  # O(numNodes) ~ O(S + B + B_u + 2 source + 2 sink) B_u are the pickup locations which is <=18, B<=S; O(S)
        parent.append(0)  # O(1)

    max_flow = 0

    # 1. If T_limit is infinity, it runs until path_flow is 0 (native max-flow).
    # 2. If T_limit is a number, it stops when max_flow reaches that number.

    # Complexity of the whole block: O(F * (V + E)) where F is T and the max flow and O(V+E) from DFS is O(S); O(ST)
    while max_flow < T_limit:
        visited = [False] * numNodes
        path_flow = dfs(graph, source, sink, float('inf'), parent, visited,
                        numNodes)  # O(S); check dfs() for complexity

        # Here: dfs(...) tries to find a path 34->33->0->(student node)->(location node)->(bus node)->35

        if path_flow == 0:
            break

        # Restrict the path flow if adding the full path_flow would exceed T_limit
        path_flow = min(path_flow, T_limit - max_flow)  # min(1, 18 - 0) = 1

        max_flow = max_flow + path_flow  # 0 + 1 = 1
        v = sink  # v = 35  (super_sink)

        # Update residual capacities along the path using the parent array
        while v != source:  # O(V) ~ O(S) ;  # 35 != 34 -> True
            u = parent[v]  # parent[35] = 29
            # Retrieve edge index from the second half of the parent array
            edge_idx_forward = parent[numNodes + v]  # parent[36 + 35] = index of (29 -> 35)

            # Retrieve the reverse edge index
            edge_idx_backward = graph[u][edge_idx_forward][2]  # reverse index of (35 -> 29)

            # Decrease capacity on forward edge
            fwd_data = graph[u][edge_idx_forward]  # fwd_data = (35, cap=3, rev_idx) for (29 -> 35)
            graph[u][edge_idx_forward] = (fwd_data[0], fwd_data[1] - path_flow, fwd_data[2])  # (29 -> 35): 3 -> 2

            # Increase capacity on backward edge (residual edge)
            bwd_data = graph[v][edge_idx_backward]  # bwd_data = (29, cap=0, rev_idx) for (35 -> 29)
            graph[v][edge_idx_backward] = (bwd_data[0], bwd_data[1] + path_flow, bwd_data[2])  # (35 -> 29): 0 -> 1
            v = u  # v = 29
            # loop ends when v == source (34)

    return max_flow  # O(1)


def preprocess_distances(L, roads, buses):
    """
    Runs Dijkstra's algorithm from each unique bus pickup location to pre-compute all
    necessary shortest paths up to a distance D. This step does the expensive pathfinding work once,
    allowing for very fast distance lookups later.

    Approach:
    1. The city's road network is built into an adjacency list representation from the
       input 'roads' list to allow for efficient graph traversal.
    2. To avoid redundant computations, the function identifies the set of 'unique' pickup
       locations (B_u). If multiple buses are at the same location, Dijkstra's only needs to
       be run from that location once. A mapping array is created to link raw location
       IDs to a compact index for these unique locations.
    3. It iterates through only the unique pickup locations and runs an optimized
       Dijkstra's algorithm from each one. The results (a list of distances from that
       pickup to all other locations) are stored in a list of lists.
    4. It defines and returns a nested helper function, 'get_shortest_dist', which
       performs a fast, O(1) lookup into the pre-computed distance tables.

    Detailed Time Complexity Analysis:
    1. Graph Construction:
       - Creating the initial adjacency list takes O(L).
       - Populating it by iterating through all R roads takes O(R).
       - Total: O(L + R).
    2. Finding Unique Locations:
       - Creating the mapping array takes O(L).
       - Looping through all B buses takes O(B).
       - Total: O(L + B).
    3. Running Dijkstra:
       - The loop runs B_u times, where B_u is the number of unique bus locations.
       - Each call to the Dijkstra function costs O(R*log L) in the worst case.
       - Total for this part: O(B_u *  (R * log L)).
    4. Function Definition and Return:
       - Defining the nested function and returning the final tuple are O(1) operations.

    Combined Time Complexity:
    - The total complexity is the sum of the parts: O(L + R) + O(L + B) + O(B_u * (R*log L)).
    - The Dijkstra loop is the dominant term. The general complexity is O(B_u * (R*log L)).
    - However, since the problem states that the number of unique pickup locations (B_u)
      is a small constant (at most 18), we treat B_u as O(1).
    - Therefore, the final simplified complexity for this entire function is O(L + B + R*log L).

    Detailed Space Complexity Analysis:
    1. 'city_adj_list': Stores 2*R entries for the bidirectional roads, plus L list heads. Space is O(L + R).
    2. 'unique_pickup_locations': Stores at most B_u location IDs. Space is O(B_u).
    3. 'location_to_dijkstra_index': A mapping array of size L. Space is O(L).
    4. 'min_distances_from_pickups': This is the largest data structure created. It stores B_u lists, and each list is of size L.
        The total space is O(B_u * L).

    Explanation of Space Complexity:
    - The total space is the sum of the data structures: O(L+R) + O(L) + O(B_u * L).
    - Since B_u is a constant (O(1)), the O(B_u * L) term simplifies to O(L).
    - The total space is therefore O(L + R + L) = O(L + R).

    Args:
    L (int): The total number of locations in the city.
    roads (list): A list of tuples representing roads, where each tuple is (u, v, w).
    buses (list): A list of tuples representing buses, where each tuple is (location, min, max).
    D (int): The maximum travel distance, used to optimize Dijkstra's runs.

    Returns:
    tuple: A tuple containing three items:
        1. get_shortest_dist (function): A fast, O(1) helper function that takes a student
           location and a pickup location and returns the pre-computed the shortest distance.
        2. unique_pickup_locations (list): The list of unique location IDs where buses are parked.
        3. location_to_dijkstra_index (list): The mapping array that converts a raw location
           ID into its compact index.
    """
    # L = 16
    # roads = 15
    # buses = [(0,3,5),(6,5,10),(15,5,10),(6,5,10)]

    # Building the Adjacency List
    city_adj_list = []

    # For finding unique pickup locations
    unique_pickup_locations = []

    location_to_dijkstra_index = []

    for i in range(L):  # O(L)
        # location_to_dijkstra_index =
        # [-1, -1, -1, -1, -1, -1, -1, -1,
        #  -1, -1, -1, -1, -1, -1, -1, -1] of length 18
        location_to_dijkstra_index.append(-1)

    # Loop L times, once for each location in the city.
    for _ in range(L):
        # In each iteration, add a new empty list. This new list will
        # eventually hold the neighbors for one of the locations.
        # after loop completes:
        # city_adj_list =
        # [
        #   [],  # neighbors of location 0
        #   [],  # neighbors of location 1
        #   [],  # neighbors of location 2
        #   [],  # ...]
        city_adj_list.append([])

    for road_tuple in roads:  # O(R) ; run for 15 times from the test case
        u, v, w = road_tuple  # unpacking the tuple ; u = 0 , v = 1 , w = 3 for first iteration

        # road_tuple = (0, 1, 3) ; u=0, v=1, w=3
        # add (1,3) to city_adj_list[0]
        # add (0,3) to city_adj_list[1]

        # Result:
        # 0: [(1,3)]
        # 1: [(0,3)]
        # Next list:
        # (0, 2, 5) ; u=0, v=2, w=5
        # 0: [(1,3), (2,5)]
        # 2: [(0,5)]
        # Next List:
        # (0, 3, 10) ; u=0, v=3, w=10
        # 0: [(1,3), (2,5), (3,10)]
        # 3: [(0,10)].....

        city_adj_list[u].append((v, w))  # O(1)
        city_adj_list[v].append((u, w))

    # Loop directly over the items in the buses list
    for bus_tuple in buses:  # O(B) , buses = [(0, 3, 5), (6, 5, 10), (15, 5, 10), (6, 5, 10)]
        bus_loc, _, _ = bus_tuple  # unpacking the tuple , bus_loc is 0, then 6, then 15, then 6 again
        if location_to_dijkstra_index[
            bus_loc] == -1:  # check if duplicate exists for instance we have 6 twice for bus_loc
            # iter 1:
            # bus_tuple = (0, 3, 5) ; bus_loc = 0
            # location_to_dijkstra_index[0] == -1 ; True
            # len(unique_pickup_locations) = len([]) = 0
            # Assign location_to_dijkstra_index[0] = 0
            # Append 0 to unique_pickup_locations
            # State now:
            # unique_pickup_locations = [0]
            # location_to_dijkstra_index = [0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]
            # this continues...
            # iter 4 (duplicate pickup):
            # bus_tuple = (6, 5, 10) -> bus_loc = 6
            # location_to_dijkstra_index[6] == -1 ; False (it's 1)
            # Skip assignment and append (already recorded).
            # Final state unchanged:
            # unique_pickup_locations = [0, 6, 15]
            # location_to_dijkstra_index = [0, -1, -1, -1, -1, -1, 1, -1, -1, -1, -1, -1, -1, -1, -1, 2]

            location_to_dijkstra_index[bus_loc] = len(unique_pickup_locations)
            unique_pickup_locations.append(bus_loc)

    min_distances_from_pickups = []

    # Here the unique_pickup_locations = [0, 6, 15] for our example
    for loc_id in unique_pickup_locations:  # O(B_u) where B_u is unique locations where buses pickup from
        # L = 16, city_adj_list populated from previous line in this method, loc_id: 0, 6, 15
        min_distances_from_pickups.append(dijkstra(L, city_adj_list, loc_id))  # O(L + R log L)

    # Using the three precomputed tables for this case:
    #   dijkstra_idx = 0 -> distances from pickup 0
    #   dijkstra_idx = 1 -> distances from pickup 6
    #   dijkstra_idx = 2 -> distances from pickup 15
    #
    # For Test Case 1:
    # From 0 to 4: 0 -> 1 (3) + 1 -> 4 (1) = 4  -> min_distances_from_pickups[0][4] == 4
    # From 6 to 12: direct edge (6,12,2) -> 2  -> min_distances_from_pickups[1][12] == 2
    # From 15 to 7: direct edge (7,15,1) -> 1  -> min_distances_from_pickups[2][7]  == 1
    # From 0 to 5: 0 -> 2 (5) + 2 -> 5 (2) = 7 -> min_distances_from_pickups[0][5] == 7
    # If pickup_loc isn't one of [0,6,15], mapping is -1 -> return inf (unreachable for lookup)

    def get_shortest_dist(student_loc, pickup_loc):
        dijkstra_idx = location_to_dijkstra_index[pickup_loc]
        if dijkstra_idx != -1:
            return min_distances_from_pickups[dijkstra_idx][student_loc]  # O(1)
        return float('inf')

    # Return the data as a tuple
    return get_shortest_dist, unique_pickup_locations, location_to_dijkstra_index


def add_edge(u, v, cap, graph):  # O(1)
    """
    Add a forward edge (u -> v) with capacity 'cap' and its reverse edge (v -> u) with
    capacity 0 to a residual graph stored as adjacency lists.
    Each adjacency list entry is a tuple: (to, capacity, rev_idx)
      - to: neighbor vertex id
      - capacity: current residual capacity on this directed edge
      - rev_idx: index of the reverse edge in the neighbor's adjacency list

    Time Complexity:  O(1)
    Space Complexity: O(1) extra (two appended tuples)

    Parameters:
        u (int): source vertex id
        v (int): destination vertex id
        cap (int | float): capacity for the forward edge
        graph (list[list[tuple]]): residual network adjacency lists
    """

    # forward edge stores index where backward will be appended
    graph[u].append((v, cap, len(graph[v])))
    # backward edge stores index of the forward edge we just appended
    graph[v].append((u, 0, len(graph[u]) - 1))


def build_flow_network(students, buses, D, get_shortest_dist,
                       unique_pickup_locations, location_to_dijkstra_index):
    """
    Builds the complete flow graph, including the circulation with demands transformation for handling
    minimum bus capacities.

    Approach:
    The function models the assignment problem as a complex flow network and solves it using a two-stage max-flow problem.
    The main path of the graph represents as follows;
    starting from a source_core which connects to each Student node with capacity 1 to ensure each student is assigned only once.
    A Student node is then connected to Location nodes if they are within the travel distance D, modeling eligibility.
    These Location nodes act as hubs, distributing students to the specific Bus nodes parked there.
    Finally, each Bus node connects to a sink_core with a capacity equal to (max_capacity - min_capacity), representing
    the number of "optional" or "flexible" seats available on that bus.
    A "return edge" is added from the sink_core back to the source_core, turning the graph into a closed loop. Then, the super_source
    injects a flow equal to the total_min_capacity into the network, representing the total mandatory demand for the entire trip. To collect
    this demand, each Bus node is given an edge to a super_sink with a capacity equal to its own min_capacity, representing its individual
    quota that must be filled. This construction allows the algorithm to first check if satisfying all minimums is feasible (Stage 1) and then,
    on the remaining residual network, solve for the optional students needed to meet the exact target T (Stage 2).

    Core Graph Layers:
    1. Source Node ('source_core'): Represents the pool of all students. Edges with capacity 1
       go from here to each student node, ensuring each student is assigned at most once.
    2. Student Nodes: One node for each of the S students.
    3. Location Nodes: One node for each of the B_u unique pickup locations. An edge exists
       from a student to a location if the student can travel to that location within distance D.
    4. Bus Nodes: One node for each of the B buses. An edge connects a location node to a
       bus node if that bus picks up at that location.
    5. Sink Node ('sink_core'): The destination for all assigned flow. The capacity of the
       edge from a Bus node to the Sink represents the "optional" seats on that bus, calculated
       as (max_capacity - min_capacity).
    6. Return Edge (sink_core -> source_core)
       Added with infinite capacity to turn the core graph into a circulation. Setup for enforcing lower bounds
       via super nodes.
    7. Super Source (super_source)
       Single edge super_source -> sink_core with capacity sum of (min_capacity).
       This injects exactly the total mandatory flow that must be satisfied across all buses.
    8. Super Sink (super_sink)
        For each bus j, add edge Bus_j -> super_sink with capacity min_capacity_j. This collects
        the mandatory flow per bus, forcing each bus to receive at least its minimum.

    Detailed Time Complexity Analysis:
    1. Initialization:
       - 'len(students)', 'len(buses)', 'len(unique_pickup_locations)': These are all O(1).
       - The loop to calculate 'total_min_capacity' runs B times, taking O(B) time.
       - Defining all node indices are O(1) arithmetic operations.
       - Initializing 'flow_graph' creates V empty lists, taking O(V) time, where V is the total
         number of nodes. Since V is proportional to S, this is O(S).

    2. Edge Addition Loops:
       - Source -> Student: The loop runs S times. Cost: O(S).
       - Student -> Location: A nested loop running S * B_u times. Since B_u is a small
         constant (<= 18), the cost is effectively O(S * 1) = O(S).
       - Location -> Bus: The loop runs B times. Cost: O(B).
       - Bus -> Sink: The loop runs B times. Cost: O(B).
       - Edges: One edge plus a loop that runs B times. Cost: O(B).

    Combined Time Complexity:
    - The total time is the sum of all steps: O(B) + O(S) + O(S*B_u) + O(B) + O(B) + O(B).
    - This simplifies to O(S*B_u + B).
    - We apply the problem's constraints:
      1. The number of unique locations 'B_u' is a small constant, so O(S*B_u) becomes O(S).
      2. For any solvable problem, the number of buses B is less than or equal to the number
         of students S (B <= S).
    - Therefore, the expression O(S + B) simplifies to just O(S). The total time complexity
      to build the graph is linear in the number of students.

    Detailed Space Complexity Analysis:
    - 'flow_graph': This is the primary data structure. Its space is O(V + E), where V is the
      number of vertices and E is the number of edges.
      - As analyzed in the time complexity, V = O(S) and E = O(S).
      - Therefore, the space for the graph is O(S).
    - 'node_indices': This is a tuple holding a constant number of integers, so it is O(1).
    - Other variables are simple integers, taking O(1) space.

    Explanation of Space Complexity:
    - The space is dominated by the storage for the flow graph itself. Since both the
      number of nodes and the number of edges in our efficient model scale linearly with the
      number of students (S), the overall space complexity of this function is O(S).

    Args:
        students (list): The list of student locations.
        buses (list): The list of bus tuples (location, min, max).
        D (int): The maximum travel distance for a student.
        get_shortest_dist (function): The pre-computed O(1) lookup function for distances.
        unique_pickup_locations (list): The list of unique location IDs for bus pickups.
        location_to_dijkstra_index (list): The mapping array from a location ID to its
            compact index for the Dijkstra results.

    Returns:
        tuple: A tuple containing two items:
        1. flow_graph (list of lists): The fully constructed adjacency list for the flow network.
        2. node_indices (tuple): A tuple containing the integer indices for all key nodes
           (source, sink, student_start, etc.) needed by other functions.
    """

    num_students = len(students)  # for test case 1; 25
    num_buses = len(buses)  # 4
    num_unique_locations = len(unique_pickup_locations)  # 3

    total_min_capacity = 0

    # Loop through each 'bus' tuple in the 'buses' list.
    for b in buses:  # O(B)  # iter over: (0,3,5), (6,5,10), (15,5,10), (6,5,10)
        # Get the minimum capacity from the bus tuple

        # after iter1: 0 + 3  = 3
        # after iter2: 3 + 5  = 8
        # after iter3: 8 + 5  = 13
        # after iter4: 13 + 5 = 18
        # final total_min_capacity = 18
        min_capacity_for_this_bus = b[1]
        total_min_capacity = total_min_capacity + min_capacity_for_this_bus

    # Define node indices
    source_core = 0
    student_node_start = 1  # student nodes = 1..25  (since num_students = 25)
    location_node_start = student_node_start + num_students  # 1 + 25 = 26
    bus_node_start = location_node_start + num_unique_locations  # 26 + 3 = 29
    sink_core = bus_node_start + num_buses  # 29 + 4 = 33
    super_source = sink_core + 1  # 33 + 1 = 34
    super_sink = super_source + 1  # 34 + 1 = 35
    num_total_nodes = super_sink + 1  # 35 + 1 = 36 (nodes 0..35)

    flow_graph = []

    for _ in range(num_total_nodes):  # O(super_source + source_core + students + B_u + sink + super_sink); 36
        flow_graph.append([])

    # Add the return edge to make it a circulation problem
    # adds: 33 -> 0 with cap = inf, and reverse 0 -> 33 with cap = 0
    add_edge(sink_core, source_core, float('inf'), flow_graph)

    # Edges: Source -> Student
    for i in range(num_students):  # O(S)
        # adds: 0 -> 1, 0 -> 2, ..., 0 -> 25 each with cap = 1 (one seat per student)
        add_edge(source_core, student_node_start + i, 1, flow_graph)

    # Edges: Student -> Location
    # O(S*B_u) ~ O(S); (B_u <= 18)
    for i in range(num_students):  # O(S)
        student_loc = students[i]
        for B_u_idx in range(num_unique_locations):  # O(B_u)
            pickup_loc = unique_pickup_locations[B_u_idx]
            if get_shortest_dist(student_loc, pickup_loc) <= D:
                add_edge(student_node_start + i, location_node_start + B_u_idx, 1, flow_graph)
    # (what this builds in Test Case 1; D = 5, pickups = [0,6,15])
    #   location nodes: loc(0) -> 26, loc(6) -> 27, loc(15) -> 28
    #   examples of eligibility edges (cap=1 each):
    #   i=0, student_loc=4:  dist(4->0)=4 <=5 -> add 1->26
    #   i=1, student_loc=10: dist(10->0)=1 <=5 -> add 2->26
    #   i=2, student_loc=8:  dist(8->0)=1 <=5 -> add 3->26
    #   i=3, student_loc=12: dist(12->6)=2 <=5 -> add 4->27
    #   i=14, student_loc=7: min(dist to 0,6,15) = 1 (to 15) -> add 15->28
    #   i=19, student_loc=15: dist(15->15)=0 <=5 -> add 20->28
    #   (many more are added similarly; each student can have edges to multiple pickup locations if within D)

    # Edges: Location -> Bus
    total_min_capacity = 0

    for j in range(num_buses):  # O(B)
        bus_loc = buses[j][0]  # bus pickup location: [0, 6, 15, 6]
        bus_min = buses[j][1]  # mins: [3, 5, 5, 5]
        bus_max = buses[j][2]  # maxs: [5,10,10,10]

        # Location -> Bus
        B_u_idx = location_to_dijkstra_index[bus_loc]
        add_edge(location_node_start + B_u_idx, bus_node_start + j, num_students, flow_graph)
        #   maps to:
        #   j=0 (bus at 0): 26 -> 29 cap 25
        #   j=1 (bus at 6): 27 -> 30 cap 25
        #   j=2 (bus at 15):28 -> 31 cap 25
        #   j=3 (bus at 6): 27 -> 32 cap 25
        #   (cap=25 means the location node can forward up to all students; real limits come next)

        # Bus -> Sink (optional seats)
        add_edge(bus_node_start + j, sink_core, bus_max - bus_min, flow_graph)
        #   optional caps:
        #   j=0: 29 -> 33 cap (5-3) = 2
        #   j=1: 30 -> 33 cap (10-5) = 5
        #   j=2: 31 -> 33 cap (10-5) = 5
        #   j=3: 32 -> 33 cap (10-5) = 5

        # Bus -> Super Sink (mandatory seats)
        add_edge(bus_node_start + j, super_sink, bus_min, flow_graph)
        #   mandatory caps (collecting mins):
        #   j=0: 29 -> 35 cap 3
        #   j=1: 30 -> 35 cap 5
        #   j=2: 31 -> 35 cap 5
        #   j=3: 32 -> 35 cap 5
        #   Track total mandatory flow
        total_min_capacity = total_min_capacity + bus_min  # accumulates to: 3 + 5 + 5 + 5 = 18

    # Super Source injects the total mandatory flow after we know the sum
    add_edge(super_source, sink_core, total_min_capacity,
             flow_graph)  # adds: 34 -> 33 with cap = 18  (this “injects” the exact total mins into the circulation)

    # Store necessary indices in a tuple for the return value
    node_indices = (source_core, sink_core, super_source, super_sink,
                    student_node_start, location_node_start,
                    bus_node_start)  # node_indices = (0, 33, 34, 35, 1, 26, 29)

    return flow_graph, node_indices


def disable_return_edge(flow_graph, source_core, sink_core):
    """
    Disables the 'return edge' from the main sink to the main source after Stage 1 of the
    max-flow algorithm. This is important to correctly transition from the
    circulation feasibility check (Stage 1) to the standard s-t max-flow for the
    remaining capacity (Stage 2).

    Approach:
    - The function iterates through the adjacency list of the 'sink_core' node to find the
      specific edge that points back to the 'source_core'.
    - Once this forward 'return edge' is found, its capacity is set to 0
    - The function then uses the reverse-edge index stored in that edge's
      tuple to perform a direct, O(1) lookup of the corresponding reverse edge (from
      'source_core' to 'sink_core').
    - The capacity of this reverse edge is also set to 0. This is needed because Stage 1
      pushed flow through the return edge, creating a positive residual capacity on this reverse edge.
      Disabling it prevents the algorithm from using this credit as an invalid shortcut path directly from source to sink in Stage 2.
    - The function breaks the loop immediately after finding and disabling the edges for
      efficiency.

    Detailed Time Complexity Analysis:
    1. Loop Initialization:
       - The 'for' loop iterates through the neighbors of 'sink_core'. The number of
         neighbors is the degree of the 'sink_core' node.
       - 'sink_core' is connected to B bus nodes, the 'super_source', and the 'source_core'.
         Therefore, its degree is O(B). The loop runs at most O(B) times.
    2. Work Inside the Loop:
       - All operations within the loop are constant-time, O(1). This includes tuple
         unpacking, integer comparison, list lookups, and tuple creation/assignment.
    3. Break Statement:
       - The 'break' ensures that the loop terminates as soon as the target edge is found,
         making the average case faster, but the worst-case remains O(B).

    Combined Time Complexity:
    - The function's runtime is dominated by the 'for' loop. In the worst case, it
      iterates through all neighbors of the 'sink_core'.
    - The complexity is O(degree(sink_core)), which is O(B).
    - In the context of the problem, we know that B <= S (the number of buses
      is less than or equal to the number of students), so the complexity is also bounded by O(S).

    Detailed Space Complexity Analysis:
        - The function does not create any new data structures that scale with the size of the
          input graph.
        - It only uses a few local variables for iteration and storing tuple data (e.g.,
          'i', 'v_neighbor', 'cap'), which consume a constant amount of space.
        - The 'flow_graph' itself must already exist in memory and is modified in-place.

        Explanation of Space Complexity:
        - The space complexity is Θ(V + E), dominated by the existing 'flow_graph' structure.
          The function requires only O(1) additional space, as it performs all operations
          in-place without allocating extra memory proportional to the input size.

    Args:
        flow_graph (list of lists): The flow network, which will be modified in-place.
        source_core (int): The index of the main source node.
        sink_core (int): The index of the main sink node (where the return edge originates).
    """
    for i in range(len(flow_graph[sink_core])):  # scan neighbors of node 33
        # Get the edge tuple at the current index 'i'.
        edge_tuple = flow_graph[sink_core][i]  # edge (33 -> v_neighbor)
        v_neighbor, cap, rev_idx_in_source = edge_tuple

        if v_neighbor == source_core:  # found the return edge 33 -> 0
            flow_graph[sink_core][i] = (v_neighbor, 0, rev_idx_in_source)
            v_rev, cap_rev, rev_idx_rev = flow_graph[source_core][rev_idx_in_source]  # reverse edge 0 -> 33
            flow_graph[source_core][rev_idx_in_source] = (v_rev, 0, rev_idx_rev)
            break


def reconstruct_assignment(flow_graph, students, buses, node_indices):
    """
    Builds the final answer list of student-to-bus assignments. After the main
    algorithm finds a valid solution, this function looks at the final state of the
    flow network. For each student who was assigned, it traces the path their 'flow'
    took through the network to determine which specific bus they ended up on,
    creating the final output list (e.g., [0, -1, 1, ...]).

     Approach:
    - The function iterates through each student from 0 to S-1 to determine their assignment.
    - For each student, it performs a two-step trace through the residual graph:
      1. Trace Student -> Location: It first finds which 'Location Node' the student's flow
         was sent to. This is done by finding the outgoing edge from the 'Student Node'
         whose residual capacity is 0 (that is a flow of 1 was pushed through it).
      2. Trace Location -> Bus: Once the intermediate 'Location Node' is found, it searches
         for a 'Bus Node' that received flow from that same 'Location Node'. This is
         identified by checking the reverse edges (Bus -> Location) for a positive
         capacity, which indicates that flow was pushed forward during the algorithm.
    - To correctly handle cases where multiple students flow paths merged at the same
      'Location Node' hub, the function "consumes" one unit of reverse-flow capacity from
      a bus's edge each time an assignment is made. This prevents a single bus slot from
      being incorrectly assigned to multiple students.
    - The search is made efficient by using 'break' statements to stop searching for a bus
      as soon as a valid one is found for the current student.

    Detailed Time Complexity Analysis:
    1. Initialization:
       - 'allocation = [-1] * num_students' takes O(S) time.
       - Unpacking the 'node_indices' tuple is an O(1) operation.
    2. Main Loop (over students):
       - The main 'while i < num_students:' loop runs S times. The total complexity is
         S multiplied by the cost of the work done inside for each student.
    3. First Inner Loop (finding location node):
       - The 'while neighbor_idx < len(flow_graph[student_node]):' loop iterates over
         the direct neighbors of a single student node.
       - A student node is only connected to the source and at most B_u location nodes,
         where B_u is a small constant (<= 18). Thus, this loop runs in O(1) time.
    4. Second Nested Loop (finding bus node):
       - The 'while j < num_buses:' loop iterates up to B times in the worst case.
       - The innermost 'while k < len(flow_graph[bus_node]):' loop iterates over the
         neighbors of a single bus node, which is a small constant number. This is O(1).
       - Therefore, the work to find the bus for one student is dominated by the j-loop, making it O(B).
    5. Work Inside the Main Loop:
       - For each student, the work is the sum of the sequential inner loops: O(1) + O(B) = O(B).

    Combined Time Complexity:
    - The total time is (Outer loop iterations) * (Work inside the loop).
    - Total = O(S) * O(B) = O(S * B).
    - Since we know that for any solvable problem B <= S, the worst-case complexity occurs
      when B is proportional to S. This makes the complexity O(S * S) = O(S^2).

    Detailed Space Complexity Analysis:
    - 'allocation' list: This is the main data structure created. Its size is S, so it
      requires O(S) space.
    - Local Variables: All other variables ('i', 'j', 'k', 'loc_node_flowed_to', etc.)
      use a constant, O(1), amount of space.
    - The function modifies the input 'flow_graph' in-place but does not create a copy.
    - The dominant space used by this function is for the 'allocation' list,
      which has a size proportional to the number of students, S.
    - Therefore, the space complexity is O(S).

   Args:
        flow_graph (list of lists): The final residual graph after all max-flow stages.
            This graph is modified in-place as assignments are "consumed".
        students (list): The original list of student locations, used to get 'num_students'.
        buses (list): The original list of buses, used to get 'num_buses'.
        node_indices (tuple): A tuple containing the integer indices for all key nodes
           (source, sink, student_start, etc.) needed to navigate the graph.

    Returns:
        list: The final 'allocation' list of size S, where allocation[i] is either -1
              or the bus ID to which student i is assigned.
    """

    # following test case is only depicted for Iteration i = 0 (student0 at loc 4 -> pickup 0 -> bus 0)
    num_students = len(students)  # 25
    num_buses = len(buses)  # 4

    allocation = []
    for _ in range(num_students):
        allocation.append(-1)  # O(S)  # [-1, -1, ...] (25 entries)

    # Unpack the indices tuple
    _, _, _, _, student_node_start, location_node_start, bus_node_start = node_indices  # (0, 33, 34, 35, 1, 26, 29)

    # Entire block time comp: O(S*B) ~ O(S^2) as B <= S
    i = 0
    while i < num_students:  # O(S)  # 0 < 25 -> True
        student_node = student_node_start + i  # 1 + 0 = 1
        loc_node_flowed_to = -1
        # Find which location node the student's flow went to
        neighbor_idx = 0

        while neighbor_idx < len(flow_graph[student_node]):  # O(1)   # scan neighbors of node 1
            v_neighbor, cap, _ = flow_graph[student_node][neighbor_idx]
            # first few neighbors until we hit the edge to a location with cap==0
            # the used edge is (1 -> 26) with cap == 0
            if location_node_start <= v_neighbor < bus_node_start and cap == 0:
                loc_node_flowed_to = v_neighbor
                break
            neighbor_idx = neighbor_idx + 1

        if loc_node_flowed_to != -1:  # 26 != -1 -> True
            # Now find which bus that location's flow went to
            j = 0
            while j < num_buses:  # O(B)  # try buses j = 0..3
                bus_node = bus_node_start + j  # 29, 30, 31, 32 ...
                bus_found_for_student = False

                # Check the reverse edge (bus -> location) for available flow credit
                # O(1);  it iterates through the neighbors of a single bus node,
                # which is always a small, constant number of connections (to its Location node, the sink_core, and the super_sink).

                k = 0
                while k < len(flow_graph[bus_node]):
                    v_rev, cap_rev, rev_idx = flow_graph[bus_node][k]
                    # look for reverse credit on bus -> location:
                    # we find (bus_node=29 -> v_rev=26) with cap_rev > 0  (was 1+ from Stage 2)
                    if v_rev == loc_node_flowed_to and cap_rev > 0:
                        allocation[i] = j
                        # Consume this unit of flow so it's not reused
                        flow_graph[bus_node][k] = (v_rev, cap_rev - 1, rev_idx)
                        bus_found_for_student = True
                        break  # Stop searching edges for this bus
                    k = k + 1

                if bus_found_for_student:
                    break  # Stop searching for other buses for this student
                j = j + 1
        i = i + 1

    return allocation


def assign(L, roads, students, buses, D, T):
    """
    Determines if a valid assignment of 'T' students to 'B' buses is possible given a
    city map and various constraints, returning one such assignment if it exists.
    The function solves this problem by modeling it as a max-flow problem with demands, which
    is solved using a two-stage Ford-Fulkerson algorithm on a flow network.

    Approach:
    1.  Initial Feasibility Check: Firstly it performs a simple, high-level check to ensure
        the target 'T' is within the globally possible range defined by the sum of all
        minimum and maximum bus capacities.
    2.  Distance Preprocessing: It calls 'preprocess_distances' to build a representation
        of the city map and run Dijkstra's algorithm from each unique pickup spot. This
        pre-computes all necessary travel times for fast, O(1) lookups later.
    3.  Graph Construction: It calls 'build_flow_network' to create the flow graph. This
        graph includes a main path for student assignments and a "supervisory
        layer" (using a circulation with demands transformation) to handle the minimum
        capacity constraints.
    4.  Two-Stage Max-Flow Execution:
        - Stage 1 (Feasibility): It calls 'ford_fulkerson' on the supervisory sub-problem
          to check if it is possible to meet all minimum capacity
          requirements simultaneously. If this fails, no solution exists.
        - Stage 2 (Optional Assignment): If Stage 1 succeeds, it calls 'disable_return_edge'
          to modify the graph for a standard s-t flow, then calls 'ford_fulkerson' a
          second time to assign the remaining "optional" students needed to reach the
          final target 'T'.
    5.  Reconstruction: If both flow stages are successful, it calls 'reconstruct_assignment'
        to translate the final state of the residual graph back into a user-friendly
        list of student assignments, which is then returned.

    Detailed Time Complexity Analysis:
    1.  Feasibility Check: The loops to calculate total min/max capacity run B times, taking O(B).
    2.  preprocess_distances(L, roads, buses) : O(L + B + R*log L).
    3.  Building the Flow Network: build_flow_network(): O(S)
    4.  Check if minimums can be met: ford_fulkerson(flow_graph, super_source, super_sink, T_limit=total_min_capacity): O(ST)
    5.  Find the remaining optional assignments: ford_fulkerson(flow_graph, source_core, sink_core, T_limit=F_remaining)  # O(ST)
    6.  Reconstruct the Final Assignment:  reconstruct_assignment(flow_graph, students, buses, node_indices): O(S^2)

    Combined Time Complexity Analysis:
    O(B) + O(L + B + Rlog L) + O(S) + O(ST) + O(ST) + O(S^2)
    = O(S) + O(L) + O(S) + O(Rlog L) + O(ST) + O(S^2) ; since B<=S
    = O(L + Rlog L + ST); since in worst case T<=S

    Detailed Space Complexity Analysis:
    - The 'city_adj_list' inside 'preprocess_distances' requires O(L+R) space.
    - The 'min_distances_from_pickups' array requires O(B_u * L), which simplifies to O(L)
      since B_u is a constant.
    - The 'flow_graph' created in 'build_flow_network' requires O(V+E) = O(S) space.
    - Helper arrays inside 'ford_fulkerson' ('parent', 'visited') require O(V) = O(S) space.
    - The final 'allocation' list requires O(S) space.

    Explanation of Space Complexity:
    - The total space is the sum of the largest data structures required at any
      one time. The main contributors are the space for the city map, O(L+R), and the
      space for the flow network and its related arrays, O(S).
    - The final combined space complexity is O(S + L + R).

    Args:
        L (int): The total number of locations in the city.
        roads (list): A list of tuples (u, v, w) representing bidirectional roads.
        students (list): A list of integers denoting the location of each student.
        buses (list): A list of tuples (location, min_capacity, max_capacity) for each bus.
        D (int): The maximum distance a student is willing to travel to a pickup point.
        T (int): The exact number of students that must be transported in a valid solution.

    Returns:
        list or None: If a valid assignment exists, it returns a list of length S where
        allocation[i] is the bus ID for student i, or -1 if the student is not
        traveling. If no solution exists, it returns None.
    """
    # Example test case includes Test Case 1 from the brief where;
    # buses = [(0,3,5), (6,5,10), (15,5,10), (6,5,10)]

    # Step 1: Processing & Feasibility Check
    total_min_capacity = 0
    for bus in buses:  # O(B) ;   buses = [(0,3,5), (6,5,10), (15,5,10), (6,5,10)]
        min_capacity_of_this_bus = bus[1]  # iter 1: 3; iter 2: 5; iter 3: 5; iter 4: 5
        # after each iter:
        # after bus0: 0 + 3  = 3
        # after bus1: 3 + 5  = 8
        # after bus2: 8 + 5  = 13
        # after bus3: 13 + 5 = 18
        # result: total_min_capacity: 18
        total_min_capacity = total_min_capacity + min_capacity_of_this_bus

    total_max_capacity = 0
    for bus in buses:  # O(B) ;  buses = [(0,3,5), (6,5,10), (15,5,10), (6,5,10)]
        max_capacity_of_this_bus = bus[2]  # iter 1: 5; iter 2: 10; iter 3: 10; iter 4: 10
        # after each iter:
        # after bus0: 0 + 5   = 5
        # after bus1: 5 + 10  = 15
        # after bus2: 15 + 10 = 25
        # after bus3: 25 + 10 = 35
        # result: total_max_capacity = 35
        total_max_capacity = total_max_capacity + max_capacity_of_this_bus

    if T < total_min_capacity or T > total_max_capacity:  # T = 22 ; if 22 < 18 or 22 > 35 return None; which is false for this case
        return None

    # L=16, roads = 15, buses = [(0,3,5),(6,5,10),(15,5,10),(6,5,10)]
    # unique_pickup_locations = [0,6,15]
    # location_to_dijkstra_index = [0, -1, -1, -1, -1, -1, 1, -1, -1, -1, -1, -1, -1, -1, -1, 2]
    # loc 0 , loc 6, loc 15 are the pickup points hence it's not -1
    get_shortest_dist, unique_pickup_locations, location_to_dijkstra_index = preprocess_distances(L, roads,
                                                                                                  buses)  # O(L + B + R*log L)

    # Step 2: Building the Flow Network
    # students =  [4, 10, 8, 12, 12, 13, 13, 13, 13, 13, 13, 13, 13, 5, 7, 7, 7, 7, 7, 15, 15, 7, 4, 8, 9]
    # buses =  [(0, 3, 5), (6, 5, 10), (15, 5, 10), (6, 5, 10)]
    # D = 5
    # example for get_shortest distance
    # get_shortest_dist(4, 0) == 4
    # get_shortest_dist(12, 6) == 2
    # get_shortest_dist(7, 15) == 1
    # get_shortest_dist(10, 14) == inf
    # unique_pickup_locations = [0, 6, 15]
    # location_to_dijkstra_index = [0, -1, -1, -1, -1, -1, 1, -1, -1, -1, -1, -1, -1, -1, -1, 2]

    graph_data = build_flow_network(students, buses, D, get_shortest_dist,
                                    unique_pickup_locations, location_to_dijkstra_index)  # O(S)
    flow_graph, node_indices = graph_data
    # For Test Case 1:
    #   node_indices = (source_core, sink_core, super_source, super_sink,
    #                    student_node_start, location_node_start, bus_node_start)
    #                = (0, 33, 34, 35, 1, 26, 29)
    #
    #   So node id ranges are:
    #     source_core = 0
    #     students = 1..25
    #     locations (pickups)  = 26 (for 0), 27 (for 6), 28 (for 15)
    #     buses = 29 (bus0 at loc 0), 30 (bus1 at loc 6), 31 (bus2 at loc 15), 32 (bus3 at loc 6)
    #     sink_core = 33
    #     super_source = 34
    #     super_sink = 35
    #     total nodes = 0..35 (36 nodes)

    source_core, sink_core, super_source, super_sink, _, _, _ = node_indices

    # source_core  = 0
    # sink_core = 33
    # super_source = 34
    # super_sink   = 35

    # Flow edges that were created inside build_flow_network:
    # Return edge:          33 -> 0   (cap = inf)
    # Source -> Students:    0 -> 1..25 (cap = 1 each; 25 edges)
    # Students -> Locations: only if dist(student_loc, pickup) <= D (cap = 1)
    # e.g. 1->26 (student0 at loc 4 to pickup 0), 2->26 (student1 at loc 10 to pickup 0), 4->27 (student3 at loc 12 to pickup 6), 15->28 (student14 at loc 7 to pickup 15), etc.
    # Location -> Bus:       26->29, 27->30, 28->31, 27->32 (cap = 25 each)
    # Bus -> Sink (optional):29->33 cap 2; 30->33 cap 5; 31->33 cap 5; 32->33 cap 5
    # Bus -> SuperSink (min):29->35 cap 3; 30->35 cap 5; 31->35 cap 5; 32->35 cap 5
    # SuperSource -> Sink:   34->33 cap 18  (sum of mins)

    # Step 3: Run Max-Flow
    # Check if minimums can be met

    # super_source = 34, super_sink = 35, T_limit = 18
    flow_req = ford_fulkerson(flow_graph, super_source, super_sink, total_min_capacity)  # O(S*T)

    # if  flow_req < 18  -> infeasible mins
    # for Test Case 1, flow_req == 18, so we DO NOT return
    if flow_req < total_min_capacity:
        return None

    # Disable the circulation return edge before Stage 2
    disable_return_edge(flow_graph, source_core, sink_core)  # O(S)

    # Find the remaining optional assignments
    F_remaining = T - total_min_capacity  # 22 - 18 = 4 (optional seats to fill in Stage 2)
    flow_rem = ford_fulkerson(flow_graph, source_core, sink_core, F_remaining)  # O(S*T)
    # source_core=0, sink_core=33, T_limit=4 -> try to route 4 more units via
    # 0 -> student -> location(26/27/28) -> bus(29..32) -> 33   (optional edges bus->33)

    if total_min_capacity + flow_rem < T:  # if 18 + flow_rem < 22 -> infeasible, for Test Case 1, flow_rem should be 4, so we continue
        return None

    # Step 4: Reconstruct the Final Assignment
    allocation = reconstruct_assignment(flow_graph, students, buses, node_indices)  # O(S^2)

    return allocation


"Question 2:"


class SuffixTrie:
    def __init__(self):
        """
        A single node in a compressed suffix trie for transposition-invariant motifs.

        Approach:
        - This node is used in a trie where each edge is keyed by an integer
          difference in the range [-25..25] (i.e. shifts between lowercase notes).
        - To allow O(1) child access, the node keeps a fixed array of 51 slots:
          index 0 maps to diff = -25, index 25 -> diff = 0, index 50 -> diff = 25.
        - The node also tracks:
            - frequency: how many distinct songs reach this node (counted once per song).
            - original_pattern_info: a pointer (song_id, char_start) to reconstruct a
              representative pattern for getFrequentPattern(K).
            - latest_song_id: remembers which song last updated this node so that the method do not
              double-count the same song.

        Attributes:
        - children (list[list|None], length 51) child pointers bucketed by first difference.
        - frequency (int): number of distinct songs that reach this node.
        - original_pattern_info (tuple[int, int] | None): (song_id, char_start) to rebuild a chosen best-K pattern.
        - latest_song_id (int): the most recent song_id that updated this node.

        Complexity:
            Time: O(1) per node creation
            Space: O(1) per node
        """
        # 51 edges for diffs in [-25..25] mapped to [0..50]
        self.children = []
        i = 0
        while i < 51:  # O(1)
            self.children.append(None)
            i = i + 1

        self.frequency = 0
        self.original_pattern_info = None
        self.latest_song_id = -1

    def suffix_edge(self, start_idx, end_idx, target_node):
        """
        Create a compressed edge list.
        It stores pointers [start, end] into a global difference array
        and the node it points to.

        Field order convention:
        edge[0] = start_idx
        edge[1] = end_idx
        edge[2] = target_node

        Arguments:
        start_idx (int): Inclusive start index in the global differences array.
        end_idx (int):   Exclusive end index in the global differences array.
        target_node (SuffixTrie): The downstream node this edge points to.

        Returns:
        list: A 3-item list [start_idx, end_idx, target_node], where interval [start_idx, end_idx]
         indexes into the global differences array.

        """
        return [start_idx, end_idx, target_node]


class Analyser:
    def __init__(self, sequences):
        """
        Initialize the Analyser by preprocessing all note sequences and
        constructing a generalized compressed suffix-trie over their
        difference representations.

         Approach:
            1. Store all input note sequences in memory for later motif reconstruction.

            2. Compute:
                 - M => the maximum song length among all input sequences.
                 - N => the total number of songs in the dataset.
               These values determine the bounds for array sizes and complexity limits.

            3. Initialize key data structures:
                 - best_freq[K]: Tracks, for each possible pattern length K (2 <= K <= M),
                   the highest number of distinct songs that contain a pattern of that length.
                 - precalculated_patterns[K]: Stores a pointer (song_id, char_start)
                   to one representative pattern achieving that highest frequency.
                   This enables getFrequentPattern(K) to return the result in O(K) time.

            4. Convert each song into its "difference sequence" to achieve transposition
               invariance. Consecutive notes are replaced by their integer pitch intervals,
               e.g., 'acd' => [+2, +1]. Concatenate all songs' differences into one global
               list (global_diffs) while recording (start, end) index boundaries for each
               song in song_boundaries. This prevents any pattern from crossing song limits.

            5. Build a compressed suffix-trie:
                 - Each edge references a slice [start, end] of the global_diffs array
                   instead of duplicating the data.
                 - Each node maintains per-song occurrence counts (frequency) and
                   records the best pattern observed for each K.
                 - For every song, insert all its suffixes (starting positions within its
                   diff interval) using insert_suffix(). Each insertion either traverses
                   existing edges or creates new branches, updating node statistics on the fly.

            Detailed Time Complexity:
            Let N be the number of songs, M the maximum song length, and for a given song
            of length L (notes) its diff length is (L−1).

            1) Find the maximum length. O(N)

            2) Initialise arrays (size = M+1) : O(M)
               - best_freq and precalculated_patterns.

            3) create_global_diffs(sequences)  : O(NM)

            4) Suffix insertions:
               - For one song of length L <= M:
                   a. You insert every suffix of its diff array. The r-th call processes a
                     suffix of remaining length R = r (from L−1 down to 1).
                   b. Each call to insert_suffix runs in O(R) time (forward-only scans on
                     compressed edges; no backtracking).
                   c. Summation over that song: sigma{r=1}^{L−1} O(r) = O(L^2) <= O(M^2). (upper bound is signified as ^{L-1})
               - Across all N songs: O(NM^2).

            Total Time Comp:  O(N) + O(M) + O(NM) + O(NM^2) = O(NM^2)

            Detailed Space Complexity:
            Let C = sigma|s_i| be the total number of characters across all songs (C <= N*M).
            - self.seqs => O(C)
              Stores the original input strings for reconstruction.
            - self.global_diffs => O(C)
              One integer per adjacent pair across all songs: sigma(|s_i|−1) = O(C).
            - self.song_boundaries => O(N)
              One (start,end) pair per song.
            - Suffix-trie nodes + edges => O(C) = O(NM)
                Compressed edges hold (start,end,target) pointers into global_diffs (no substring copies).
                Total nodes/edges is linear in total diffs; fixed 51-slot child arrays are a constant factor.
                Per-node fields (frequency, latest_song_id, original_pattern_info) are O(1) each.
            - best_freq (size M+1) => O(M)
            - precalculated_patterns (size M+1) => O(M)

            Time Complexity: O(NM^2)
            Space Complexity: O(NM)
        """

        # Using Test Case 1 from brief:
        # demo_songs = ["cegec", "gdfhd", "cdfhd"]

        self.seqs = sequences  # ["cegec", "gdfhd", "cdfhd"]

        i = 0
        max_len = 0
        while i < len(sequences):  # O(N) , M = 5 (longest sequence has length 5, from given Test case 1)
            s = sequences[i]
            if len(s) > max_len:
                max_len = len(s)
            i += 1

        self.max_song_len = max_len

        self.num_songs = len(sequences)  # 3  (there are three songs)

        size = self.max_song_len + 1  # 6

        self.best_freq = []
        self.precalculated_patterns = []
        i = 0
        while i < size:  # O(M), runs size = M+1 times; with Test Case 1, M=5 -> size=6 -> j = 0,1,2,3,4,5
            self.best_freq.append(0)  # after loop: self.best_freq == [0, 0, 0, 0, 0, 0]
            self.precalculated_patterns.append(None)
            i += 1  # after loop: self.precalculated_patterns == [None, None, None, None, None, None]

        # 1. Create one master array for all difference sequences

        # Test Case 1 sequences = ["cegec", "gdfhd", "cdfhd"]
        # Compute diffs for each:
        #   "cegec"  -> letter codes [2,4,6,4,2] -> diffs [ +2, +2, -2, -2 ]
        #   "gdfhd"  -> letter codes [6,3,5,7,3] -> diffs [ -3, +2, +2, -4 ]
        #   "cdfhd"  -> letter codes [2,3,5,7,3] -> diffs [ +1, +2, +2, -4 ]
        # Concatenated:
        #   self.global_diffs == [2, 2, -2, -2, -3, 2, 2, -4, 1, 2, 2, -4]
        # Per-song bounds in global_diffs:
        #   song 0 ("cegec")  : [0, 4]
        #   song 1 ("gdfhd")  : [4, 8]
        #   song 2 ("cdfhd")  : [8, 12]
        # So:
        #   self.song_boundaries == [(0, 4), (4, 8), (8, 12)]

        self.global_diffs, self.song_boundaries = self.create_global_diffs(sequences)  # O(NM)

        # allocates a node with 51 None children, freq=0, latest_song_id=-1
        root = SuffixTrie()  # O(1)

        # 2. Insert all suffixes from the master array into the tree
        for song_id in range(self.num_songs):  # O(N) , loops over song_id = 0,1,2
            start_of_song, end_of_song = self.song_boundaries[song_id]
            # Insert every non-empty suffix of the song's difference sequence
            for suffix_start_ptr in range(start_of_song, end_of_song):
                # song 0 ("cegec") : bounds [0,4]  -> suffix_start_ptr = 0,1,2,3
                # song 1 ("gdfhd") : bounds [4,8]  -> suffix_start_ptr = 4,5,6,7
                # song 2 ("cdfhd") : bounds [8,12] -> suffix_start_ptr = 8,9,10,11
                self.insert_suffix(root, song_id, suffix_start_ptr)  # O(M^2)

    def getFrequentPattern(self, K: int) -> list[str]:
        """
        Returns a list of chars for a most-frequent length-K motif.

        Approach:
        - Uses results precomputed during __init__: for each K (2...M) the method store a
          pointer to an occurrence (song_id, start_index) of a K-length pattern
          that appears in the maximum number of distinct songs.
        - At query time,:
            (1) validate K,
            (2) read the stored pointer for K,
            (3) reconstruct the K characters from the original song via collect_pattern().

        Inputs:
        - K (int): Desired motif length, must satisfy 2 <= K <= M.

        Functional Description:
        - If K is out of range, returns [].
        - Looks up self.precalculated_patterns[K]; if no pattern was recorded,
          returns [].
        - Otherwise reconstructs and returns the motif characters using
          collect_pattern(song_id, start, K).

        Time & Space Complexity Analysis:
        - Time: O(K)
            - O(1) to validate K and fetch ptr from self.precalculated_patterns.
            - O(K) to materialise the answer: read exactly K consecutive
              characters from the source song and place them into the output list.
              Each character access is O(1), repeated K times : O(K) total.
            - If we return early (invalid K or ptr is None), the work is O(1).
        -  Space: O(K)
            - The only asymptotic cost is the output list holding K characters.
              Local variables and references are O(1). If the method return early with [],
              extra space is O(1).

        Time complexity: O(K)
        Space complexity: O(K) for the returned list
        """

        # I will discuss the case, where K = 3
        if K < 2 or K > self.max_song_len:  # K=3, M=5 -> False  (valid K within range)
            return []
        ptr = self.precalculated_patterns[K]  # ptr == (1, 0) -> pattern starts in song 1 ("gdfhd") at index 0
        if ptr is None:
            return []
        s_id, start_char_idx = ptr  # s_id = 1, start_char_idx = 0
        return self.collect_pattern(s_id, start_char_idx,
                                    K)  # O(K); builds a list of K chars from the original string slice  # returns ['d','f','h']

    def create_global_diffs(self, sequences):
        """
        Converts all songs to difference sequences and concatenates them into
        a single list, returning the list and the boundaries for each song.

        Approach:
        - For each song s, map letters 'a'...'z' to integers 0..25 and compute
          consecutive differences: diff[i] = code[i+1] - code[i].
        - Append these per-song differences to one global array 'global_diffs'
        - Track (start, end) indices per song in 'song_boundaries' so later code
          can cap suffix insertions at song boundaries (no cross-song matches).

        Detailed Time Complexity Analysis:
        - Let C = total number of characters across all songs
          Since each song has at most M notes and there are N songs, C <= NM

        Time:
        - The method performs one complete linear scan over all characters in
          all songs. For each character, it performs constant-time arithmetic,
          comparison, and (when applicable) an amortised O(1) append.
        - Outer loop: O(N)
        - Inner loop per song: O(|s|); summed across songs : O(C) <= O(NM)
        - Therefore total worst time complexity is O(NM).

        Time Comp: O(NM)

        Detailed Space Complexity Analysis:
        - 'global_diffs': stores one integer per adjacent note pair: O(C) = O(NM)
        - 'song_boundaries': stores N pairs (2N integers): O(N)
        - Temporary variables (pos, prev, cur): O(1)
        - Total space = O(NM) + O(N) + O(1) = O(NM)
        """
        # Song 0: "cegec"
        #   codes: [2,4,6,4,2]
        #   diffs: [+2, +2, -2, -2]
        #   start=0, pos after this song=4
        #   global_diffs so far = [2, 2, -2, -2]
        #   song_boundaries = [(0,4)]

        # Song 1: "gdfhd"
        #   codes: [6,3,5,7,3]
        #   diffs: [-3, +2, +2, -4]
        #   start=4, pos after this song=8
        #   global_diffs so far = [2, 2, -2, -2, -3, 2, 2, -4]
        #   song_boundaries = [(0,4), (4,8)]

        # Song 2: "cdfhd"
        #   codes: [2,3,5,7,3]
        #   diffs: [+1, +2, +2, -4]
        #   start=8, pos after this song=12
        #   global_diffs so far = [2, 2, -2, -2, -3, 2, 2, -4, 1, 2, 2, -4]
        #   song_boundaries = [(0,4), (4,8), (8,12)]

        global_diffs = []
        song_boundaries = []
        pos = 0

        for s in sequences:  # O(N) , iterate 3 songs for our test case
            start = pos
            prev = None
            for ch in s:  # O(s)
                cur = ord(ch) - ord('a')  # [Example: "cegec" codes 2,4,6,4,2 ; "gdfhd" 6,3,5,7,3 ; "cdfhd" 2,3,5,7,3]
                if prev is not None:
                    global_diffs.append(cur - prev)
                    pos = pos + 1  # [Example pos evolves: 0->4->8->12]
                prev = cur
            song_boundaries.append((start, pos))  # O(1)

        return global_diffs, song_boundaries

    def insert_suffix(self, root, song_id, suffix_start_ptr):
        """
        Insert one suffix (identified by its start pointer into the song's diff array)
        into the compressed suffix trie. Handles the case where the suffix ends
        exactly at a split point (no lookahead).

        Approach:
        - The method traverse the trie from 'root', following the current suffix starting at
          'cursor = suffix_start_ptr' up to 'song_end' (the end index of this song's
          differences in 'global_diffs').
        - At each step the method either:
            (a) finds no outgoing edge for the next diff and create a new edge that
                consumes all remaining diffs in this suffix; or
            (b) match along an existing compressed edge; if a mismatch occurs in the
                middle, the method split the edge at the mismatch and attach a new edge for
                the remainder of this suffix; if the method fully match the edge, the method advance.
        - The method updates node statistics ('frequency', 'latest_song_id', and best-for-K
          pattern pointer) at each visited node.

        - Let 'song_end = song_boundaries[song_id][1]'. The method process diffs in
         'global_diffs[cursor : song_end]'.
        - If the next-child slot is empty, we create a new compressed edge that
          spans to 'song_end' and return (this suffix is done).
        - If there is an edge, we scan along its label comparing diffs:
            - Full match (edge fully consumed): advance 'cursor' by 'match_len',
              move to the edge's target node, and continue.
            - Partial match (mismatch inside edge): split the edge at 'match_len';
              attach a new edge for the remainder of this suffix; update stats and
              return.
        - If we finish exactly at a node ('cursor == song_end'), we perform a final
          stats update at that node.

        Inputs:
        - root (SuffixTrie): The root node of the suffix-trie.
        - song_id (int): Which song this suffix belongs to; used for counting
          "once per song" frequency.
        - suffix_start_ptr (int): Start index (in 'global_diffs') of the suffix
          within this song's diff interval.

        Detailed Time Complexity:
        Let R = (song_end - suffix_start_ptr) be the number of diffs remaining in
        this suffix. During this call:
          - When scanning an existing compressed edge, the method compares up to
            'match_len' diffs. If the method fully matches, it performs
            'cursor += match_len', ensuring those positions are never re-compared
            (forward-only traversal).
          - If the method partially matches, it splits the edge and returns immediately.

        Therefore, the total number of diff comparisons across the entire call is
        bounded by R. There is no backtracking, so the inner and outer loops do not
        multiply; they account for the same total work.

        - Per call on one suffix (length R <= M): Time O(R)
        - For all suffixes of one song with length L <= M:
            sigma{r=1}^{L-1} O(r) = O(L^2) = O(M^2)
        - Across all N songs: O(N*M^2) time for all insertions in the constructor.
        - Persistent structure size (nodes/edges over all songs) is O(NM),
          since total diffs are sigma(|s_i|−1) = O(NM), and the compressed suffix trie
          remains linear in input size up to constant factors (51-way child arrays).

        Detailed Space Complexity:
        - Per call: O(1) extra space.
            - Uses only a few local variables (node, cursor, depth, etc.) and no recursion.
            - May create at most one new edge and one new node per call (or one split with
              O(1) new structures), each constant in size due to fixed 51-slot child arrays.
        - Across all suffix insertions:
            - Each insertion adds O(1) new nodes/edges.
            - Total persistent space across all songs is O(NM), proportional to the total
              number of diffs (sigma(|s_i|-1) <= NM).

        Time Complexity: O(M)
        Space Complexity: O(NM)

        """
        node = root
        cursor = suffix_start_ptr
        depth = 1
        song_end = self.song_boundaries[song_id][1]
        # [Example: song 1 start=5 -> cursor=5, song_end=8, suffix=[2,2,-4]]

        # For example, inserting the first suffix of song 0 ("cegec"):
        # song_id = 0 -> song_end = 4 (its diffs occupy indices [0,4))
        # suffix_start_ptr = 0 -> current suffix = [2, 2, -2, -2]

        while cursor < song_end:  # O(R) iterations per call (worst-case) [Per call on one suffix of length R]
            # Update at current node before descending
            self.update_node_stats(node, song_id, suffix_start_ptr, depth)

            diff = self.global_diffs[cursor]  # take current diff value (e.g. +2)
            idx = diff + 25  # map diff range [-25..25] -> [0..50], e.g. 2 -> 27

            edge = node.children[idx]
            if edge is None:
                # No edge: create one consuming all remaining diffs of this song
                new_node = SuffixTrie()
                new_edge_end = song_end
                node.children[idx] = node.suffix_edge(cursor, new_edge_end, new_node)

                # Example for first insertion (song0, suffix0):
                # new edge label -> global_diffs[0:4] = [2,2,-2,-2]
                # connects root -> new_node

                self.update_node_stats(new_node, song_id, suffix_start_ptr, depth + (new_edge_end - cursor))
                return

            # Edge exists: match along it (bounded by song_end)
            edge_len = edge[1] - edge[0]
            match_len = 0
            while match_len < edge_len and cursor + match_len < song_end:
                if self.global_diffs[cursor + match_len] != self.global_diffs[edge[0] + match_len]:
                    break
                match_len += 1

            # Example detailed trace for song1,start=5 against edge [1:4]=[2,-2,-2]:
            # compare 2 vs 2 -> match_len=1
            # compare -4 vs -2 -> mismatch -> exit with match_len=1 (< edge_len=3)]

            if match_len < edge_len:
                # Partial match - split the edge
                old_target_node = edge[2]
                intermediate_node = SuffixTrie()

                # Maintain stats through the newly created intermediate node
                intermediate_node.frequency = old_target_node.frequency
                intermediate_node.latest_song_id = old_target_node.latest_song_id

                original_end = edge[1]
                edge[1] = edge[0] + match_len
                edge[2] = intermediate_node

                # Remainder of old edge
                split_diff = self.global_diffs[edge[0] + match_len]
                split_idx = split_diff + 25  # map to child index via idx = diff + 25 (shifts [-25..25] -> [0..50])
                intermediate_node.children[split_idx] = intermediate_node.suffix_edge(edge[0] + match_len, original_end,
                                                                                      old_target_node)

                # If our suffix ends exactly at the split point, the method stop here after updating stats
                if cursor + match_len >= song_end:
                    self.update_node_stats(intermediate_node, song_id, suffix_start_ptr, depth + match_len)
                    return

                # Otherwise, create new edge for the remainder of our current suffix
                new_diff = self.global_diffs[cursor + match_len]
                new_idx = new_diff + 25
                new_node = SuffixTrie()
                new_edge_end = song_end
                intermediate_node.children[new_idx] = intermediate_node.suffix_edge(cursor + match_len, new_edge_end,
                                                                                    new_node)

                self.update_node_stats(intermediate_node, song_id, suffix_start_ptr, depth + match_len)
                self.update_node_stats(new_node, song_id, suffix_start_ptr, depth + (new_edge_end - cursor))
                return

            # Full match of this edge: advance
            cursor = cursor + match_len
            depth = depth + match_len
            node = edge[2]

        # Landed exactly at a node (cursor == song_end)
        self.update_node_stats(node, song_id, suffix_start_ptr, depth)

    def update_node_stats(self, node, song_id, suffix_start_ptr, depth_K):
        """
         Update per-node statistics during suffix insertion:
          - increment distinct-song frequency once per song, and
          - maintain the best (most-frequent) pattern pointer for length K.

        Approach:
        - Implement a distinct-song counting rule using the per-node field 'latest_song_id'.
          Each node's frequency reflects the number of unique songs in which its pattern
          occurs. When the current song ID differs from 'latest_song_id', the method updates
          'latest_song_id = song_id' and increments 'frequency' by one.
        - Let K = depth_K (the current matched pattern length, measured in notes/characters).
          For 2 <= depth_K <= M, the method compares this node's 'frequency' with the global
          maximum 'best_freq[depth_K]'. If it exceeds the existing best, the node becomes the new
          representative pattern of length K. The method records a reconstruction pointer
          in 'precalculated_patterns[depth_K]', enabling later retrieval of the corresponding motif.

        Time Complexity: O(1)
        Space Complexity: O(1)

        """
        if node.latest_song_id != song_id:
            node.latest_song_id = song_id
            node.frequency += 1

        if 2 <= depth_K <= self.max_song_len:
            # Update only when strictly better
            if node.frequency > self.best_freq[depth_K]:
                self.best_freq[depth_K] = node.frequency
                original_char_start = self.find_original_start(song_id, suffix_start_ptr)  # O(1)
                if original_char_start:
                    node.original_pattern_info = original_char_start
                    self.precalculated_patterns[depth_K] = node.original_pattern_info

        # Example end-state:
        #   nodes for [+2] and [+2,+2] are visited from songs 0,1,2 -> frequency becomes 3
        #   best_freq[2] = 3 ; best_freq[3] = 3
        #   precalculated_patterns[2,3,4] eventually point to (1,0) giving 'g','d','f','h','d'

    def find_original_start(self, song_id, diff_ptr):
        """
        Map a pointer into the global difference array back to the original
        (song_id, char_index).

        Approach:
        - Each song's differences occupy a contiguous interval
          [start, end] inside 'global_diffs', recorded in 'song_boundaries[song_id]'.
        - Given a diff pointer 'diff_ptr' within that interval, the corresponding
          starting character index in the original song is simply:
              char_index = diff_ptr - start
         Inputs:
        - song_id (int): Identifier of the song.
        - diff_ptr (int): Index into 'global_diffs' that lies within this song's
          diff interval '[start, end]'.

         Time & Space Complexity:
        - Time: O(1) - one table lookup and one subtraction.
        - Extra Space: O(1) - no allocations.
        """

        start, _ = self.song_boundaries[song_id]  # Example song 1 -> start=4
        char_idx = diff_ptr - start  # Example diff_ptr=4 -> char_idx=0
        return song_id, char_idx  # Example -> (1,0)

    def collect_pattern(self, song_id, start, K):
        """
        Reconstruct a K-length motif from the original song as a list of chars.

        Args:
            song_id (int): ID of the source song.
            start (int): Starting character index in that song.
            K (int): Pattern length (number of characters).

        Returns:
            list[str]: The K characters s[start : start+K].

            Time complexity: O(K)  (copy K characters)
            Space complexity: O(K) (output list)
        """
        s = self.seqs[song_id]  # Example -> "gdfhd"
        out = []

        for i in range(K):  # O(K)
            note = s[start + i]  # get note at index  Example K=3 -> 'd','f','h'
            out.append(note)  # append note to result

        return out  # Example returns ['d','f','h'] for K=3

