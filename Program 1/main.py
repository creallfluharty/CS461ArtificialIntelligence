from collections import defaultdict
import heapq


def load_graph(adjacencies, coordinates):
    adj = defaultdict(list)
    for line in adjacencies:
        city, *neighbors = line.split()
        for neighbor in neighbors:
            adj[city].append(neighbor)
            adj[neighbor].append(city)

    coords = {city: [float(c) for c in coords[::-1]] for city, *coords in (line.split() for line in coordinates)}

    return {
        city: {
            "neighbors": [neighbor for neighbor in adj[city] if neighbor in coords],
            "position": pos,
        }
        for city, pos in coords.items()
    }


def euclidean_distance(x1, y1, x2, y2):
    return ((x1 - x2)**2 + (y1 - y2)**2) ** 1/2  # assuming the earth is flat :P


def best_first_search(graph, source, destination):
    closed = {}  # key is city name, value is parent
    open_queue = [(0.0, source, None)]  # heap of (dist, city, parent)

    while open_queue and destination not in closed:
        dist, examining, parent = heapq.heappop(open_queue)
        if examining in closed:  # node may be added by multiple parents
            continue

        for neighbor in graph[examining]['neighbors']:
            if neighbor not in closed:  # no use adding them if they'll just be skipped anyway
                heuristic = euclidean_distance(
                    *graph[neighbor]["position"],
                    *graph[source]["position"],  # s/source/destination for greedy best-first
                )

                heapq.heappush(
                    open_queue,
                    (heuristic, neighbor, examining),
                )

        closed[examining] = parent

    path = [destination]

    while parent := closed.get(path[-1]):
        path.append(parent)

    return path[::-1]


def get_city_selection(graph, prompt, error_message):
    while (selection := input(prompt)) not in graph:
        print(error_message)

    return selection


def main():
    with open("Adjacencies.txt") as a, open("coordinates.txt") as c:
        graph = load_graph(a, c)

    sorry = "Sorry! That city is not in our database :("
    source = get_city_selection(graph, "Where would you like to navigate from? ", sorry)
    destination = get_city_selection(graph, "Where would you like to navigate to? ", sorry)
    # source = "Topeka"
    # destination = "Kiowa"

    path = best_first_search(graph, source, destination)
    print(f"Here's a path between {source} and {destination}:", " → ".join(path))


if __name__ == '__main__':
    main()
