def find_connected_components(graph):
    visited = set()
    components = []

    def dfs(vertex, component):
        component.append(vertex)
        visited.add(vertex)
        for neighbor in graph[vertex]:
            if neighbor not in visited:
                dfs(neighbor, component)

    for vertex in graph:
        if vertex not in visited:
            component = []
            dfs(vertex, component)
            components.append(component)

    return components


def get_identified_pairs(distances):
    min_pairs = []
    min_dist = distances['dist'].max() * len(distances) + 1.0

    for i in range(len(distances)):
        start_with_row = i
        distances['used'] = False
        cur_pairs = []
        cur_dist = 0

        while len(distances[~distances['used']]) != 0:
            row = distances[~distances['used']].iloc[start_with_row]
            start_with_row = 0

            cur_pairs.append((row['id_1'], row['id_2']))
            cur_dist += row['dist']

            if cur_dist > min_dist:
                break

            distances.loc[distances['id_1'] == row['id_1'], 'used'] = True
            distances.loc[distances['id_2'] == row['id_2'], 'used'] = True

        if cur_dist < min_dist:
            min_dist = cur_dist
            min_pairs = cur_pairs.copy()

        distances.drop(columns=['used'])

    return min_pairs, min_dist
