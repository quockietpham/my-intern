from scipy.spatial import Delaunay
import numpy as np
from matplotlib import pyplot as plt

nodes = np.array([(0, 0, 0), (1, -0.5, 0), (1, 0.5, 0), (2, 0, 0), (1, 0, 0)])
elements = np.array([(0, 1, 2), (1, 3, 4), (2, 3, 4)])

x = []
y = []
z = []

# def detect_free_edges(elements, edges):
#     free_edges = list(range(len(edges)))
#     temp = []
#     for i in elements:
#         temp.append(i[0])
#         temp.append(i[1])
#         temp.append(i[2])
#     for i in range(len(edges)):
#         cout = 0
#         for j in temp:
#             if i == j:
#                 cout+=1
#             if cout >= 2:
#                 free_edges.pop(i)
#                 break
#     return free_edges


def point_in_polygon(point, polygon):
    """
    Determines whether a point lies within a polygon using the ray casting algorithm.

    Parameters:
    point (tuple): the (x, y) coordinates of the point to test
    polygon (list): a list of (x, y) tuples defining the vertices of the polygon

    Returns:
    bool: True if the point is inside the polygon, False otherwise
    """
    # Count the number of intersections with the polygon edges
    intersections = 0
    n = len(polygon)
    for i in range(n):
        p1 = polygon[i]
        p2 = polygon[(i + 1) % n]
        if point_on_line(point, p1, p2):
            # Point lies exactly on an edge of the polygon
            return True
        if point[1] > min(p1[1], p2[1]) and point[1] <= max(p1[1], p2[1]):
            if point[0] <= max(p1[0], p2[0]) and p1[1] != p2[1]:
                x_intersection = (point[1] - p1[1]) * (p2[0] - p1[0]) / (p2[1] - p1[1]) + p1[0]
                if p1[0] == p2[0] or point[0] <= x_intersection:
                    intersections += 1
    # Return True if the number of intersections is odd, False otherwise
    return intersections % 2 == 1


def point_on_line(point, line_start, line_end):
    """
    Determines whether a point lies exactly on a line segment.

    Parameters:
    point (tuple): the (x, y) coordinates of the point to test
    line_start (tuple): the (x, y) coordinates of the start point of the line segment
    line_end (tuple): the (x, y) coordinates of the end point of the line segment

    Returns:
    bool: True if the point lies exactly on the line segment, False otherwise
    """
    epsilon = 1e-6
    x1, y1 = line_start[:-1]
    x2, y2 = line_end[:-1]
    x, y = point[:-1]
    # Check if the point lies within the bounding box of the line segment
    if x < min(x1, x2) - epsilon or x > max(x1, x2) + epsilon:
        return False
    if y < min(y1, y2) - epsilon or y > max(y1, y2) + epsilon:
        return False
    # Check if the point lies on the line segment
    if abs((y2 - y1) * (x - x1) - (x2 - x1) * (y - y1)) > epsilon:
        return False
    return True

is_point_in_polygon = []
for i in range(len(elements)):
    result = []
    result.append(i)
    # is_point_in_polygon.append([])
    nodes1 = list(nodes[elements[i][0]])
    nodes2 = list(nodes[elements[i][1]])
    nodes3 = list(nodes[elements[i][2]])
    # print(nodes1, nodes2, nodes3)
    polygon = [nodes1, nodes2, nodes3]
    node_true = []
    for j in range(len(nodes)):
        point = list(nodes[j])
        if point == nodes1 or point == nodes2 or point == nodes3:
            continue
        else:
            if point_in_polygon(nodes[j], polygon):
                node_true.append(j)

    result.append(node_true)
    is_point_in_polygon.append(result)

print(is_point_in_polygon)

for i in range(len(is_point_in_polygon)):
    if(is_point_in_polygon[i][1] != []):
        print(is_point_in_polygon[i])





fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
for i in range(len(nodes)):
    x.append(nodes[i][0])
    y.append(nodes[i][1])
    z.append(nodes[i][2])

x = np.array(x)
y = np.array(y)
z = np.array(z)


# ve nodes
ax.scatter(x, y, z, c='r', marker='o')


# ve edges
# for i in range(len(elements)):
#
#     x = [nodes[edges[i][0]][0], nodes[edges[i][1]][0]]
#     y = [nodes[edges[i][0]][1], nodes[edges[i][1]][1]]
#     z = [nodes[edges[i][0]][2], nodes[edges[i][1]][2]]
#     ax.plot(x, y, z, "k--")

# ve surface
faces = np.array([[0, 1, 2]])
for i in range(len(elements)):
    x = []
    y = []
    z = []
    for k in range(3):
        x.append(nodes[elements[i][k]][0])
        y.append(nodes[elements[i][k]][1])
        z.append(nodes[elements[i][k]][2])
    ax.plot_trisurf(x, y, z, triangles=faces)

ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')
plt.show()

# Define the vertices of the triangular mesh

plane = list(elements[is_point_in_polygon[0][0]])
vertices = []
vertices.append(list(nodes[plane[0]])[:-1])
vertices.append(list(nodes[plane[1]])[:-1])
vertices.append(list(nodes[plane[2]])[:-1])
vertices.append(list(nodes[is_point_in_polygon[0][1][0]])[:-1])
vertices = np.array(vertices)

plane.append(is_point_in_polygon[0][1][0])
mapping = np.array(plane)


# Create the Delaunay triangulation of the vertices
tri = Delaunay(vertices)

# Plot the mesh
fig, ax = plt.subplots()

elements = list(elements)

for i in range(len(tri.simplices)):
    # (mapping[tri.simplices[i]])

    elements.append(mapping[tri.simplices[i]])

elements.pop(is_point_in_polygon[0][0])


ax.triplot(vertices[:,0], vertices[:,1], tri.simplices)


# # Plot the vertex indices
# for i, vertex in enumerate(vertices):
#     ax.text(vertex[0], vertex[1], str(i), ha='center', va='bottom')


# # Get the indices of the vertices that make up each triangle
# triangles = tri.simplices

# # Create a set to store the free edges
# free_edges = set()

# # Iterate over each triangle
# for i, triangle in enumerate(triangles):
#     # Iterate over each edge of the triangle
#     for j in range(3):
#         # Get the indices of the vertices that make up the edge
#         edge = (triangle[j], triangle[(j+1)%3])
#         # Iterate over each other triangle
#         for k, other_triangle in enumerate(triangles):
#             # Skip the current triangle
#             if k == i:
#                 continue
#             # Check if the other triangle shares the current edge
#             if set(edge).issubset(set(other_triangle)):
#                 # If the edge is shared, it is not a free edge
#                 break
#         else:
#             # If the edge is not shared by any other triangle, it is a free edge
#             free_edges.add(frozenset(edge))

# # Print the free edges
# for edge in free_edges:
#     print(f"Free edge: {list(edge)}")
# Show the plot
plt.show()

print(len(elements))
fig1 = plt.figure()
ax1 = fig1.add_subplot(111, projection='3d')
faces1 = np.array([[0, 1, 2]])
for i in range(len(elements)):
    x = []
    y = []
    z = []
    for k in range(3):
        x.append(nodes[elements[i][k]][0])
        y.append(nodes[elements[i][k]][1])
        z.append(nodes[elements[i][k]][2])
    ax1.plot_trisurf(x, y, z, triangles=faces1)

# ax1.set_xlabel('X Label')
# ax1.set_ylabel('Y Label')
# ax1.set_zlabel('Z Label')
ax1.scatter(x, y, z, c='r', marker='o')
plt.show()