point = [[1,2], [2,3], [1,-3]]
vertex = [2,2]
k = 2

def distance(p1, p2):
    sum = 0
    for x, y in zip(p1, p2):
        sum += (y-x)**2
    return sum
def closestDist(vert, pts, k):
    dists = [[i, distance(x, vert)] for i, x in enumerate(pts)]
    sortDist = sorted(dists, key = lambda x: x[1])
    final = []
    for i in range(0, k):
        final.append(point[sortDist[i][0]])
    return final
print(closestDist(vertex, point, k))