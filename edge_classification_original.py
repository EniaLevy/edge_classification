import pybresenham as bres
import cv2
import math
import numpy as np
import glob
import re

image_list = []
image_names = []
for filename in glob.glob('selected\*.png'): #assuming gif
    f = re.search(r'\d+', filename).group(0)
    im=cv2.imread(filename)
    image_list.append(im)
    image_names.append(f)

'''
filename = "testimg8.jpg"
color_img = cv2.imread(filename) #modify canny if real img
'''
synthetic = True
small = 1 #higher value for smaller images

for image_counter, color_img in enumerate(image_list):
    [h, w] = color_img.shape[0:2]

    #change as required. Values used in paper: Td=10, angle = 15, r=30
    distance_threshold = 10
    angle_threshold = 15 #check to see if a branch should be considered parallel
    angle2_threshold = angle_threshold*1.5 #minimum angle necessary between branches
    range_threshold = round(30/small)
    minimum_line_length = range_threshold/3
    percent_kept = 0.8 #upper percent of junctions to be kept (based on scale)

    if synthetic:
        cannyth = 90
        rad_thresh = 0.7
        percent_kept = 1 #upper percent of junctions to be kept (based on scale)
        tag_max = 3
        minimum_line_length = 0
    else:
        cannyth = 255
        rad_thresh = 0.8
        tag_max = 2

    img = cv2.cvtColor(color_img, cv2.COLOR_BGR2GRAY)
    height, width, channels = color_img.shape
    edges = cv2.Canny(img, cannyth/3, cannyth)
    lsd = cv2.createLineSegmentDetector(0)
    detected_lines = lsd.detect(img)[0] #Position 0 of the returned tuple are the detected lines

    drawn_img = lsd.drawSegments(img,detected_lines)

    class Line:
        id_num = 0
        def __init__(self, edgepoints, calculate_direction = False):
            Line.id_num += 1
            self.id = Line.id_num #check if this works
            self.edge_points = edgepoints
            self.a1 = edgepoints[0]
            self.a2 = edgepoints[1]
            self.x1 = self.a1[0]
            self.x2 = self.a2[0]
            self.y1 = self.a1[1]
            self.y2 = self.a2[1]
            self.m = self.slope()
            self.b = self.y_intercept()
            self.length = self.lineLength()

            self.direction = 0
            if calculate_direction:
                self.calculateDirection()

            self.angles = []

        def __eq__(self, other):
            return self.id==other.id
        
        def __hash__(self):
            return hash(('id', self.id))
        
        def slope(self):
            if self.x1 == self.x2: #if it is a vertical line return a very large number to simulate an almost vertical line
                #return math.inf
                return 99999999
            return (self.y2 - self.y1) / (self.x2 - self.x1)
        
        def y_intercept(self):
            return self.y1 - self.m * self.x1
        
        def lineLength(self):
            return math.dist(self.a1, self.a2)

        def angle_trunc(self, a):
            while a < 0.0:
                a += math.pi * 2
            return a

        def calculateDirection(self):
            x_orig = self.x1
            y_orig = self.y1
            x_landmark = self.x2
            y_landmark = self.y2
            deltaY = y_landmark - y_orig
            deltaX = x_landmark - x_orig
            self.direction = math.degrees(self.angle_trunc(math.atan2(deltaY, deltaX)))


        def pointIsOnLine(self, point, close = False, segment = True, dis_thres = 0): #check if point is on specific line SEGMENT
            x = point[0]
            y = point[1]

            if not close:
                condition = math.isclose(y,((self.m * x) + self.b))
            else:
                condition = self.distanceFromPointToLineSegment(point, segment=False) <= dis_thres #threshold for distance to line
                #condition = (y-((self.m * x) + self.b)) < 1
                #print(abs(y-((self.m * x) + self.b)))
            
            if condition:
                if segment:
                    pt3_between = self.pointIsOnSegment(point)
                    if pt3_between:
                        return True
                else:
                    return True
            return False
        
        def pointIsOnSegment(self, point):
            x = point[0]
            y = point[1]
            check = (min(self.x1, self.x2) <= x <= max(self.x1, self.x2)) and (min(self.y1, self.y2) <= y <= max(self.y1, self.y2))
            return check

        def distanceFromPointToLineSegment(self, point, segment = True):
            p4 = self.closestPointOnLine(point)
            check = self.pointIsOnSegment(p4)
            if check or not segment:
                distance = math.dist(point, p4)
            else:
                distance = self.distanceFromEdgepoints(point)
            return distance
        
        def distanceFromEdgepoints(self, point):
            return min(math.dist(point, self.a1), math.dist(point, self.a2))

        def closestPointOnLine(self, p3):
                (x1, y1), (x2, y2), (x3, y3) = self.a1, self.a2, p3
                dx, dy = x2-x1, y2-y1
                det = dx*dx + dy*dy
                a = (dy*(y3-y1)+dx*(x3-x1))/det
                return x1+a*dx, y1+a*dy

        
    class Junction:
        junct_id = 0
        def __init__(self, coords, centr = False):
            Junction.junct_id += 1
            self.id = Junction.junct_id#check if this works

            self.coordinates = coords
            self.x = coords[0]
            self.y = coords[1]
            self.centroid = centr

            self.junction_radius = 1
            self.type = None
            self.og_r = 0
            self.r = 0
            self.junction_circumference = []
            self.branches = []
            self.branch_lines = []
            self.branch_parallel_index = []
            self.spine_branches = []
            self.line_segments = []
            self.line_distances = []
            self.search_ranges = []
            self.circumference = []
            self.belonging_distance = None
            self.orientation = None
        def __eq__(self, other):
            return math.isclose(self.x, other.x) and math.isclose(self.y, other.y)
        
        def __hash__(self):
            return hash(('coordinates', self.coordinates))
        
        def addLineSeg(self, line_segment): #add a line segment associated to the junction - they could be more than 2
            if line_segment not in self.line_segments:
                self.line_segments.append(line_segment)
                self.line_distances.append(self.calculateBelongingDistance(line_segment))
        
        def addBranch(self, branch):
            if branch not in self.branches:
                self.branches.append(branch)
                self.addBranchLine([branch[0], branch[-1]])
        
        def addBranchLine(self, branch_edges):
            self.branch_lines.append(Line(branch_edges, True))
        
        def getY(self, x):
            return (self.m*x+self.b)
        
        def sortBranches(self):
            self.branches = [x for _, x in sorted(zip(self.branch_lines, self.branches), key=lambda x: x[0].direction)]
            self.branch_lines.sort(key=lambda x: x.direction)
        
        def calculateSearchRanges(self): #check the radius of the circle for searching
            for line in self.line_segments:
                r = 0
                if line.pointIsOnLine(self.coordinates): #never goes in here checkkk
                    r = line.distanceFromPointToLineSegment(self.coordinates)
                    r = max(distance_threshold, r)
                else:
                    closest_point = line.closestPointOnLine(self.coordinates)
                    r = line.distanceFromEdgepoints(closest_point)
                    if r < distance_threshold:
                        r = line.length
                r = int(r)
                self.search_ranges.append(r)
        
        def calculateCircumference(self):
            self.og_r = min(self.search_ranges)
            radius = min(self.og_r, range_threshold)
            radius = radius + self.junction_radius
            self.r = radius
            circle = list(bres.circle(round(self.x), round(self.y), round(self.r)))
            circumference = []
            for point in circle:
                try:
                    for neighbor in getPixelNeighborhood(point):
                        if edges[neighbor[1]][neighbor[0]]>0:
                            circumference.append(neighbor)
                except IndexError:
                    pass
            self.circumference = self.sortPoints(np.asarray(circumference))
        
        def checkJunctionCircumference(self):
            junction_circumference = list(bres.circle(round(self.x), round(self.y), round(self.junction_radius)))
            for point in junction_circumference:
                try:
                    if edges[point[1]][point[0]]>0:
                        self.junction_circumference.append(point)
                except IndexError:
                    pass
        
        def sortPoints(self, xy: np.ndarray) -> np.ndarray:
            # normalize data  [-1, 1]
            xy_sort = np.empty_like(xy)
            xy_sort[:, 0] = 2 * (xy[:, 0] - np.min(xy[:, 0]))/(np.max(xy[:, 0] - np.min(xy[:, 0]))) - 1
            xy_sort[:, 1] = 2 * (xy[:, 1] - np.min(xy[:, 1])) / (np.max(xy[:, 1] - np.min(xy[:, 1]))) - 1

            # get sort result
            sort_array = np.arctan2(xy_sort[:, 0], xy_sort[:, 1])
            sort_result = np.argsort(sort_array)

            return xy[sort_result].tolist()

        def getLinesAndDistance(self):
            return list(zip(self.line_segments, self.line_distances))
        
        def calculateBelongingDistance(self, line):
            distance = 0

            if not line.pointIsOnLine(self.coordinates):
                distance = line.distanceFromPointToLineSegment(self.coordinates)

            return distance
        
        def turnInt(self):
            self.coordinates = [round(self.x), round(self.y)]
            self.x = round(self.x)
            self.y = round(self.y)

    class StraightEdge:
        def __init__(self):
            self.line_segments = []
            self.junctions = []
            self.junction_types = []
            self.type = None
            self.branch_types = []

            self.ob_score = 0.02
            self.rc_score = 0
            self.sc_score = 0.01

        def addLineSeg(self, seg):
            self.line_segments.append(seg)
        
        def addJunction(self, j):
            self.junctions.append(j)
            self.junction_types.append(j.type)

        def calculateScore(self): #check weights
            types = self.junction_types+self.branch_types
            for t in types:
                match t:
                    case "l":
                        self.ob_score += 1
                        self.sc_score += 0.8
                    case "ts":
                        self.ob_score += 1.5 #higher weight
                    case "tb":
                        self.ob_score += 0.8 #lower weight
                        self.rc_score += 0.8 #lower weight
                    case "ps":
                        self.sc_score += 1
                    case "pb":
                        self.rc_score += 1
                    case "y":
                        self.sc_score += 1.3 #higher weight
                    case "ys":
                        self.sc_score += 1
                    case "yb":
                        self.ob_score += 1
                    case "x":
                        self.rc_score += 0.8 #lower weight
        
        def checkType(self):
            if self.ob_score > self.rc_score and self.ob_score > self.sc_score:
                if self.ob_score > 0.1:
                    self.type = "ob"
            elif self.rc_score > self.ob_score and self.rc_score > self.sc_score:
                self.type = "rc"
            elif self.sc_score > self.ob_score and self.sc_score > self.rc_score:
                self.type = "sc"

    def line_intersect(m1, b1, m2, b2):
        if m1 == m2:
            return None

        x = (b2 - b1) / (m1 - m2)
        y = m1 * x + b1
        return x,y

    def calculateAngle(line1, line2):
        s1 = line1.m
        s2 = line2.m 
        if s1 == s2:
            return 0
        try:
            return abs(math.degrees(math.atan((s2-s1)/(1+(s2*s1))))) #using slopes
        except ZeroDivisionError:
            angle = abs(1+(s1*s2))
            angle /=math.sqrt(1+math.pow(s1,2))*math.sqrt(1+math.pow(s2,2))
            angle = math.acos(angle)
            return angle

    def checkIfParallel(l1,l2):
        ang = calculateAngle(l1, l2)
        if ang < angle_threshold or abs(ang-180) < angle_threshold:
            return True
        else:
            return False

    def checkBranchParallels(junct):
        junct.sortBranches()
        count = 0
        for i in range(len(junct.branch_lines)):
            for j in range(i+1, len(junct.branch_lines)):
                if checkIfParallel(junct.branch_lines[i], junct.branch_lines[j]):
                    count += 1
                    junct.branch_parallel_index.append([i,j])
        if count == 1:
            junct.spine_branches.extend(junct.branch_parallel_index[0])
        return count

    def getPixelNeighborhood(middle, size = 1):
        points = list(range(-size,size+1))
        neighborhood = []
        for x in points:
            for y in points:
                neighborhood.append([middle[0]+x, middle[1]+y])
        return neighborhood
            

    def search_region(branch_candidates, tag_check = True):
        chain = []
        tags = {0}
        for branch_point in branch_candidates:
            try:
                if not tag_check:
                    neighborhood = getPixelNeighborhood(branch_point, size=2)
                else:
                    neighborhood = getPixelNeighborhood(branch_point)
                flag = False
                for pixel in neighborhood:
                    flag = edges[pixel[1]][pixel[0]] > 0
                    if flag:
                        if branch_point not in branch_candidates[3:-3] and tag_check:
                            tags.add(line_id_map[branch_point[1]][branch_point[0]])
                        if pixel == branch_point:
                            chain.append(branch_point)
                        break
            except IndexError:
                flag = False
            if len(tags) > tag_max:
                chain = []
                return chain #checking if they belong to the same line segment
            if flag:
                chain.append(branch_point)
        return chain

    def popItems(popindeces, l):
        thislist = l.copy()
        popindeces = list(set(popindeces))
        for index in sorted(popindeces, reverse=True):
            thislist.pop(index)
        return thislist

    detected_lines = np.squeeze(detected_lines)
    detected_lines = np.resize(detected_lines, (detected_lines.shape[0],2,2))
    lines = []

    line_id_map = np.zeros((h, w), dtype=np.int32)

    for potential_line in detected_lines:
        line = Line(potential_line)
        if line.length < minimum_line_length:
            continue
        else:
            points = list(bres.line(int(line.x1), int(line.y1), int(line.x2), int(line.y2)))
            for point in points:
                try:
                    line_id_map[point[1]][point[0]]=line.id
                except IndexError:
                    pass
            lines.append(line)

    junction_candidates = []

    for i in range(len(lines)):
        for j in range(i+1, len(lines)):
            ang = calculateAngle(lines[i], lines[j])

            if ang < angle_threshold or abs(ang-180) < angle_threshold:
                continue
            
            candidate = Junction(line_intersect(lines[i].m, lines[i].b, lines[j].m, lines[j].b))

            if junction_candidates and candidate in junction_candidates:
                junction_candidates[junction_candidates.index(candidate)].addLineSeg(lines[i])
                junction_candidates[junction_candidates.index(candidate)].addLineSeg(lines[j])
                continue
            
            d1 = candidate.calculateBelongingDistance(lines[i])
            d2 = candidate.calculateBelongingDistance(lines[j])

            if candidate.x>=width or candidate.x<=0 or candidate.y>=height or candidate.y<=0 or candidate is None:
                continue

            if d1 <= distance_threshold and d2 <= distance_threshold:
                for neighbor in getPixelNeighborhood(candidate.coordinates): #also set whole pixel neighborhood as a potential junction (to see if it is closer to other lines)
                    try:
                        if edges[round(neighbor[1])][round(neighbor[0])] == 0:
                            continue
                        if Junction(neighbor) == candidate:
                            junction = Junction(neighbor, True)
                        else:
                            junction = Junction(neighbor)
                        junction.addLineSeg(lines[i])
                        junction.addLineSeg(lines[j])
                        junction_candidates.append(junction)
                    except IndexError:
                        continue

    def branchez (candidate):
        candidate.calculateSearchRanges()
        candidate.calculateCircumference()
        candidate.checkJunctionCircumference()
        candidate.turnInt()

        for point in candidate.circumference:
            if not candidate.junction_circumference:
                continue
            d = math.inf
            for coords in candidate.junction_circumference:
                distance = math.dist(coords, point)
                if distance < d:
                    d = distance
                    junction_point = coords

            branch_edgepoints = [[junction_point[0], junction_point[1]],[point[0], point[1]]]

            branch_candidate = list(bres.line(branch_edgepoints[0][0], branch_edgepoints[0][1], branch_edgepoints[1][0], branch_edgepoints[1][1]))
            chain = search_region(branch_candidate)
            if len(chain) > ((rad_thresh)*(candidate.r-candidate.junction_radius)): #chec if thtey belong to a single line segment?; changing this to 0.8 gives way less hits
                candidate.addBranch(chain)

    visited = []
    edge_objects = []

    for i, line in enumerate(lines):
        if line in visited:
            continue
        search = [line]
        line_group = []

        while search:
            search_line = search.pop(0)
            if search_line in line_group:
                continue
            line_group.append(search_line)
            visited.append(search_line)
            for j in range(i+1, len(lines)):
                ang = calculateAngle(search_line, lines[j])
                if ang < angle_threshold/2 or abs(ang-180) < angle_threshold/2:
                    a1_d=0
                    a2_d=0
                    if not search_line.pointIsOnLine(lines[j].a1):
                        a1_d = search_line.distanceFromPointToLineSegment(lines[j].a1)
                    if not search_line.pointIsOnLine(lines[j].a2):
                        a2_d = search_line.distanceFromPointToLineSegment(lines[j].a2)
                    if a1_d <=distance_threshold/2 or a2_d <= distance_threshold/2: #edge distance threshold
                        search.append(lines[j])
        e = StraightEdge()
        for l in line_group:
            e.addLineSeg(l)
        edge_objects.append(e)

    pop_index = []
    visited = []
    add=[]

    for i, candidate in enumerate(junction_candidates):
        if candidate in visited:
            continue
        search = [candidate]
        close_junctions = []
        junction_lines = []

        while search:
            junction = search.pop(0)
            if junction in close_junctions:
                continue
            close_junctions.append(junction)
            visited.append(junction)
            for j in range(i+1, len(junction_candidates)):
                if math.dist(junction.coordinates, junction_candidates[j].coordinates) <= distance_threshold/2:
                    search.append(junction_candidates[j])
        
        if len(close_junctions) <= 9:
            count = 0
            for neighbor in close_junctions:
                if neighbor.centroid:
                    neighbor.junction_radius = 2
                    branchez(neighbor)
                    count +=1
                else:
                    pop_index.append(junction_candidates.index(neighbor))
            if count > 0:
                continue

        for neighbor in close_junctions:
            junction_lines.extend(neighbor.line_segments)
        
        junction_lines=list(set(junction_lines))

        d = 0

        positions = []

        for neighbor in close_junctions:
            positions.append(neighbor.coordinates)
            pop_index.append(junction_candidates.index(neighbor))
        
        positions = np.asarray(positions)
        junct = Junction(positions.mean(axis=0))
        
        for neighbor in close_junctions:
            dis = math.dist(neighbor.coordinates, junct.coordinates)
            if dis > d:
                d = dis
        d = math.ceil(d)
        junct.junction_radius = d

        for line in junction_lines:
            junct.addLineSeg(line)

        branchez(junct)
        if junct.junction_radius > (junct.r-junct.junction_radius):
            continue
        add.append(junct)

    junction_candidates = popItems(pop_index, junction_candidates)
    junction_candidates.extend(add)

    pop_index = []
    junction_candidates.sort(key=lambda x: x.og_r, reverse=True)
    #junction_candidates = junction_candidates[int(len(junction_candidates)*0.8):int(len(junction_candidates)*percent_kept)]
    junction_candidates = junction_candidates[:int(len(junction_candidates)*percent_kept)]


    pop_index = []
    for index, junction in enumerate(junction_candidates):
        if len(junction.branches) < 2:
            pop_index.append(index)
        else:
            pop_index_branches = []
            add = []
            visited = []
            for index_line, line1 in enumerate(junction.branch_lines):
                if line1 in visited:
                    continue
                search = [line1]
                #add.append(junction.branches[index_line])
                close_lines = []
                close_branches = []
                close_branchez = []
                branch_lengths = []
                while search:
                    line = search.pop(0)
                    if line in close_lines:
                        continue
                    close_lines.append(line)
                    pop_index_branches.append(junction.branch_lines.index(line))
                    close_branches.extend(junction.branches[junction.branch_lines.index(line)])
                    close_branchez.append(junction.branches[junction.branch_lines.index(line)])
                    branch_lengths.append(len(junction.branches[junction.branch_lines.index(line)]))
                    visited.append(line)
                    for j in range(index_line+1, len(junction.branch_lines)):
                        if abs(line.direction - junction.branch_lines[j].direction) <= angle2_threshold:
                            search.append(junction.branch_lines[j])
                """
                x, y = np.array([i for i, j in close_branches]), np.array([j for i, j in close_branches])
                fit_line = np.linalg.lstsq(np.vstack([x, np.ones(len(x))]).T, y, rcond=None)[0]

                x1 = 1
                y1 = fit_line[0]*x1+fit_line[1]
                x2 = 100
                y2 = fit_line[0]*x2+fit_line[1]

                fit_line = Line([[x1,y1],[x2,y2]])
                d = math.inf
                closest_index = None
                for index_branch, branch in enumerate(close_branchez):
                    if fit_line.distanceFromPointToLineSegment(branch[-1], False) < d:
                        d = fit_line.distanceFromPointToLineSegment(branch[-1], False)
                        closest_index = junction.branches.index(branch)
                
                add.append(junction.branches[closest_index])
                """
                for index_branch, branch in enumerate(close_branchez):
                    if len(branch) == max(branch_lengths):
                        add.append(branch)
                        break
            # add.append(junction.branches[junction.branches.index(maxclose_branches)])
            
                
            junction.branches = popItems(pop_index_branches, junction.branches)
            junction.branch_lines = popItems(pop_index_branches, junction.branch_lines)
            for branch in add:
                junction.addBranch(branch)
            if len(junction.branches)>4 or len(junction.branches) < 2: #getting rid of all junctions which cannot be classified - might be better to keep them
                pop_index.append(index)


    junction_candidates = popItems(pop_index, junction_candidates)

    junction_final = junction_candidates

    #########################################################　JUNCTION CLASSIFICATION


    pop_index = []

    junctions_l =[]
    junctions_t = []
    junctions_y = []
    junctions_x = []
    junctions_p = []

    for index, junction in enumerate(junction_final):
        if len(junction.branches) == 2:
            if checkIfParallel(junction.branch_lines[0], junction.branch_lines[1]):
                pop_index.append(index)
            else:
                junction.type = "l"
                junctions_l.append(junction)
        elif len(junction.branches) == 3:
            parallel_count = checkBranchParallels(junction)
            if parallel_count > 0:
                junction.type = "t"
                junctions_t.append(junction)
            else:
                junction.type = "y"
                junctions_y.append(junction)
        elif len(junction.branches) == 4:

            parallel_count = checkBranchParallels(junction)

            if parallel_count == 1:
                i, j = junction.branch_parallel_index[0] 
                if i+1 == j or (i == 0 and j == 3):
                    junction.type = "k" #k junctions are not very meaningful
                    pop_index.append(index)
                else :
                    junction.type = "p"
                    junctions_p.append(junction)
            else:
                junction.type = "x"
                junctions_x.append(junction)
    junction_final = popItems(pop_index, junction_final)

    ######################################################### FINDING JUNCTION ORIENTATION (ONLY L, T, INVERSE-Y & PHI)

    for junction in junctions_t:
        for index, branch in enumerate(junction.branches):
            if index not in junction.branch_parallel_index[0]:
                dominant_branch = branch
                break
        junction.orientation = dominant_branch[-1] #orientation vector head

    def branchToVec(b1):
        a1 = np.array([b1[-1][0], b1[-1][1]])
        a2 = np.array([b1[0][0], b1[0][1]])
        #print(a1)
        #print(a2)
        return a2-a1
    def branchAddition(b1,b2):
        v1 = branchToVec(b1)
        v2 = branchToVec(b2)
        #print(v1+v2)
        return v1+v2
    def magVec(v, mag):
        unit_vector = v/np.linalg.norm(v)
        return unit_vector*mag*-1
    def ogCoords(v, coords):
        new_v = np.around(v+coords)
        final_v = new_v.astype(int).flatten().tolist()
        return final_v

    for junction in junctions_l:
        new_vector = branchAddition(junction.branches[0], junction.branches[1])
        new_vector = magVec(new_vector, junction.r)
        junction.orientation = ogCoords(new_vector, junction.coordinates)

    for junction in junctions_p:
        index = junction.branch_parallel_index[0][0]
        if junction.branch_lines[index+1].direction - junction.branch_lines[index].direction < 90:
            junction.orientation = junction.branches[index][-1]
        else:
            junction.orientation = junction.branches[junction.branch_parallel_index[0][1]][-1]

    for junction in junctions_y:
        for i in range(len(junction.branch_lines)):
            count = 0
            for j in range(len(junction.branch_lines)):
                if j == i:
                    continue
                if abs(junction.branch_lines[i].direction-junction.branch_lines[j].direction) < 90:
                    count += 1
                    if count >= 2:
                        junction.type = "iy"
                        junction.orientation = junction.branches[i][-1]
                        junction.spine_branches.append(i)

    ######################################################### JUNCTION ON STRAIGHT LINE ANALISIS
    edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

    predicted = np.zeros((h, w), dtype=np.uint8)
    final_output = np.copy(edges)

    for e in edge_objects:
        for junction in junction_final:
            for line in junction.line_segments:
                if line in e.line_segments:
                    e.addJunction(junction)
                    if junction.spine_branches:
                        spine_check = False
                        for branch_index in junction.spine_branches:
                            if checkIfParallel(junction.branch_lines[branch_index], line):
                                spine_check = True
                                break
                        if spine_check:
                            match junction.type:
                                case "t":
                                    e.branch_types.append("ts")
                                case "p":
                                    e.branch_types.append("ps")
                                case "iy":
                                    e.branch_types.append("ys")
                        else:
                            match junction.type:
                                case "t":
                                    e.branch_types.append("tb")
                                case "p":
                                    e.branch_types.append("pb")
                                case "iy":
                                    e.branch_types.append("yb")
                    break
        e.calculateScore()
        e.checkType()

        class_label = 0
        
        match e.type:
            case "ob":
                color = (255,0,0)
                class_label = 1
            case "sc":
                color = (0,255,0)
                class_label = 2
            case "rc":
                color = (0,0,255)
                class_label = 3
            case _:
                color = (255,255,255)

        for line_seg in e.line_segments:
            final_output = cv2.line(final_output, [round(line_seg.x1),round(line_seg.y1)], [round(line_seg.x2),round(line_seg.y2)],color=(int(color[0]),int(color[1]),int(color[2])), thickness=2)
            hits = list(bres.line(round(line_seg.x1),round(line_seg.y1),round(line_seg.x2),round(line_seg.y2)))
            for point in hits:
                try:
                    predicted[point[1]][point[0]] = class_label
                except IndexError:
                    continue
        """
        j_types = set(e.junction_types)

        if "l" in j_types and "t" in j_types:
            e.type = "ob"
            color = (255,0,0)
        elif "p" in j_types and "y" in j_types:
            e.type = "sc"
            color = (0,255,0)
        elif "p" in j_types and "t" in j_types:
            e.type = "rc"
            color = (0,0,255)
        else:
            color = (255,255,255)
        """
    #########################################################　VISUALIZATION

    for line in lines:
        color = list(np.random.choice(range(256), size=3))
        color_img = cv2.line(color_img, [int(line.x1),int(line.y1)], [int(line.x2),int(line.y2)], color=(int(color[0]),int(color[1]),int(color[2])), thickness=1)

    for candidate in junction_final:
        color_img = cv2.circle(color_img, candidate.coordinates, radius=candidate.r, color=(0, 255, 0), thickness=1)
        color_img = cv2.circle(color_img, candidate.coordinates, radius=candidate.junction_radius, color=(0, 255, 0), thickness=1)
        for line in candidate.branch_lines:
            if candidate.type == "l":
                color = (255, 0, 0)
            elif candidate.type == "t":
                color = (255, 255, 0)
            elif candidate.type == "y" or candidate.type == "iy":
                color = (0, 255, 255)
            elif candidate.type == "x":
                color = (255, 0, 255)
            else:
                color = (0, 0, 255)
            color_img = cv2.line(color_img, [int(line.x1),int(line.y1)], [int(line.x2),int(line.y2)], color=color, thickness=3)
        color_img = cv2.circle(color_img, candidate.coordinates, radius=1, color=(0, 0, 255), thickness=-1)
        
        if candidate.orientation:
            color_img = cv2.arrowedLine(color_img, candidate.coordinates, candidate.orientation, color=(0, 0, 0), thickness=2)
            pass

    dif = round(255-np.amax(line_id_map))
    if dif < 0:
        dif = 0
    line_id_map = (line_id_map+dif).astype(dtype='uint8')
    line_id_map[line_id_map==dif]=0
    line_id_map[line_id_map>0]=255
    cv2.imwrite("output/"+str(image_names[image_counter])+"_predictions.png", predicted)
    cv2.imwrite("output/"+str(image_names[image_counter])+"_junctions.png", color_img)
    cv2.imwrite("output/"+str(image_names[image_counter])+"_edges.png", edges)
    cv2.imwrite("output/"+str(image_names[image_counter])+"_linemap.png", line_id_map)
    cv2.imwrite("output/"+str(image_names[image_counter])+"_classified.png", final_output)
    """
    cv2.imshow("LSD",color_img )
    cv2.imshow("Edges", final_output)
    cv2.imshow("line id", line_id_map)
    cv2.waitKey(0)
    """