import numpy as np
from scipy.odr import *
from math import atan, sin, cos, sqrt, inf, pi


class FeatureDetection:

    def __init__(self) -> None:
        self.laser_points = []
        self.number_points = 0
        self.adjacent_points_distance_threshold = 0.08
        self.vertical_distance_threshold = 0.05
        self.predicted_distance_threshold = 0.05
        self.minimum_starting_seed = 6
        self.minimum_points = 10
        self.minimum_line_length = 0.6
        self.line_params = None

    def range_bearing_conversion(self, polar_coordinates):
        distance = polar_coordinates[0]
        measurement_angle = polar_coordinates[1]
        angle = measurement_angle 
    
        return np.array([distance*cos(angle), distance*sin(angle)])
    
    def set_laser_points(self, laser_scan):
        self.laser_points = []
        measurement_angle = laser_scan.angle_min
        angle_increment = laser_scan.angle_increment

        for distance in laser_scan.ranges:
            if distance >= laser_scan.range_min and distance <= laser_scan.range_max:
                self.laser_points.append(self.range_bearing_conversion([distance, measurement_angle]))
            
            measurement_angle += angle_increment
        self.number_points = len(self.laser_points)

    def linear_func(self, p, x):
        m, b = p
        return m * x + b
    
    def odr_fit(self, laser_points):
        x = np.array([i[0] for i in laser_points])
        y = np.array([i[1] for i in laser_points])

        # Create a model for fitting
        linear_model = Model(self.linear_func)
        
        # Create a RealData object using our initiated data from above
        data = RealData(x, y)

        # Set up ODR with the model and data
        odr_model = ODR(data, linear_model, beta0=[0., 0.])

        # Run the regression.
        out = odr_model.run()
        m, b = out.beta
        return m, b

    def projection_point2line(self, point, m, b):
        x, y = point
        m2 = -1 / m
        c2 = y - m2 * x
        intersection_x = - (b - c2) / (m - m2)
        intersection_y = m2 * intersection_x + c2
        return [intersection_x, intersection_y]
    
    def distance_2points(self, point1, point2):
        return np.sqrt((point2[0] - point1[0]) ** 2 + (point2[1] - point1[1]) ** 2)

    def points_2line(self, point1, point2):
        m, b = 0, 0
        if point2[0] == point1[0]:
            pass
        else:
            m = (point2[1] - point1[1]) / (point2[0] - point1[0])
            b = point2[1] - m * point2[0]
        return m, b

    def predict_point(self, line_params, laser_point):
        m1, b1 = self.points_2line([0,0], laser_point)  #Laser is in the origin of the frame
        m2, b2 = line_params

        if m1 == m2:
            return inf, inf
        
        predx = (b2 - b1) / (m1 - m2)
        predy = m2 * predx + b2
        return predx, predy

    def vertical_distance_point_to_line(self, line_params, point):
        m, b = line_params
        x, y = point
        return abs(-m*x + y - b) / sqrt(1 + m**2)
       
    def seed_segment_detection(self, breakpoint_index):

        for start_seed_index in range(breakpoint_index, self.number_points - self.minimum_points):
            flag = True
            last_seed_index = start_seed_index + self.minimum_starting_seed
            seed_points = self.laser_points[start_seed_index:last_seed_index + 1]
            m, b = self.odr_fit(seed_points)

            index = 0            
            for point in seed_points:
                if index != 0:
                    distance = self.distance_2points(point, seed_points[index - 1])
                    if distance > self.adjacent_points_distance_threshold:
                        flag = False
                        break

                vertical_distance = self.vertical_distance_point_to_line([m, b], point)
                if vertical_distance > self.vertical_distance_threshold:
                    flag = False
                    break
                    
                predicted_point = self.predict_point([m, b], point)
                distance = self.distance_2points(point, predicted_point)
                if distance > self.predicted_distance_threshold:
                    flag = False
                    break
                
                index += 1
            if flag:
                self.line_params = [m, b]
                return [start_seed_index, last_seed_index]

        return False
    
    def seed_segment_growing(self, seed, breakpoint_index):
        line_eq = self.line_params

        number_points = len(self.laser_points)
        start_seed_index, last_seed_index = seed
        PB = max(breakpoint_index, start_seed_index - 1)
        PF = min(last_seed_index + 1, number_points - 1)

        while self.vertical_distance_point_to_line(self.line_params, self.laser_points[PF]) < self.vertical_distance_threshold and self.distance_2points(self.laser_points[PF], self.laser_points[PF - 1]) < self.adjacent_points_distance_threshold:
            
            predicted_point = self.predict_point([line_eq[0], line_eq[1]], self.laser_points[PF])
            distance = self.distance_2points(self.laser_points[PF], predicted_point)
            if distance > self.predicted_distance_threshold:
                break
            
            line_eq = self.odr_fit(self.laser_points[PB:PF + 1])
            
            PF += 1
            if PF == number_points:
                break

        PF -= 1

        while self.vertical_distance_point_to_line(self.line_params, self.laser_points[PB]) < self.vertical_distance_threshold and self.distance_2points(self.laser_points[PB], self.laser_points[PB + 1]) < self.adjacent_points_distance_threshold:
            
            predicted_point = self.predict_point([line_eq[0], line_eq[1]], self.laser_points[PB])
            distance = self.distance_2points(self.laser_points[PB], predicted_point)
            if distance > self.predicted_distance_threshold:
                break

            line_eq = self.odr_fit(self.laser_points[PB:PF + 1])
            
            PB -= 1
            if PB == breakpoint_index - 1:
                break

        PB += 1

        #line_length = self.distance_2points(self.projection_point2line(self.laser_points[PB][0], line_eq[0], line_eq[1]), self.projection_point2line(self.laser_points[PF][0], line_eq[0], line_eq[1]))
        line_length = self.distance_2points(self.laser_points[PB], self.laser_points[PF])
        line_points = PF - PB + 1

        if line_length >= self.minimum_line_length and line_points >= self.minimum_points:
            self.line_params = line_eq
            return [PB, PF]
        
        return False
    
    def overlap_region_processing(self, lines): #lines[0] = [PB, PF] lines[1] = eq
        new_lines = lines

        current_line_index = 0
        while current_line_index < len(new_lines) - 1:
            next_line_index = current_line_index + 1
            endpoint_current_line_delimitors, current_line_eq = new_lines[current_line_index] 
            endpoint_next_line_delimitors, next_line_eq = new_lines[next_line_index]
            
            #if endpoint_next_line_delimitors[0] <= endpoint_current_line_delimitors[1]:

            if self.distance_2points(self.laser_points[endpoint_current_line_delimitors[1]], self.laser_points[endpoint_next_line_delimitors[0]]) < self.adjacent_points_distance_threshold:
                tehta1 = atan(current_line_eq[0])
                tehta2 = atan(next_line_eq[0])
                dif = abs(tehta1 - tehta2)
                if (dif < 0.05 or dif > pi - 0.05):
                    left = min(endpoint_current_line_delimitors[0], endpoint_next_line_delimitors[0])
                    right = endpoint_next_line_delimitors[1]

                    m, b = self.odr_fit(self.laser_points[left:right+1])
                    new_lines[current_line_index] = [[left, right], [m, b]]
                    new_lines.pop(next_line_index)
                    current_line_index -= 1
            
            current_line_index += 1
        #return new_lines            
        
        for current_line_index in range(0, len(new_lines) - 1):
            next_line_index = current_line_index + 1
            endpoint_current_line_delimitors, current_line_eq = new_lines[current_line_index]
            endpoint_next_line_delimitors, next_line_eq = new_lines[next_line_index]

            if endpoint_next_line_delimitors[0] <= endpoint_current_line_delimitors[1]:
                for point_index in range(endpoint_next_line_delimitors[0], endpoint_current_line_delimitors[1] + 1):
                    distance_current_line = self.vertical_distance_point_to_line(current_line_eq, self.laser_points[point_index])
                    distance_next_line = self.vertical_distance_point_to_line(next_line_eq, self.laser_points[point_index])

                    if distance_current_line < distance_next_line:
                        continue

                    break

                endpoint_current_line_delimitors[1] = point_index - 1
                endpoint_next_line_delimitors[0] = point_index

            else:
                continue
            
            m1, b1 = self.odr_fit(self.laser_points[endpoint_current_line_delimitors[0]: endpoint_current_line_delimitors[1] + 1])
            m2, b2 = self.odr_fit(self.laser_points[endpoint_next_line_delimitors[0]: endpoint_next_line_delimitors[1] + 1])

            new_lines[current_line_index] = [endpoint_current_line_delimitors, [m1, b1]]
            new_lines[next_line_index] = [endpoint_next_line_delimitors, [m2, b2]]
        
        return new_lines
    
    def cart2pol(self, point):
        x = point[0]
        y = point[1]
        rho = np.sqrt(x**2 + y**2)
        phi = np.arctan2(y, x)

        if phi < 0:
            phi = phi + 2*np.pi
        
        #print("POINT : " + str(point) + ". " + "CONVERTED : " + str([rho, phi]))
        return np.array([rho, phi])
    
    def pol2cart(self, pol):
        r, theta = pol

        return  np.array([r*cos(theta), r*sin(theta)])
    
    def dist2pol(self, pol1, pol2):
        r1, theta1 = pol1
        r2, theta2 = pol2
        return np.sqrt(r1 ** 2 + r2**2 - 2 * r1 * r2 * cos(theta1 - theta2))
    
    def convert_lines(self, lines):
        features = []
        for line in lines:
            m = line[1][0]
            b = line[1][1]
            features.append([self.cart2pol(self.projection_point2line([0,0], m, b)), [self.projection_point2line(self.laser_points[line[0][0]], m, b), self.projection_point2line(self.laser_points[line[0][1]], m, b)], [m, b]])

        return features

    def feature_extraction(self, laser_scan):
        self.set_laser_points(laser_scan)
        breakpoint = 0
        lines = []
       
        while breakpoint < self.number_points - self.minimum_points:
            new_seed = self.seed_segment_detection(breakpoint)
            if new_seed == False:
                break
            
            new_line = self.seed_segment_growing(new_seed, breakpoint)
            if new_line == False:
                breakpoint = new_seed[1]
                continue
            
            PB, PF = new_line
            m,b = self.line_params
            lines.append([new_line, [m, b]])
            breakpoint = PF + 1
        
        lines = self.overlap_region_processing(lines)

        return self.convert_lines(lines)

