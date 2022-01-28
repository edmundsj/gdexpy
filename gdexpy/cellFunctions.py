import gdspy
import numpy as np

# Currently this does not maintain the hierarchy.
def cellTranslate(inputCell, translationVector):
    newName = inputCell.name + "_"
    tempCell = gdspy.Cell(newName)
    for polygon in inputCell.get_polygonsets():
        polygon.translate(translationVector[0], translationVector[1])
        tempCell.add(polygon)
    return tempCell

def cellRotate(inputCell, rotationAngleDegrees):
    newName = inputCell.name + "_"
    tempCell = gdspy.Cell(newName)
    for polygon in inputCell.get_polygonsets():
        polygon.rotate(rotationAngleDegrees * np.pi / 180)
        tempCell.add(polygon)
    return tempCell

def cellScale(inputCell, magnification):
    newName = inputCell.name + "_"
    tempCell = gdspy.Cell(newName)
    for polygon in inputCell.get_polygonsets():
        polygon.scale(magnification)
        tempCell.add(polygon)
    return tempCell

def cellChangeLayer(inputCell, layer=0):
    newName = inputCell.name + "_"
    tempCell = gdspy.Cell(newName)
    for polygon in inputCell.get_polygonsets():
        polygon.layers = [layer]
        tempCell.add(polygon)
    return tempCell

def cellTransform(inputCell, offset=[0,0], angle=0, magnification=1):
    newName = inputCell.name + "_"
    tempCell = gdspy.Cell(newName)
    for polygon in inputCell.get_polygonsets():
        polygon.rotate(rotationAngle)
        polygon.translate(translationVector[0], translationVector[1])
        polygon.scale(magnification)
        tempCell.add(polygon)

def convert_to_reticle(polygons, layer=0, number_fields=4, invert=False, final_layer=5,
                      field_size=20000, field_spacing=4800):
    inner_cell = gdspy.Cell('LAYER' + str(layer))
    coordinates = polygons[(layer, 0)]
    x_offset = -1/2 + layer % 2

    if number_fields <= 4:
        y_offset = (1/2 - int(layer/2))
    elif number_fields >= 4 and number_fields <= 6:
        y_offset =  (1 - int(layer/2))

    for coords in coordinates:
        tempPolygon = gdspy.Polygon(coords, layer=final_layer)
        tempPolygon.scale(4)
        tempPolygon.translate(x_offset*(field_size + field_spacing), y_offset*(field_size + field_spacing))
        inner_cell.add(tempPolygon)

    layer_mask = gdspy.Rectangle([-field_size/2, -field_size/2], [field_size/2, field_size/2], layer=final_layer)
    layer_mask.translate(x_offset*(field_size + field_spacing), y_offset*(field_size + field_spacing))

    if invert == True:
        inner_cell = gdspy.boolean(layer_mask, inner_cell, 'not', layer=final_layer)
    elif invert == False:
        inner_cell = gdspy.boolean(layer_mask, inner_cell, 'and', layer=final_layer)

    return inner_cell

def bond_pad_group(offset=[0,0], pad_spacing=50, pad_size=100, layer=2, trace_width=5, trace_spacing=5):

    bond_pad_center = gdspy.Rectangle(
        [offset[0],
            offset[1] - pad_size/2],
        [offset[0] + pad_size,
            offset[1] + pad_size/2],
            layer=layer)

    bond_pad_top = gdspy.Rectangle(
        [offset[0],
            offset[1] - pad_size/2 - pad_spacing - pad_size],
        [offset[0] + pad_size,
            offset[1] + pad_size/2 - pad_spacing - pad_size],
            layer=layer)

    bond_pad_bottom = gdspy.Rectangle(
        [offset[0],
            offset[1] - pad_size/2 + pad_spacing + pad_size],
        [offset[0] + pad_size,
            offset[1] + pad_size/2 + pad_spacing + pad_size],
        layer=layer)

    connector_trace = gdspy.Rectangle(
        [offset[0] + pad_size*2 + trace_spacing,
            offset[1] - pad_spacing - pad_size/2 - pad_size],
        [offset[0] + pad_size*2 + trace_spacing + trace_width,
            offset[1] + pad_size/2 + pad_spacing + pad_size],
        layer=layer)

    connector_top = gdspy.Rectangle(
        [offset[0] + pad_size,
            offset[1] + pad_spacing + pad_size/2 + pad_size/2],
        [offset[0] + pad_size*2 + trace_spacing + trace_width,
            offset[1] + pad_spacing + pad_size/2 + pad_size/2 + \
            trace_width],
        layer=layer)

    connector_middle = gdspy.Rectangle(
        [offset[0] + pad_size,
            offset[1]],
        [offset[0] + pad_size*2 + trace_width,
            offset[1] + trace_width],
        layer=layer)

    connector_bottom = gdspy.Rectangle(
        [offset[0] + pad_size,
            offset[1] - pad_spacing - pad_size/2 - pad_size/2],
        [offset[0] + pad_size*2 + trace_width,
            offset[1] - pad_spacing - pad_size/2 - pad_size/2 + trace_width],
        layer=layer)

    full_polygon_1 = gdspy.boolean(bond_pad_center, bond_pad_top,
            "or", layer=layer)
    full_polygon_2 = gdspy.boolean(bond_pad_bottom, connector_trace,
            "or", layer=layer)
    connector_polygon_1 = gdspy.boolean(connector_top, connector_bottom,
            "or", layer=layer)
    connector_polygon_2 = gdspy.boolean(connector_middle, connector_trace,
            "or", layer=layer)

    connector_polygon = gdspy.boolean(connector_polygon_1,
            connector_polygon_2,
            "or", layer=layer)
    full_polygon = gdspy.boolean(full_polygon_1, full_polygon_2,
            "or", layer=layer)

    full_polygon = gdspy.boolean(full_polygon, connector_polygon,
            "or", layer=layer)
    return full_polygon

class TraceRouter:
    def __init__(self, start, end, trace_width=5, layer=0):
        self.start = start
        self.end = end
        self.trace = gdspy.Path(trace_width, start)
        self.layer = layer
        self.current_location = start.copy()
        self.trace_width=trace_width
        self.previous_theta = None

    def direction_to_angle(self, direction):

        if direction == '+x':
            angle = 0
        elif direction == '-x':
            angle = np.pi
        elif direction == '+y':
            angle = np.pi/2
        elif direction == '-y':
            angle = -np.pi/2
        else:
            angle = direction
        return angle

    def route(self, direction=0, length=None):
        """
        :param angle: The angle to route the trace at, in radians
        """
        theta = self.direction_to_angle(direction)
        direction_vector = np.array([np.cos(theta), np.sin(theta)])
        target_vector = self.end - self.current_location

        if self.previous_theta != None:
            if theta != self.previous_theta:
                self.trace.turn(self.trace_width, angle=theta - self.previous_theta, layer=self.layer)

        if length == None:
            vector_to_move = clip_vector(direction_vector, target_vector)
        elif length < 0:
            vector_to_move = clip_vector(direction_vector, target_vector)
            vector_to_move += direction_vector * length
        else:
            vector_to_move = direction_vector * length

        self.current_location[0] = self.current_location[0] + vector_to_move[0]
        self.current_location[1] = self.current_location[1] + vector_to_move[1]
        actual_length = np.linalg.norm(vector_to_move)

        self.trace.segment(actual_length, direction, layer=self.layer)
        self.previous_theta = theta

def clip_vector(input_vector, target_vector):
    epsilon = 1e-3
    v_hat = input_vector.copy() / np.linalg.norm(input_vector)

    if abs(v_hat[0]) > epsilon:
        actual_length_x = target_vector[0] / v_hat[0]
    else:
        actual_length_x = np.inf

    if abs(v_hat[1]) > epsilon:
        actual_length_y = target_vector[1] / v_hat[1]
    else:
        actual_length_y = np.inf

    if actual_length_x < 0:
        actual_length_x = np.inf
    if actual_length_y < 0:
        actual_length_y = np.inf

    actual_length = min(actual_length_x, actual_length_y)

    if actual_length == np.inf:
        actual_length = 0

    final_vector = v_hat * actual_length
    return final_vector
