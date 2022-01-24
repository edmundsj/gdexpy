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

