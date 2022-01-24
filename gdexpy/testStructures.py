import gdspy
import numpy as np

def addTLMStructures(lib, width=100, spacing=[50,100,150,200], padding=5, name="TLM",
        offset=[0,0], alignment='top', textSize=50):
    tempCell = lib.new_cell(name)
    if alignment == 'top':
        for i in range(len(spacing)+1):
            if(i<len(spacing)):
                texti = gdspy.Text(str(spacing[i]) + 'u', textSize,
                        [offset[0] + i*width + sum(spacing[0:i]) + width/2 + spacing[i]/2, offset[1] - textSize],
                        layer=2)
                tempCell.add(texti)

            contacti = gdspy.Rectangle(
                    [offset[0] + padding + i*width + sum(spacing[0:i]), offset[1] - padding - textSize],
                    [offset[0] + padding + (i+1)*width + sum(spacing[0:i]), offset[1] - textSize - padding - width], layer=2)
            tempCell.add(contacti)

        filmRegion = gdspy.Rectangle([offset[0], offset[1] - textSize],
            [offset[0] + 2*padding + 5*width + sum(spacing), offset[1] - textSize - 2*padding - width], layer=1)
        widthText = gdspy.Text(str(width) + 'u', textSize,
            [offset[0] - textSize*4, offset[1] - textSize - width/2], layer=2)
        tempCell.add(filmRegion)
        tempCell.add(widthText)

        return tempCell
    else:
        raise NotImplementedError

def addCapacitorStructures(lib, width=[50,100,200,400,800],
        spacing=100, name="CAPS",
        offset=[0,0], alignment='top', textSize=50, layer=2):

    tempCell = lib.new_cell(name)
    if alignment == 'top':
        for i in range(len(width)):
            texti = gdspy.Text(str(width[i]) + 'u', textSize,
                    [offset[0] + i*spacing + sum(width[0:i]), offset[1] - textSize],
                    layer=layer)
            capi = gdspy.Rectangle(
                    [offset[0] + i*spacing + sum(width[0:i]), offset[1] - textSize],
                    [offset[0] + i*spacing + sum(width[0:i+1]), offset[1] - textSize - width[i]], layer=layer)

            tempCell.add(texti)
            tempCell.add(capi)

        return tempCell
    else:
        raise NotImplementedError

def addTestLines(lib, width=5, length=300, name="LINES5", offset=[0, 0], alignment='top', layer=0,
        textSize=50):
    spacing=width
    tempCell = lib.new_cell(name)
    if alignment == 'top':
        texti = gdspy.Text(str(width) + 'u', textSize,
                [offset[0], offset[1] - textSize], layer=layer)
        tempCell.add(texti)
        for i in range(5):
            linei = gdspy.Rectangle([offset[0] + i*width + i*spacing, offset[1] - textSize],
                    [offset[0] + (i+1)*width + i*spacing, offset[1] - textSize - length],
                    layer=layer)
            tempCell.add(linei)
            tempCell.add(texti)
        return tempCell
    else:
        raise NotImplementedError

def generateRings(lib, centerWidth=2200, centerHeight=2200, ringWidth=[750, 450, 270, 162, 97, 58],
        traceWidth=0, traceSpacing=0, name="RINGS", layer=1, offset=[0,0]):
    rings = lib.new_cell(name)
    for i in range(len(ringWidth)):

        outerHalfWidth = centerWidth/2 + np.sum(ringWidth[0:i+1]) + np.sum(ringWidth[0:i])
        innerHalfWidth = centerWidth/2 + 2*np.sum(ringWidth[0:i])
        outerHalfHeight = centerHeight/2 + np.sum(ringWidth[0:i+1]) + np.sum(ringWidth[0:i])
        innerHalfHeight = centerHeight/2 + 2*np.sum(ringWidth[0:i])

        outerPoints = np.array(offset) + np.array([
            [-outerHalfWidth, -outerHalfHeight],
            [-outerHalfWidth, outerHalfHeight],
            [outerHalfWidth, outerHalfHeight],
            [outerHalfWidth, -outerHalfHeight]])

        innerPoints = np.array(offset) + np.array([
            [-innerHalfWidth, -innerHalfHeight],
            [-innerHalfWidth, innerHalfHeight],
            [innerHalfWidth, innerHalfHeight],
            [innerHalfWidth, -innerHalfHeight]])

        cutoutPoints = np.array(offset) + np.array([[-innerHalfWidth, traceSpacing + traceWidth/2],
            [-outerHalfWidth, traceSpacing + traceWidth/2],
            [-outerHalfWidth, -traceSpacing -traceWidth/2],
            [-innerHalfWidth, -traceSpacing -traceWidth/2]])

        ringOuter = gdspy.Polygon(outerPoints)
        ringInner = gdspy.Polygon(innerPoints)
        ringCutout = gdspy.Polygon(cutoutPoints)
        ring = gdspy.boolean(ringOuter, ringInner, 'not')
        ring = gdspy.boolean(ring, ringCutout, 'not', layer=layer)

        rings.add(ring)
    return rings
