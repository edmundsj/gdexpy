import gdspy
import numpy as np
import math
import os
from gdexpy.cellFunctions import cellTranslate, cellRotate

class ASMLBarcodeGenerator():
    mm = 1e3 # All GDS units are by default in microns. This scales them to mm
    x0 = 69*mm                       #Initial horizontal position of the barcode 
                                    #(where the beginning of the barcode is written).
    y = (29.15+48.3/2)*mm		#The vertical center position of the barcode.
    barheight = 5*mm			#The vertical height of the barcode
    quietzonewidth2 = 8*mm		#The horizontal width of quiet zone 2 
    quietzonewidth1 = 2*mm		#The horizontal width of quiet zone 1

    wb = 0.450*mm #wide bar width
    ns = -0.200*mm #narrow space width
    nb = 0.200*mm #narrow bar width
    ws = -0.450*mm #wide space width

    startstop = [nb,ws,nb,ns,wb,ns,wb,ns,nb,ns]
    codearray = {"A": [wb,ns,nb,ns,nb,ws,nb,ns,wb,ns],
                            "B": [nb,ns,wb,ns,nb,ws,nb,ns,wb,ns],
                            "C": [wb,ns,wb,ns,nb,ws,nb,ns,nb,ns],
                            "D": [nb,ns,nb,ns,wb,ws,nb,ns,wb,ns],
                            "E": [wb,ns,nb,ns,wb,ws,nb,ns,nb,ns],
                            "F": [nb,ns,wb,ns,wb,ws,nb,ns,nb,ns],
                            "G": [nb,ns,nb,ns,nb,ws,wb,ns,wb,ns],
                            "H": [wb,ns,nb,ns,nb,ws,wb,ns,nb,ns],
                            "I": [nb,ns,wb,ns,nb,ws,wb,ns,nb,ns],
                            "J": [nb,ns,nb,ns,wb,ws,wb,ns,nb,ns],
                            "K": [wb,ns,nb,ns,nb,ns,nb,ws,wb,ns],
                            "L": [nb,ns,wb,ns,nb,ns,nb,ws,wb,ns],
                            "M": [wb,ns,wb,ns,nb,ns,nb,ws,nb,ns],
                            "N": [nb,ns,nb,ns,wb,ns,nb,ws,wb,ns],
                            "O": [wb,ns,nb,ns,wb,ns,nb,ws,nb,ns],
                            "P": [nb,ns,wb,ns,wb,ns,nb,ws,nb,ns],
                            "Q": [nb,ns,nb,ns,nb,ns,wb,ws,wb,ns],
                            "R": [wb,ns,nb,ns,nb,ns,wb,ws,nb,ns],
                            "S": [nb,ns,wb,ns,nb,ns,wb,ws,nb,ns],
                            "T": [nb,ns,nb,ns,wb,ns,wb,ws,nb,ns],
                            "U": [wb,ws,nb,ns,nb,ns,nb,ns,wb,ns],
                            "V": [nb,ws,wb,ns,nb,ns,nb,ns,wb,ns],
                            "W": [wb,ws,wb,ns,nb,ns,nb,ns,nb,ns],
                            "X": [nb,ws,nb,ns,wb,ns,nb,ns,wb,ns],
                            "Y": [wb,ws,nb,ns,wb,ns,nb,ns,nb,ns],
                            "Z": [nb,ws,wb,ns,wb,ns,nb,ns,nb,ns],
                            "1": [wb,ns,nb,ws,nb,ns,nb,ns,wb,ns],
                            "2": [nb,ns,wb,ws,nb,ns,nb,ns,wb,ns],
                            "3": [wb,ns,wb,ws,nb,ns,nb,ns,nb,ns],
                            "4": [nb,ns,nb,ws,wb,ns,nb,ns,wb,ns],
                            "5": [wb,ns,nb,ws,wb,ns,nb,ns,nb,ns],
                            "6": [nb,ns,wb,ws,wb,ns,nb,ns,nb,ns],
                            "7": [nb,ns,nb,ws,nb,ns,wb,ns,wb,ns],
                            "8": [wb,ns,nb,ws,nb,ns,wb,ns,nb,ns],
                            "9": [nb,ns,wb,ws,nb,ns,wb,ns,nb,ns],
                            "0": [nb,ns,nb,ws,wb,ns,wb,ns,nb,ns],
                            "-": [nb,ws,nb,ns,nb,ns,wb,ns,wb,ns],
                            ".": [wb,ws,nb,ns,nb,ns,wb,ns,nb,ns],
                            "$": [nb,ws,nb,ws,nb,ws,nb,ns,nb,ns],
                            "/": [nb,ws,nb,ws,nb,ns,nb,ws,nb,ns],
                            "+": [nb,ws,nb,ns,nb,ws,nb,ws,nb,ns],
                            "%": [nb,ns,nb,ws,nb,ws,nb,ws,nb,ns],
                            " ": [nb,ws,wb,ns,nb,ns,wb,ns,nb,ns]}

    @classmethod
    def lookupCharacterCode(cls, characterToLookup):
        characterCode = cls.codearray[characterToLookup]
        return characterCode

    @classmethod
    def validateString(cls, barcodeString):
        if type(barcodeString) is not str:
            raise TypeError("You must pass in a string as the first argument to this function")
        else:
            if len(barcodeString) > 12:
                raise TypeError("You cannot use a barcode longer than 12 characters")
            if barcodeString.upper() != barcodeString:
                raise TypeError("The barcode can contain only capital letters")

    @classmethod
    def stringToBarcode(cls, barcodeString="TEST"):
        barcodeString = barcodeString.upper()
        barcodeData = []
        barcodeData += cls.startstop
        for char in barcodeString:
            barcodeData.extend(cls.lookupCharacterCode(char))
        barcodeData.extend(cls.startstop)

        return barcodeData

    @classmethod
    def generateBarcode(cls, barcodeString="TEST", barcodeLayer=4):
        barcodeString = barcodeString.upper()
        cls.validateString(barcodeString)

        barcodeCell = gdspy.Cell('BARCODE')
        currentLocationX = cls.quietzonewidth2
        barcodeData = cls.stringToBarcode(barcodeString)

        for barWidth in barcodeData:
            if barWidth > 0: # bar width is positive: dark line
                barcodeLine = gdspy.Rectangle((currentLocationX, -cls.barheight/2),
                        (currentLocationX + barWidth, cls.barheight/2), barcodeLayer)
                barcodeCell.add(barcodeLine)
            # bar width is negative: empty space
            # in either case, increment the distance of the previous bar.
            currentLocationX += abs(barWidth)

        return barcodeCell

    @classmethod
    def generateBarcodeText(cls, barcodeString="test", barcodeLayer=4):
        barcodeString = barcodeString.upper()
        cls.validateString(barcodeString)

        textCell = gdspy.Cell('BARCODE_TEXT')
        textHeight = 3*cls.mm
        labelText = gdspy.Text(barcodeString, textHeight, (0,0),layer=barcodeLayer)
        labelSize = labelText.get_bounding_box()
        x1 = labelSize[1,0]
        y1 = labelSize[1,1]

        labelText.translate(-x1/2,-y1/2) # Center the text
        textCell.add(labelText)
        return textCell

    @classmethod
    def generateBarcodeAndText(cls, barcodeString="test", barcodeLayer=4):

        barcodeCell = cls.generateBarcode(barcodeString, barcodeLayer)
        textCell = cls.generateBarcodeText(barcodeString, barcodeLayer)
        barcodeCell = cellRotate(barcodeCell, 270)
        textCell = cellRotate(textCell, 90)
        barcodeCell = cellTranslate(barcodeCell, (cls.x0, cls.y))
        textCell = cellTranslate(textCell, (-69.5*cls.mm, 37.5*cls.mm))

        #gdspy.write_gds(str2wrt + 'barcode_asml.gds', unit=1.0e-6, precision=1.0e-9)

        return barcodeCell, textCell

    @classmethod
    def get_template_location(cls):
        base_path = os.path.dirname(__file__)
        full_path = os.path.join(base_path, 'asml_templates')
        return full_path

# Uncommenting this line will execute all code
#BarcodeGenerator.placeBarcode()
