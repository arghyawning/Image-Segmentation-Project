from __future__ import division
import cv2
import numpy as np
import os
import argparse
from math import exp, pow
from functools import partial
from augmentingPath import augmentingPath
from pushRelabel import pushRelabel
from boykovKolmogorov import boykovKolmogorov

graphCutAlgo = {"ap": augmentingPath, "pr": pushRelabel, "bk": boykovKolmogorov}

SIGMA = 30
OBJCOLOR, BKGCOLOR = (0, 0, 255), (0, 255, 0)
OBJCODE, BKGCODE = 1, 2
OBJ, BKG = "OBJ", "BKG"
CUTCOLOR = (0, 0, 255)

SOURCE, SINK = -2, -1
SF = 10


def show_image(image):
    windowname = "Segmentation"
    cv2.namedWindow(windowname, cv2.WINDOW_NORMAL)
    cv2.imshow(windowname, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def plantSeed(image):
    def drawLines(x, y, pixelType, radius, thickness):
        if pixelType == OBJ:
            color, code = OBJCOLOR, OBJCODE
        else:
            color, code = BKGCOLOR, BKGCODE
        cv2.circle(image, (x, y), radius, color, thickness)
        cv2.circle(seeds, (x // SF, y // SF), radius // SF, code, thickness)

    def onMouse(event, x, y, flags, param):
        pixelType, radius, thickness = param
        if event == cv2.EVENT_LBUTTONDOWN:
            drawLines(x, y, pixelType, radius, thickness)
        elif event == cv2.EVENT_MOUSEMOVE and flags & cv2.EVENT_FLAG_LBUTTON:
            drawLines(x, y, pixelType, radius, thickness)

    def paintSeeds(pixelType):
        print("Planting", pixelType, "seeds")
        radius = 10
        thickness = -1  # fill the whole circle
        windowname = "Plant " + pixelType + " seeds"
        cv2.namedWindow(windowname, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(windowname, onMouse, param=(pixelType, radius, thickness))
        while True:
            cv2.imshow(windowname, image)
            if cv2.waitKey(1) & 0xFF == 27:  # Exit on ESC key
                break
        cv2.destroyAllWindows()

    seeds = np.zeros(image.shape, dtype="uint8")
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    image = cv2.resize(image, (0, 0), fx=SF, fy=SF)

    paintSeeds(OBJ)
    paintSeeds(BKG)
    return seeds, image


def boundaryPenalty(ip, iq):
    return 100 * exp(-pow(int(ip) - int(iq), 2) / (2 * pow(SIGMA, 2)))


def buildGraph(image):
    V = image.size + 2
    graph = np.zeros((V, V), dtype="int32")
    K = makeNLinks(graph, image)
    seeds, seededImage = plantSeed(image)
    makeTLinks(graph, seeds, K)
    return graph, seededImage


def makeNLinks(graph, image):
    K = -float("inf")
    r, c = image.shape
    for i in range(r):
        for j in range(c):
            x = i * c + j
            if i + 1 < r:  # Pixel below
                y = (i + 1) * c + j
                bp = boundaryPenalty(image[i][j], image[i + 1][j])
                graph[x][y] = graph[y][x] = bp
                K = max(K, bp)
            if j + 1 < c:  # Pixel to the right
                y = i * c + j + 1
                bp = boundaryPenalty(image[i][j], image[i][j + 1])
                graph[x][y] = graph[y][x] = bp
                K = max(K, bp)
    return K


def makeTLinks(graph, seeds, K):
    r, c = seeds.shape
    for i in range(r):
        for j in range(c):
            x = i * c + j
            if seeds[i][j] == OBJCODE:
                graph[SOURCE][x] = K
            elif seeds[i][j] == BKGCODE:
                graph[x][SINK] = K


def displayCut(image, cuts):
    def colorPixel(i, j):
        image[i][j] = CUTCOLOR

    r, c = image.shape
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    for c in cuts:
        if c[0] != SOURCE and c[0] != SINK and c[1] != SOURCE and c[1] != SINK:
            colorPixel(c[0] // r, c[0] % r)
            colorPixel(c[1] // r, c[1] % r)
    return image


def imageSegmentation(imagefile, size=(30, 30), algo="ap"):
    pathname = os.path.splitext(imagefile)[0]
    image = cv2.imread(imagefile, cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image, size)
    graph, seededImage = buildGraph(image)
    cv2.imwrite(pathname + "seeded.jpg", seededImage)

    global SOURCE, SINK
    SOURCE += len(graph)
    SINK += len(graph)

    cuts = graphCutAlgo[algo](graph, SOURCE, SINK)
    print("cuts:")
    print(cuts)
    image = displayCut(image, cuts)
    image = cv2.resize(image, (0, 0), fx=SF, fy=SF)
    show_image(image)
    savename = pathname + "cut.jpg"
    cv2.imwrite(savename, image)
    print("Saved image as", savename)


def parseArgs():
    def algorithm(string):
        if string in graphCutAlgo:
            return string
        raise argparse.ArgumentTypeError(
            f"Algorithm should be one of: {', '.join(graphCutAlgo.keys())}"
        )

    parser = argparse.ArgumentParser()
    parser.add_argument("imagefile")
    parser.add_argument("--size", "-s", default=30, type=int, help="Defaults to 30x30")
    parser.add_argument("--algo", "-a", default="ap", type=algorithm)
    return parser.parse_args()


if __name__ == "__main__":
    args = parseArgs()
    imageSegmentation(args.imagefile, (args.size, args.size), args.algo)
