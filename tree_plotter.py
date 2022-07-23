import matplotlib.pyplot as plt

decisionNode = dict(boxstyle="sawtooth", fc="0.8")
leafNode = dict(boxstyle="round4", fc="0.8")
arrow_args = dict(arrowstyle="<-")


def getNumLeafs(DecisionTree):
    numLeafs = 0
    firstNode = list(DecisionTree.keys())[0]
    secondDict = DecisionTree[firstNode]
    for key in secondDict.keys():
        if type(secondDict[key]).__name__ == 'dict':  # 如果是字典型则表明该结点不是叶子结点，否则该结点为叶子结点
            numLeafs += getNumLeafs(secondDict[key])
        else:
            numLeafs += 1
    return numLeafs


def getTreeDepth(DecisionTree):
    maxDepth = 0
    firstNode = list(DecisionTree.keys())[0]
    secondDict = DecisionTree[firstNode]
    for key in secondDict.keys():
        if type(secondDict[key]).__name__ == 'dict':  # 如果是字典型则表明该结点不是叶子结点，否则该结点为叶子结点
            thisDepth = 1 + getTreeDepth(secondDict[key])
        else:
            thisDepth = 1
        if thisDepth > maxDepth:
            maxDepth = thisDepth
    return maxDepth


def plotNode(nodeTxt, centerPt, parentPt, nodeType):
    createPlot.ax1.annotate(nodeTxt, xy=parentPt, xycoords='axes fraction',
                            xytext=centerPt, textcoords='axes fraction',
                            va="center", ha="center", bbox=nodeType, arrowprops=arrow_args)


def plotMidText(cntrPt, parentPt, txtString):
    xMid = (parentPt[0] - cntrPt[0]) / 2.0 + cntrPt[0]
    yMid = (parentPt[1] - cntrPt[1]) / 2.0 + cntrPt[1]
    createPlot.ax1.text(xMid, yMid, txtString, va="center", ha="center", rotation=30)


def plotTree(DecisionTree, parentPt, nodeTxt):  # if the first key tells you what feat was split on
    numLeafs = getNumLeafs(DecisionTree)  # this determines the x width of this tree
    depth = getTreeDepth(DecisionTree)
    firstNode = list(DecisionTree.keys())[0]  # the text label for this node should be this
    cntrPt = (plotTree.xOff + (1.0 + float(numLeafs)) / 2.0 / plotTree.totalW, plotTree.yOff)
    plotMidText(cntrPt, parentPt, nodeTxt)
    plotNode(firstNode, cntrPt, parentPt, decisionNode)
    secondDict = DecisionTree[firstNode]
    plotTree.yOff = plotTree.yOff - 1.0 / plotTree.totalD
    for key in secondDict.keys():
        if type(secondDict[key]).__name__ == 'dict':  # 如果是字典型则表明该结点不是叶子结点，否则该结点为叶子结点
            plotTree(secondDict[key], cntrPt, str(key))  # recursion
        else:  # 如果是叶子结点就将其画出
            plotTree.xOff = plotTree.xOff + 1.0 / plotTree.totalW
            plotNode(secondDict[key], (plotTree.xOff, plotTree.yOff), cntrPt, leafNode)
            plotMidText((plotTree.xOff, plotTree.yOff), cntrPt, str(key))
    plotTree.yOff = plotTree.yOff + 1.0 / plotTree.totalD


def createPlot(inputTree):
    fig = plt.figure(1, facecolor='white')
    fig.clf()
    axprops = dict(xticks=[], yticks=[])
    createPlot.ax1 = plt.subplot(111, frameon=False, **axprops)  # no ticks
    plotTree.totalW = float(getNumLeafs(inputTree))
    plotTree.totalD = float(getTreeDepth(inputTree))
    plotTree.xOff = -0.5 / plotTree.totalW
    plotTree.yOff = 1.0
    plotTree(inputTree, (0.5, 1.0), '')
    plt.show()