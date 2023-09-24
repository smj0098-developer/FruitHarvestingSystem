# +
import numpy as np
import tensorflow.keras as K
import tensorflow as tf
from tensorflow.keras.datasets import cifar10
import copy
import cv2
import matplotlib.pyplot as plt
import json
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model
from os import listdir
from os.path import isfile, join
import matplotlib.patches as patches

class Tools:
    def __init__(self,ImageHeight,ImageWidth):
        self.ImageHeight = ImageHeight
        self.ImageWidth = ImageWidth
        
    def ImageDataSetter(self,trainData):
        self.trainData = trainData
        
    def LabelSetter(self,TrainingCategoryLabels,TrainingbboxLabels):
        self.TrainingCategoryLabels = TrainingCategoryLabels
        self.TrainingbboxLabels = TrainingbboxLabels
        
    def CenterXYSetterForDraw(self,DepictingCenterX,DepictingCenterY):
        self.DepictingCenterX = DepictingCenterX
        self.DepictingCenterY = DepictingCenterY
        
        
    def ImageDataExtractor(self,dataPortion):
        Data = []
        for i in dataPortion:
            Data.append((i[0][0].numpy()).astype(np.uint8))
        return Data

    def annotationgenerator(self,directory):  
        categoryLabels = []
        bboxLabels = []
        mypath = directory
        onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
        for fileName in onlyfiles:
            f = open(mypath+fileName)
            dict1 = json.load(f)
            tmpBoxArr = []
            tmpCategoryArr = []
            for instance in dict1['annotations']:
                tmpCategoryArr.append(instance["category_id"])
                xmin = instance["bbox"][0]/1008 # image width is 1008 in the dataSet
                ymin = instance["bbox"][1]/756 # image height is 756 in the dataSet
                xmax = instance["bbox"][2]/1008 # image width is 1008 in the dataSet
                ymax = instance["bbox"][3]/756 # image height is 756 in the dataSet
                resizedbbox = [xmin,ymin,xmax,ymax]
                tmpBoxArr.append(resizedbbox)
            categoryLabels.append(tmpCategoryArr)
            bboxLabels.append(tmpBoxArr)
        return bboxLabels,categoryLabels
    
    
    # drawing an image with bounding boxes and print its category
    def drawIthTrainingData(self,ithData):
        fig, ax = plt.subplots(1)
        ax.imshow((self.trainData[ithData]).astype(np.uint8))
        print(self.TrainingCategoryLabels[ithData])
        for bbox in self.TrainingbboxLabels[ithData]:

            x_min, y_min, x_max, y_max = bbox
            print(x_min*self.ImageWidth, y_min*self.ImageHeight, x_max*self.ImageWidth, y_max*self.ImageHeight)
            width = x_max*self.ImageWidth - x_min*self.ImageWidth
            height = y_max*self.ImageHeight - y_min*self.ImageHeight
            rect = patches.Rectangle((x_min*self.ImageWidth, y_min*self.ImageHeight), width, height, linewidth=1, edgecolor='r', facecolor='none')
            ax.add_patch(rect)
        plt.show()
        
        
        
    # drawing the dots on the ith image in our 'Training' dataset
    def drawAnchorBoxCenters(self,ithData=0):
        fig, ax = plt.subplots(1)
        ax.imshow((self.trainData[ithData]).astype(np.uint8))
        print(self.TrainingCategoryLabels[ithData])
        for bbox in self.TrainingbboxLabels[ithData]:
            x_min, y_min, x_max, y_max = bbox
            print(x_min*self.ImageWidth, y_min*self.ImageHeight, x_max*self.ImageWidth, y_max*self.ImageHeight)
            width = x_max*self.ImageWidth - x_min*self.ImageWidth
            height = y_max*self.ImageHeight - y_min*self.ImageHeight
            rect = patches.Rectangle((x_min*self.ImageWidth, y_min*self.ImageHeight), width, height, linewidth=1, edgecolor='r', facecolor='none')
            ax.add_patch(rect)
        plt.scatter(x=self.DepictingCenterX, y=self.DepictingCenterY, c='b', s=1)
        plt.xlim(-50,710)
        plt.ylim(550,-50)
        plt.show()

        
        
    def IOUCalculator(self,thisImgBbox,anchorXYlist):
        anchorXmin,anchorYmin,anchorXmax,anchorYmax = anchorXYlist
        # in any center position, the selected anchor is chosen with the best possible iou by being compared to all groundTruth bboxes
        maxIou = 0
        realBbox = []
        for bbox in thisImgBbox:
            # real means the real groundTruth boundingBoxes, not the ones we are computing as groundTruth anchor boxes
            realXmin,realYmin,realXmax,realYmax = bbox
    #         print(anchorXmin)
            interXmin = max(realXmin*self.ImageWidth,anchorXmin)
            interYmin = max(realYmin*self.ImageHeight,anchorYmin)
            interXmax = min(realXmax*self.ImageWidth,anchorXmax)
            interYmax = min(realYmax*self.ImageHeight,anchorYmax)
            width_inter = interXmax-interXmin
            height_inter = interYmax-interYmin

            area_inter = width_inter*height_inter

            if width_inter<=0 or height_inter<=0:    
                area_inter = 0

            widthAnchor = anchorXmax-anchorXmin
            heightAnchor = anchorYmax-anchorYmin

            widthRealBbox = realXmax*self.ImageWidth-realXmin*self.ImageWidth
            heightRealBbox = realYmax*self.ImageHeight-realYmin*self.ImageHeight
            anchorArea = widthAnchor*heightAnchor
            realBboxArea = widthRealBbox*heightRealBbox
            areaUnion = anchorArea+realBboxArea-area_inter
    #         print(realBboxArea)
            iou = area_inter/areaUnion
    #         print(iou)
            if iou > maxIou:
                realBbox = [realXmin*self.ImageWidth,realYmin*self.ImageHeight,realXmax*self.ImageWidth,realYmax*self.ImageHeight]
                maxIou = iou

        return maxIou,realBbox


    # for depicting purposes    
    def anchorBoxDrawer(self,realbboxSet,ithimage,anchorsToDraw,anchorXmin,anchorYmin,anchorXmax,anchorYmax,ax):
        anchorsToDraw.append([anchorXmin,anchorYmin,anchorXmax,anchorYmax])

        for bbox in anchorsToDraw:
            x_min, y_min, x_max, y_max = bbox
            width = x_max - x_min
            height = y_max - y_min
            rect = patches.Rectangle((x_min, y_min), width, height, linewidth=1, edgecolor='b', facecolor='none')
            ax.add_patch(rect)
        for bbox in realbboxSet:
            x_min, y_min, x_max, y_max = bbox
            width = x_max*self.ImageWidth - x_min*self.ImageWidth
            height = y_max*self.ImageHeight - y_min*self.ImageHeight
            rect = patches.Rectangle((x_min*self.ImageWidth, y_min*self.ImageHeight), width, height, linewidth=1, edgecolor='r', facecolor='none')
            ax.add_patch(rect)

            
    def AnchorSetter(self,A1X,A1Y,A2X,A2Y,A3X,A3Y,A4X,A4Y,A5X,A5Y,A6X,A6Y,A7X,A7Y,A8X,A8Y,A9X,A9Y):
        self.A1X = A1X
        self.A1Y = A1Y
        self.A2X = A2X
        self.A2Y = A2Y
        self.A3X = A3X
        self.A3Y = A3Y
        self.A4X = A4X
        self.A4Y = A4Y
        self.A5X = A5X
        self.A5Y = A5Y
        self.A6X = A6X
        self.A6Y = A6Y
        self.A7X = A7X
        self.A7Y = A7Y
        self.A8X = A8X
        self.A8Y = A8Y
        self.A9X = A9X
        self.A9Y = A9Y
        

    def anchorBasedLabelGenerator(self,bboxLabelsPortion,ImageDataPortion,LetsDraw):
        ithimage=46 # index of the image selected for depicting purpose
        ithCenter = 74 # there are XpointsNum*YpointsNum centers

        anchorsToDraw = []
        objectNessLabels = []
        BboxRegLabels = []

        for ithBboxSet in range(len(bboxLabelsPortion)):
            oneImageObjectNessLabels = []
            oneImageBboxRegLabels = []
            # for depicting purposes
            if ithBboxSet == ithimage and LetsDraw:
                fig, ax = plt.subplots(1)
                ax.imshow((ImageDataPortion[ithimage]).astype(np.uint8))

            counter = 0
            for centerX,centerY in zip(self.DepictingCenterX,self.DepictingCenterY):
                counter+=1
                # lets apply the 9 anchorBoxes to these centers
                A1Xmin = centerX-self.A1X/2
                A1Xmax = centerX+self.A1X/2
                A1Ymin = centerY-self.A1Y/2
                A1Ymax = centerY+self.A1Y/2

                A2Xmin = centerX-self.A2X/2
                A2Xmax = centerX+self.A2X/2
                A2Ymin = centerY-self.A2Y/2
                A2Ymax = centerY+self.A2Y/2

                A3Xmin = centerX-self.A3X/2
                A3Xmax = centerX+self.A3X/2
                A3Ymin = centerY-self.A3Y/2
                A3Ymax = centerY+self.A3Y/2

                A4Xmin = centerX-self.A4X/2
                A4Xmax = centerX+self.A4X/2
                A4Ymin = centerY-self.A4Y/2
                A4Ymax = centerY+self.A4Y/2

                A5Xmin = centerX-self.A5X/2
                A5Xmax = centerX+self.A5X/2
                A5Ymin = centerY-self.A5Y/2
                A5Ymax = centerY+self.A5Y/2

                A6Xmin = centerX-self.A6X/2
                A6Xmax = centerX+self.A6X/2
                A6Ymin = centerY-self.A6Y/2
                A6Ymax = centerY+self.A6Y/2

                A7Xmin = centerX-self.A7X/2
                A7Xmax = centerX+self.A7X/2
                A7Ymin = centerY-self.A7Y/2
                A7Ymax = centerY+self.A7Y/2

                A8Xmin = centerX-self.A8X/2
                A8Xmax = centerX+self.A8X/2
                A8Ymin = centerY-self.A8Y/2
                A8Ymax = centerY+self.A8Y/2

                A9Xmin = centerX-self.A9X/2
                A9Xmax = centerX+self.A9X/2
                A9Ymin = centerY-self.A9Y/2
                A9Ymax = centerY+self.A9Y/2


                A1IOU,A1RealBbox = self.IOUCalculator(bboxLabelsPortion[ithBboxSet],[A1Xmin,A1Ymin,A1Xmax,A1Ymax])
                A2IOU,A2RealBbox = self.IOUCalculator(bboxLabelsPortion[ithBboxSet],[A2Xmin,A2Ymin,A2Xmax,A2Ymax])
                A3IOU,A3RealBbox = self.IOUCalculator(bboxLabelsPortion[ithBboxSet],[A3Xmin,A3Ymin,A3Xmax,A3Ymax])
                A4IOU,A4RealBbox = self.IOUCalculator(bboxLabelsPortion[ithBboxSet],[A4Xmin,A4Ymin,A4Xmax,A4Ymax])
                A5IOU,A5RealBbox = self.IOUCalculator(bboxLabelsPortion[ithBboxSet],[A5Xmin,A5Ymin,A5Xmax,A5Ymax])
                A6IOU,A6RealBbox = self.IOUCalculator(bboxLabelsPortion[ithBboxSet],[A6Xmin,A6Ymin,A6Xmax,A6Ymax])
                A7IOU,A7RealBbox = self.IOUCalculator(bboxLabelsPortion[ithBboxSet],[A7Xmin,A7Ymin,A7Xmax,A7Ymax])
                A8IOU,A8RealBbox = self.IOUCalculator(bboxLabelsPortion[ithBboxSet],[A8Xmin,A8Ymin,A8Xmax,A8Ymax])
                A9IOU,A9RealBbox = self.IOUCalculator(bboxLabelsPortion[ithBboxSet],[A9Xmin,A9Ymin,A9Xmax,A9Ymax])
                T = 0.3 # the threashold by which we identify each anchorBox as either positive or negative

                if A1IOU>T:
                    oneImageObjectNessLabels.append(1)
                    oneImageBboxRegLabels += [A1RealBbox[0]-A1Xmin,A1RealBbox[1]-A1Ymin,A1RealBbox[2]-A1Xmax,A1RealBbox[3]-A1Ymax]
                else:
                    oneImageObjectNessLabels.append(0)
                    oneImageBboxRegLabels += [0,0,0,0]
                if A2IOU>T:
                    oneImageObjectNessLabels.append(1)
                    oneImageBboxRegLabels += [A2RealBbox[0]-A2Xmin,A2RealBbox[1]-A2Ymin,A2RealBbox[2]-A2Xmax,A2RealBbox[3]-A2Ymax]
                else:
                    oneImageObjectNessLabels.append(0)
                    oneImageBboxRegLabels += [0,0,0,0]
                if A3IOU>T:
                    oneImageObjectNessLabels.append(1)
                    oneImageBboxRegLabels += [A3RealBbox[0]-A3Xmin,A3RealBbox[1]-A3Ymin,A3RealBbox[2]-A3Xmax,A3RealBbox[3]-A3Ymax]
                else:
                    oneImageObjectNessLabels.append(0)
                    oneImageBboxRegLabels += [0,0,0,0]
                if A4IOU>T:
                    oneImageObjectNessLabels.append(1)
                    oneImageBboxRegLabels += [A4RealBbox[0]-A4Xmin,A4RealBbox[1]-A4Ymin,A4RealBbox[2]-A4Xmax,A4RealBbox[3]-A4Ymax]
                else:
                    oneImageObjectNessLabels.append(0)
                    oneImageBboxRegLabels += [0,0,0,0]
                if A5IOU>T:
                    oneImageObjectNessLabels.append(1)
                    oneImageBboxRegLabels += [A5RealBbox[0]-A5Xmin,A5RealBbox[1]-A5Ymin,A5RealBbox[2]-A5Xmax,A5RealBbox[3]-A5Ymax]
                else:
                    oneImageObjectNessLabels.append(0)
                    oneImageBboxRegLabels += [0,0,0,0]
                if A6IOU>T:
                    oneImageObjectNessLabels.append(1)
                    oneImageBboxRegLabels += [A6RealBbox[0]-A6Xmin,A6RealBbox[1]-A6Ymin,A6RealBbox[2]-A6Xmax,A6RealBbox[3]-A6Ymax]
                else:
                    oneImageObjectNessLabels.append(0)
                    oneImageBboxRegLabels += [0,0,0,0]
                if A7IOU>T:
                    oneImageObjectNessLabels.append(1)
                    oneImageBboxRegLabels += [A7RealBbox[0]-A7Xmin,A7RealBbox[1]-A7Ymin,A7RealBbox[2]-A7Xmax,A7RealBbox[3]-A7Ymax]
                else:
                    oneImageObjectNessLabels.append(0)
                    oneImageBboxRegLabels += [0,0,0,0]
                if A8IOU>T:
                    oneImageObjectNessLabels.append(1)
                    oneImageBboxRegLabels += [A8RealBbox[0]-A8Xmin,A8RealBbox[1]-A8Ymin,A8RealBbox[2]-A8Xmax,A8RealBbox[3]-A8Ymax]
                else:
                    oneImageObjectNessLabels.append(0)
                    oneImageBboxRegLabels += [0,0,0,0]
                if A9IOU>T:
                    oneImageObjectNessLabels.append(1)
                    oneImageBboxRegLabels += [A9RealBbox[0]-A9Xmin,A9RealBbox[1]-A9Ymin,A9RealBbox[2]-A9Xmax,A9RealBbox[3]-A9Ymax]
                else:
                    oneImageObjectNessLabels.append(0)
                    oneImageBboxRegLabels += [0,0,0,0]


                if ithBboxSet == ithimage and LetsDraw:

                    if(A1IOU>T):
    #                     print(A1IOU)
                        self.anchorBoxDrawer(bboxLabelsPortion[ithBboxSet],ithimage,anchorsToDraw,A1Xmin,A1Ymin,A1Xmax,A1Ymax,ax)
                    if(A2IOU>T):
    #                     print(A2IOU)
                        self.anchorBoxDrawer(bboxLabelsPortion[ithBboxSet],ithimage,anchorsToDraw,A2Xmin,A2Ymin,A2Xmax,A2Ymax,ax)
                    if(A3IOU>T):
    #                     print(A3IOU)
                        self.anchorBoxDrawer(bboxLabelsPortion[ithBboxSet],ithimage,anchorsToDraw,A3Xmin,A3Ymin,A3Xmax,A3Ymax,ax)
                    if(A4IOU>T):
    #                     print(A4IOU)
                        self.anchorBoxDrawer(bboxLabelsPortion[ithBboxSet],ithimage,anchorsToDraw,A4Xmin,A4Ymin,A4Xmax,A4Ymax,ax)
                    if(A5IOU>T):
    #                     print(A5IOU)
                        self.anchorBoxDrawer(bboxLabelsPortion[ithBboxSet],ithimage,anchorsToDraw,A5Xmin,A5Ymin,A5Xmax,A5Ymax,ax)
                    if(A6IOU>T):
    #                     print(A6IOU)
                        self.anchorBoxDrawer(bboxLabelsPortion[ithBboxSet],ithimage,anchorsToDraw,A6Xmin,A6Ymin,A6Xmax,A6Ymax,ax)
                    if(A7IOU>T):
    #                     print(A7IOU)
                        self.anchorBoxDrawer(bboxLabelsPortion[ithBboxSet],ithimage,anchorsToDraw,A7Xmin,A7Ymin,A7Xmax,A7Ymax,ax)
                    if(A8IOU>T):
    #                     print(A8IOU)
                        self.anchorBoxDrawer(bboxLabelsPortion[ithBboxSet],ithimage,anchorsToDraw,A8Xmin,A8Ymin,A8Xmax,A8Ymax,ax)
                    if(A9IOU>T):
    #                     print(A9IOU)
                        self.anchorBoxDrawer(bboxLabelsPortion[ithBboxSet],ithimage,anchorsToDraw,A9Xmin,A9Ymin,A9Xmax,A9Ymax,ax)

            objectNessLabels.append(oneImageObjectNessLabels)
            BboxRegLabels.append(oneImageBboxRegLabels)
            # for depicting purposes
            if ithBboxSet == ithimage and LetsDraw:
    #             print(objectNessLabels[ithBboxSet],len(objectNessLabels[ithBboxSet]))
    #             print(BboxRegLabels[ithBboxSet],len(BboxRegLabels[ithBboxSet]))
                plt.scatter(x=self.DepictingCenterX, y=self.DepictingCenterY, c='w', s=1)
                plt.xlim(-100,760)
                plt.ylim(600,-100)
                plt.show()
        return objectNessLabels,BboxRegLabels
    
    
    
    
    def accuracy(self,y_true,y_pred):
#         y_predBinary = y_pred<0.5
        y_pred_binary = tf.where(y_pred<0.5,0,1)
        correct_preds = tf.equal(tf.argmax(y_true),tf.argmax(y_pred_binary))
        accuracy = tf.reduce_mean(tf.cast(correct_preds,tf.float32))

        return accuracy

    
    
    # defining a function to calculate precision and recall
    def presRecall(self,preds,labels):
        TrainingLabels = np.array(labels) > 0.5
        TrainingPreds = preds > 0.5

        counter=0
        truepositive = 0
        allRealPositives = 0
        allPositivePreds = 0
        for i,j in zip(TrainingLabels,TrainingPreds):
                for m,n in zip(i,j):
                    if n and n==m:
                        truepositive+=1
                    if m:
                        allRealPositives+=1
                    if n:
                        allPositivePreds+=1


        print("precision:",truepositive/(allPositivePreds+1)) # adding 1 to avoid division by 0
        print("recall:",truepositive/(allRealPositives+1)) # adding 1 to avoid division by 0
# -

