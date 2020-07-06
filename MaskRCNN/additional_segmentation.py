import os
import os.path
import mask_rcnn_additional
import kutils
import numpy
import cv2
import sys
import os
import skimage.morphology
import math
import mrcnn_utils
import additional_visualize


class Segmentation:
    __mModel = None
    __mConfig = None
    __mModelDir = ""
    __mModelPath = ""
    __mLastMaxDim = mask_rcnn_additional.NucleiConfig().IMAGE_MAX_DIM
    __mConfidence = 0.5
    __NMSThreshold = 0.35

    '''
    @param pModelDir clustering Mask_RCNN model path
    '''
    def __init__(self, pModelPath, pConfidence=0.5, pNMSThreshold = 0.35, pMaxDetNum=512):
        if not os.path.isfile(pModelPath):
            raise ValueError("Invalid model path: " + pModelPath)

        self.__mConfidence = pConfidence
        self.__NMSThreshold = pNMSThreshold
        self.__mModelPath = pModelPath
        self.__mModelDir = os.path.dirname(pModelPath)
        self.__mMaxDetNum=pMaxDetNum

    def Segment(self, pImage, pPredictSize=None):

        rebuild = self.__mModel is None

        if pPredictSize is not None:
            maxdim = pPredictSize
            temp = maxdim / 2 ** 6
            if temp != int(temp):
                maxdim = (int(temp) + 1) * 2 ** 6

            if maxdim != self.__mLastMaxDim:
                self.__mLastMaxDim = maxdim
                rebuild = True

        if rebuild:
            import mrcnn_model
            import keras.backend
            keras.backend.clear_session()
            print("Max dim changed (",str(self.__mLastMaxDim),"), rebuilding model")

            self.__mConfig = mask_rcnn_additional.NucleiConfig()
            self.__mConfig.DETECTION_MIN_CONFIDENCE = self.__mConfidence
            self.__mConfig.DETECTION_NMS_THRESHOLD = self.__NMSThreshold
            self.__mConfig.IMAGE_MAX_DIM = self.__mLastMaxDim
            self.__mConfig.IMAGE_MIN_DIM = self.__mLastMaxDim
            self.__mConfig.DETECTION_MAX_INSTANCES=self.__mMaxDetNum
            self.__mConfig.__init__()

            self.__mModel = mrcnn_model.MaskRCNN(mode="inference", config=self.__mConfig, model_dir=self.__mModelDir)
            self.__mModel.load_weights(self.__mModelPath, by_name=True)

        image = kutils.RCNNConvertInputImage(pImage)
        offsetX = 0
        offsetY = 0
        width = image.shape[1]
        height = image.shape[0]

        results = self.__mModel.detect([image], verbose=0)

        r = results[0]
        masks = r['masks']
        scores = r['scores']

        if masks.shape[0] != image.shape[0] or masks.shape[1] != image.shape[1]:
            print("Invalid prediction")
            return numpy.zeros((height, width), numpy.uint16), \
                   numpy.zeros((height, width, 0), numpy.uint8),\
                   numpy.zeros(0, numpy.float)


        count = masks.shape[2]
        if count < 1:
            return numpy.zeros((height, width), numpy.uint16), \
                   numpy.zeros((height, width, 0), numpy.uint8),\
                   numpy.zeros(0, numpy.float)

        for i in range(count):
            masks[:, :, i] = numpy.where(masks[:, :, i] == 0, 0, 255)

        return kutils.MergeMasks(masks), masks, scores

    
    def Run(self, imagesDir, outputDir, showOutputs):
    
        imageFiles = [f for f in os.listdir(imagesDir) if os.path.isfile(os.path.join(imagesDir, f))]
        imcount = len(imageFiles)
        for index, imageFile in enumerate(imageFiles):
            print("Image:", str(index + 1), "/", str(imcount), "(", imageFile, ")")

            baseName = os.path.splitext(os.path.basename(imageFile))[0]
            imagePath = os.path.join(imagesDir, imageFile)
            image = skimage.io.imread(imagePath)

            maxdimValues = [1088]
            index = 0
            mask_allScales = numpy.zeros((len(maxdimValues),image.shape[0],image.shape[1]), numpy.uint16)
            for maxdim in maxdimValues:
                mask, masks, scores = self.Segment(pImage=image, pPredictSize=maxdim)
                count = masks.shape[2]
                print("  Nuclei (including cropped):", str(count))
                if count < 1:
                    continue

                mask_allScales[index,:,:] = mask
                index = index+1

            skimage.io.imsave(os.path.join(outputDir, baseName + ".tiff"), mask_allScales)

            if showOutputs:
                additional_visualize.display_nuclei(image=image, boxes=mrcnn_utils.extract_bboxes(masks), masks=masks, title=baseName)
    
    
  
