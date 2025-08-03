import math
import numpy as np
from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
from statistics import stdev
import csv
# Scroll to the bottom for a short tutorial on how to use the program.


class GUVrun:
    def __init__(self, directory, model):
        # Folder directory for img files
        self.directory = directory
        # Filepath to weights
        self.model = YOLO(model)
        # Master diameter list for all images
        self.totaldiameter = []
        # Master floppiness list for all images
        self.totalflop = []
        # Master dataset (exported to csv)
        self.data = []
        # Note about reliability of information on varying image sizes
        print("\nAt the time of writing this, all calculations done by this program are set on a very specific scale"
              "that corresponds to 5x images with a resolution of 4912 by 3264.\nAlthough GUV count and visual bounding"
              " box locations may still remain reliable, all returned calculations that depend on ratio of pixels to "
              "measurements WILL NOT. ")

    def runwith(self, diameter=True, floppiness=True, surfarea=True, totalplot=True, individualplot=False):
        # Iterate through all files in directory
        for file in tqdm(range(len(os.listdir(self.directory)))):
            # Individual datasets (for each image)
            indata = {"folder": self.directory, "filename": sorted(os.listdir(self.directory))[file], "mean floppiness": 0,
                      "total surface area": 0, "SA D<10": 0, "SA D>=10": 0, "guv yield": 0, "mean diameter": 0}
            # List of diameters of GUVs in any image
            diameterlist = []
            # List of "floppiness index" values of GUVs in any image
            floplist = []
            # List of surface areas for image. Not technically necessary but prevents error bar which is annoying
            surflist = []
            lessurf = []
            moresurf = []
            # GUV Count for image
            guvcount = 0
            # Read imagepath in grayscale
            imgpath = f"{self.directory}/{sorted(os.listdir(self.directory))[file]}"
            img = cv2.imread(imgpath, cv2.IMREAD_GRAYSCALE)
            # Check image exists
            assert img is not None, f"Image path {self.directory}\\{file} is invalid."
            # Read image width and height
            hgt, wdt = img.shape[0], img.shape[1]
            # Calculate difference from standard size (4912,3264) and create adjusted image
            wdif = wdt - 4912
            hdif = hgt - 3264
            if wdif != 0 and hdif != 0:
                print("Due to nonstandard image size (preferred is (4912,3264)), "
                      "Image is automatically adjusted. Results may vary.")
                adj = cv2.copyMakeBorder(img, 0, hdif, 0, wdif, cv2.BORDER_CONSTANT)
            else:
                adj = img.copy()
            # Intervals through which to slice image and iterate. The iteration process is slow, and will eventually
            # be made obsolete through either more thorough training or combining datasets. The reasoning behind
            # splitting the image into 256 parts is its ability to simplify the training process.
            wdtcut, hgtcut = 307, 204
            for x in range(0, 16):
                for y in range(0, 16):
                    # Create and write temporary sliced image (1/256 normal size)
                    tempimg = adj[hgtcut * y:hgtcut * (y + 1), wdtcut * x:wdtcut * (x + 1)]
                    cv2.imwrite("temp.jpg", tempimg)
                    viewfind = cv2.cvtColor(tempimg, cv2.COLOR_GRAY2BGR)
                    # Get bounding box and class definitions for results
                    results = self.model.predict("temp.jpg", verbose=False)
                    # Iterate through all bounding boxes in sliced image
                    for i in results:
                        # Get bounding box data
                        boxes = i.boxes
                        for box in boxes:
                            # Up GUV count.
                            guvcount += 1
                            # Get top left and bottom right coordinates for bounding boxes
                            b = box.xyxy[0]
                            # Calculate width and height of bounding boxes
                            nuwdt = int(b[2] - b[0])
                            nuhgt = int(b[3] - b[1])
                            # Get average "diameter" of GUV (ignoring circularity)
                            avgdiameter = (abs(nuwdt) + abs(nuhgt)) / 2
                            # Calculate diameter size with adjusted values based on 5x scale
                            avgdiameter = (1000 / 1320) * avgdiameter
                            # Append to diameter list
                            diameterlist.append(avgdiameter)
                            # Calculate "floppiness" of guvs by taking standard deviation of width and height
                            # from average diameter standardized with respect to average diameter
                            if floppiness:
                                flopindex = stdev([nuwdt, nuhgt]) / avgdiameter
                                # Append floppiness index to list
                                floplist.append(flopindex)
            # Surface area calculations done after all GUVs found for simplicity, using surface area of sphere
            if surfarea:
                surflist = [4 * math.pi * ((x / 2) ** 2) for x in diameterlist]
                # Surface area for diameter < 10
                lessurf = [4 * math.pi * ((x / 2) ** 2) for x in diameterlist if x < 10]
                # Surface area for diameter >= 10
                moresurf = [4 * math.pi * ((x / 2) ** 2) for x in diameterlist if x >= 10]
            # Plotting individual image histograms at request with some binwidth
            if individualplot:
                # Arbitrary calculation for bin width, just thought it looked nice generally.
                # This can be changed manually.
                if diameter:
                    binwidth = np.mean(diameterlist) / 21
                    plt.hist(diameterlist, bins=np.arange(min(diameterlist), max(diameterlist) + binwidth, binwidth))
                    plt.show()
                # Repeat for all data
                if floppiness:
                    binwidth = np.mean(floplist) / 21
                    print(floplist)
                    plt.hist(floplist, bins=np.arange(min(floplist), max(floplist) + binwidth, binwidth))
                    plt.show()
            # Set data based on selected parameters
            if diameter:
                indata["mean diameter"] = np.mean(diameterlist)
            if floppiness:
                indata["mean floppiness"] = np.mean(floplist)
            if surfarea:
                indata["total surface area"] = np.sum(surflist)
                indata["SA D<10"] = np.sum(lessurf)
                indata["SA D>=10"] = np.sum(moresurf)
            # Write GUV yield regardless
            indata["guv yield"] = guvcount
            # Append all data to csv formatted dataset
            self.data.append(indata)
            # Append floppiness for this image to master list
            self.totalflop.extend(floplist)
            # Append diameters for this image to master list
            self.totaldiameter.extend(diameterlist)
        # Write data to CSV format
        with open('DATA.csv', 'w', newline='') as csvfile:
            fieldnames = ["folder","filename","mean floppiness","total surface area","SA D<10","SA D>=10","guv yield",
                          "mean diameter"]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.data)
        # Plot total floppiness,diameter over all images upon request (defaults to True)
        if totalplot:
            # Plot total values across all images
            if diameter:
                binwidth = np.mean(self.totaldiameter) / 21
                plt.hist(self.totaldiameter, bins=np.arange(min(self.totaldiameter), max(self.totaldiameter) + binwidth,
                                                            binwidth))
                plt.show()
            # Repeat for all data
            if floppiness:
                binwidth = np.mean(self.totalflop) / 21
                plt.hist(self.totalflop, bins=np.arange(min(self.totalflop), max(self.totalflop) + binwidth, binwidth))
                plt.show()

    def listwrite(self,diameter=True, floppiness=True, surfarea=True, totalplot=True):
        # Overarching lists
        fulldiameter = []
        fullflop = []
        fullsurf = []
        fullguvs = []
        meansurf = []
        for file in tqdm(range(len(os.listdir(self.directory)))):
            # Average per image lists
            partdiameter = []
            partflop = []
            # Individual datasets (for each image)
            indata = {"folder": self.directory, "filename": sorted(os.listdir(self.directory))[file], "mean floppiness": 0,
                      "total surface area": 0, "SA D<10": 0, "SA D>=10": 0, "guv yield": 0, "mean diameter": 0}
            # List of diameters of GUVs in any image
            diameterlist = []
            # List of "floppiness index" values of GUVs in any image
            floplist = []
            # List of surface areas for image. Not technically necessary but prevents error bar which is annoying
            surflist = []
            lessurf = []
            moresurf = []
            # GUV Count for image
            guvcount = 0
            # Read imagepath in grayscale
            imgpath = f"{self.directory}/{sorted(os.listdir(self.directory))[file]}"
            img = cv2.imread(imgpath, cv2.IMREAD_GRAYSCALE)
            # Check image exists
            assert img is not None, f"Image path {self.directory}\\{file} is invalid."
            # Read image width and height
            hgt, wdt = img.shape[0], img.shape[1]
            # Calculate difference from standard size (4912,3264) and create adjusted image
            wdif = wdt - 4912
            hdif = hgt - 3264
            if wdif != 0 and hdif != 0:
                print("Due to nonstandard image size (preferred is (4912,3264)), "
                      "Image is automatically adjusted. Results may vary.")
                adj = cv2.copyMakeBorder(img, 0, hdif, 0, wdif, cv2.BORDER_CONSTANT)
            else:
                adj = img.copy()
            # Intervals through which to slice image and iterate. The iteration process is slow, and will eventually
            # be made obsolete through either more thorough training or combining datasets. The reasoning behind
            # splitting the image into 256 parts is its ability to simplify the training process.
            wdtcut, hgtcut = 307, 204
            for x in range(0, 16):
                for y in range(0, 16):
                    # Create and write temporary sliced image (1/256 normal size)
                    tempimg = adj[hgtcut * y:hgtcut * (y + 1), wdtcut * x:wdtcut * (x + 1)]
                    cv2.imwrite("temp.jpg", tempimg)
                    viewfind = cv2.cvtColor(tempimg, cv2.COLOR_GRAY2BGR)
                    # Get bounding box and class definitions for results
                    results = self.model.predict("temp.jpg", verbose=False)
                    # Iterate through all bounding boxes in sliced image
                    for i in results:
                        # Get bounding box data
                        boxes = i.boxes
                        for box in boxes:
                            # Up GUV count.
                            guvcount += 1
                            # Get top left and bottom right coordinates for bounding boxes
                            b = box.xyxy[0]
                            # Calculate width and height of bounding boxes
                            nuwdt = int(b[2] - b[0])
                            nuhgt = int(b[3] - b[1])
                            # Get average "diameter" of GUV (ignoring circularity)
                            avgdiameter = (abs(nuwdt) + abs(nuhgt)) / 2
                            # Calculate diameter size with adjusted values based on 5x scale
                            avgdiameter = (1000 / 1320) * avgdiameter
                            # Append to diameter list
                            diameterlist.append(avgdiameter)
                            # Calculate "floppiness" of guvs by taking standard deviation of width and height
                            # from average diameter standardized with respect to average diameter
                            if floppiness:
                                flopindex = stdev([nuwdt, nuhgt]) / avgdiameter
                                # Append floppiness index to list
                                floplist.append(flopindex)
            fullguvs.append(guvcount)
            # Surface area calculations done after all GUVs found for simplicity, using surface area of sphere
            if surfarea:
                surflist = [4 * math.pi * ((x / 2) ** 2) for x in diameterlist]
                # Surface area for diameter < 10s
                lessurf = [4 * math.pi * ((x / 2) ** 2) for x in diameterlist if x < 10]
                # Surface area for diameter >= 10
                moresurf = [4 * math.pi * ((x / 2) ** 2) for x in diameterlist if x >= 10]
                meansurf.append(np.mean(surflist))
                fullsurf.append(np.sum(surflist))
            fulldiameter.append(np.mean(diameterlist))
            fullflop.append(np.mean(floplist))
        fig, ax = plt.subplots()
        """
        Glass bar chart (treatment vs guv yield)
        
        bars = ["Casein + UV","Wbsa + UV","Pbsa + UV","Casein","Wbsa","Pbsa","UV","None"]
        ax.bar(bars, fullguvs)
        plt.xlabel("Glass treatment")
        plt.ylabel("GUV yield")
        plt.show()
        """

        """
        Plastic bar chart (treatment vs guv yield)
        
        bars = ["Casein + UV","Wbsa + UV","Pbsa + UV","Casein","Wbsa","Pbsa","UV","None"]
        ax.bar(bars, fullguvs)
        plt.xlabel("Plastic treatment")
        plt.ylabel("GUV yield")
        plt.show()
        """
        """
        Yield and lipid amt
        
        bars = ["5 ul","10 ul","20 ul"]
        ax.bar(bars,fullguvs)
        plt.xlabel("Lipid amount")
        plt.ylabel("GUV yield")
        plt.show()
        """
        """
        Surface area and composition/yield
        

        bar_colors = ['tab:red', 'tab:blue', 'tab:red', 'tab:orange']
        bars = ["10 PC","9:1","8:2","75:25",'7:3',"6:4"]
        ax.bar(bars,fullguvs)
        plt.xlabel("Lipid composition")
        plt.ylabel("GUV yield")
        plt.show()
        fig, ax = plt.subplots()
        ax.bar(bars, meansurf)
        plt.xlabel("Lipid composition")
        plt.ylabel("Average GUV surface area")
        plt.show()
        fig, ax = plt.subplots()
        ax.bar(bars, fullsurf)
        plt.xlabel("Lipid composition")
        plt.ylabel("Total GUV surface area")
        plt.show()
        """
        """
        Diameters and glucose/sucrose content
        """
        bars = [-25,-20,-15,-10,-5,0,5,10,15,20,25]
        fig, ax = plt.subplots()
        ax.bar(bars, fulldiameter)
        plt.xlabel("Glucose content - sucrose content")
        plt.ylabel("Average GUV Diameter")
        plt.show()
        fig, ax = plt.subplots()

        ax.bar(bars, fullflop)
        plt.xlabel("Glucose content - sucrose content")
        plt.ylabel("Average GUV \"Flop\" (opposite of circularity)")
        plt.show()


guvs = GUVrun(r"C:\Users\david\PycharmProjects\pythonProject2\ultralyticsWork\glucoseimages","best (1).pt")

guvs.listwrite()

r"""
HOW TO USE:
Create an instance of the GUVrun class with parameters of the folder with all your images as well as the path to the 
model weights, like so:

guvs = GUVrun(r"C:\Users\david\PycharmProjects\pythonProject2\ultralyticsWork\classtest","best.pt")

Then, do the function:
guvs.runwith()

Inside the parentheses, you can do a couple of arguments:
diameter, floppiness, surfarea, totalplot, individualplot

These are all either "True" or "False".
The first three determine whether those values will be output into the final CSV file (if False, then they are just 0.)

The last two determine when (if at all) the values are plotted. 

If totalplot = True, you will see the diameters and floppiness values plotted in histograms for all the images once all
have been processed.

If individualplot = True, you will see the diameters and floppiness values of each individual image plotted once that 
image has been processed.

The default arguments are:
diameter=True, floppiness=True, surfarea=True, totalplot=True, individualplot=False

If there are any bugs or if you have any questions, message me. It is entirely possible I made a mistake in bringing
this together. 
"""
