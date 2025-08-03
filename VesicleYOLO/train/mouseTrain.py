import io

import cv2
import os


class MouseTrain:
    def __init__(self):
        self.labels = None
        self.updated = False
        self.topx, self.topy, self.botx, self.boty = 0, 0, 0, 0
        self.count = 0
        self.testval = 0
        self.lookit = False

    def rectangledraw(event, x, y, flags, param):
        global updated, topx, topy, botx, boty, wdtcut, hgtcut, viewfind, lookit, copy, labels
        if event == cv2.EVENT_LBUTTONDOWN:
            topx = x
            topy = y
            updated = True
        if event == cv2.EVENT_LBUTTONUP:
            botx = x
            boty = y
            if updated:
                cv2.rectangle(viewfind, (topx, topy), (botx, boty), (0, 0, 255), 1)
                w = botx - topx
                h = boty - topy
                cw = round(w / wdtcut, 6)
                ch = round(h / hgtcut, 6)
                cx = round((botx - (w / 2)) / wdtcut, 6)
                cy = round((topy + (h / 2)) / hgtcut, 6)
                print(f"Writing labels as 0 {cx} {cy} {cw} {ch}")
                labels.write(f"0 {cx} {cy} {cw} {ch}\n")
                lookit = False
            updated = False
        if event == cv2.EVENT_MOUSEMOVE and updated == True:
            copy = viewfind.copy()
            cv2.rectangle(copy, (topx, topy), (x, y), (0, 0, 255))
            lookit = True

    def drawManual(self):
        for file in os.listdir("organictesting"):
            # Read image in grayscale
            img = cv2.imread(f"organictesting/{file}", cv2.IMREAD_GRAYSCALE)
            # Split image into 256 different images and then we iterate over every 20. Start by making divis by 16
            hgt, wdt = img.shape
            widthborder = wdt % 16
            heightborder = hgt % 16
            adjusted = cv2.copyMakeBorder(img, 0, heightborder, 0, widthborder, cv2.BORDER_CONSTANT)
            hgt, wdt = adjusted.shape
            hgtcut = int(hgt / 16)
            wdtcut = int(wdt / 16)

            for y in range(0, 16):
                for x in range(0, 16):
                    self.count += 1
                    if self.count % 100 == 0:
                        self.testval += 1
                        cutimg = adjusted[hgtcut * y:hgtcut * (y + 1), wdtcut * x:wdtcut * (x + 1)]
                        viewfind = cv2.cvtColor(cutimg, cv2.COLOR_GRAY2BGR)
                        if self.testval % 2 == 0:
                            cv2.imwrite(fr"datasets\actualGUV\images\train\{file}{(x, y)}.jpg", cutimg)
                            self.labels = open(fr"datasets\actualGUV\labels\train\{file}{(x, y)}.txt", "a")
                        else:
                            cv2.imwrite(fr"datasets\actualGUV\images\val\{file}{(x, y)}.jpg", cutimg)
                            self.labels = open(fr"datasets\actualGUV\labels\val\{file}{(x, y)}.txt", "a")
                        cv2.namedWindow("window", cv2.WINDOW_NORMAL)
                        cv2.setMouseCallback("window", self.rectangledraw)
                        copy = viewfind.copy()
                        while True:
                            if not lookit:
                                cv2.imshow("window", viewfind)
                            else:
                                cv2.imshow("window", copy)
                            if cv2.waitKey(20) & 0xFF == 27:
                                break
                        cv2.destroyAllWindows()
