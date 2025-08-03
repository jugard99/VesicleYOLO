import sys
from VesicleYOLO.execution import GUVrun

sys.path.append('C:\Users\david\VesicleYOLO')
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
