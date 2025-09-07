# PennAiRSoftwareChallenge
Completing the PennAiR software challenge to apply to the software team

In general the .ipynb files are used for initial testing and run simultenously on 12 selected frames from a video or a photo if one was given. The .py files have more or less the same code as the .ipynb files but run on the entire video at the same time.

The main.ipynb and test.ipynb files contain the original code to process the easy image.
The main.py file contains the original code for running on the easy video.
The mainHard.py and HSVsplitHard.py files contain different ways of processing the hard video.
 - The mainHard.py file uses a simple canny + dilate + erode + contour algorithm which gives a shaky binary mask output
The hsvTesting.ipynb file contains the same code as the HSVsplitHard.py file but in Jupyter cells
 - The HSVsplitHard.py file contains a somewhat hard-coded method that performs significantly better but relies on the polygons being a different color than the background to get crisp edges. The trapezoid (since it's white and black and hence has a saturation of zero) has slightly worse edges and instead uses Otsu's method after an HSV binary mask and some Gaussian blurring. The point is to wash out the noise and filter it from the larger trapezoid. I then union the color mask and grayscale mask together, detect contours, filter small and oddly shapes contours, and finally overlay a canny edge detection output of this binary mask onto the original image to get green outlines around the polygons.
 - hsvTesting runs at ~35fps, best one I got was 37.02 fps