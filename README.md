# Simple-Harris-Corner-Detection / RANSAC feature matching / Homography

This app shows RANSAC in action. Watch as features
are matched, be amazed while pixel groups are
 randomly selected and matched to points in stereo
images. Shocking cliffhanger ending, spoiler alert, 
all the action results in the homography! I hope you 
have about 24 hours to spare becauss that is about how
 long it takes for this algorithm to find the 
homography(with the image demonstration of the
 process), but thats just a guess. I have yet to sit
 through the entire process. However, if you have been 
interested in the nuts and bolts of RANSAC the app
is a compelling watch. 

TransA.jpg and TransB.jpg are stereo images. 
The app finds image features and displays 
them in blue and red dots. Use the spacebar to proceed 
through the first set of images. The app then produces a group
 of possible matches in TransB.jpg (blue dotted 
pixels in image on the right) for each features 
detected in TransA.jpg (green dotted pixels in
 the left image). A line is drawn from the feature
 on the left to the location in the righthand 
image the app believes is the same location on the building. Once all 
matching features have been preocessed the app
 produces the homography for the stereo images. 

To Run
Copy FeatMatchRANSAC.py / TransA.jpg, / TransB.jpg
 to the same folder and run ...

python3 FeatMatchRansac.py

expect it to run about 24 hours or more. 

ransac.png shows a feature misdetected but
 shows the general progression of the app.
