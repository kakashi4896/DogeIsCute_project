# TIME SERIES FORCASTING MACHINE LEARNING MEETS SPACE WEATHER


## HIGH-LEVEL SUMMARY
Our project is to use machine learning to predict the space weather which mainly uses Kp index as the standard criteria. The main schematic of our project is time series forcasting machine learning. Refer to most space articles, we use 180 minutes magnetic field vectors Bx, By and Bz with other dimensionless raw data introduced in DSCOVR to predict the Kp index for the next three hours as our main object. Before inputting the raw data into our model, we calculate their pearson correlations to classify data into eight modalities, and then do the time series forcasting machine learning. Finally, we use multi-task learning to achieve the tasks, predicting Kp values and classifying the severity of solar storms. Kp index is a criterion for people to judge the strength of solar winds, which is important to astronauts and people living in high-latitude places. With our predicted results, they are able to live more safely and enjoy aurora.


## PROJECT DETAILS
### Function:
- Predicts the Kp value for the next 3 hours based on input magnetic field vectors.

### Purpose:
- Predict Kp values to facilitate predictions related to auroras or disasters.

### Software/Language Used:
- Python
- PyTorch


## SPACE AGENCY DATA
### omniweb
- https://omniweb.gsfc.nasa.gov/form/dx1.html


## References
### omniweb
- https://omniweb.gsfc.nasa.gov/form/dx1.html

### nasaomnireader
- https://github.com/lkilcommons/nasaomnireader

### PatchTST
- https://github.com/yuqinie98/PatchTST