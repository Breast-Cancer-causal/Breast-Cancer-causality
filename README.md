# Breast-Cancer

## Project description

Breast cancer is the most common malignancy among women and it is the second leading cause of cancer death among women. Breast Cancer occurs as a results of abnormal growth of cells in the breast tissue, commonly referred to as a Tumor. A tumor does not mean cancer - tumors can be benign (not cancerous), pre-malignant (pre-cancerous), or malignant (cancerous). Tests such as MRI, mammogram, ultrasound and biopsy are commonly used to diagnose breast cancer performed.

The objective of this project is to apply learnt machine learning techniques to the Wisconsin Diagnostic Breast Cancer (WDBC) data. The WDBC data is class labeled, hence it will be a classification problem. The data has two classes (B=Benign, M=Malignant) and 32 attributes, or features.

Using the following machine learning techniques, we will aim to build classifiers to predict the classes of the data. For instance, 

# 1) Linear regression,

# 2) Random Forest, 

# 3) Neural network and 

# 4) K-means algorithm.

Data used:  Kaggle-Breast Cancer Prediction Dataset.

•	Attribute Information:

1.	ID number 
2.  Diagnosis (M = malignant, B = benign) (3-32)

Ten real-valued features are computed for each cell nucleus:

a.	radius (mean of distances from center to points on the perimeter)
b.	texture (standard deviation of gray-scale values)
c.	perimeter
d.	area
e.	smoothness (local variation in radius lengths)
f.	compactness (perimeter^2 / area - 1.0)
g.	concavity (severity of concave portions of the contour)
h.	concave points (number of concave portions of the contour)
i.	symmetry
j.	fractal dimension (“coastline approximation” - 1)
The mean, standard error and “worst” or largest (mean of the three largest values) of these features were computed for each image, resulting in 30 features. For instance, field 3 is Mean Radius, field 13 is Radius SE, field 23 is Worst Radius.
