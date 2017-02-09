# Decision Tree And Visualization

Following modules used  
* DecisionTreeClassifier (iris.py)  
* DecisionTreeRegressor  
* RandomForestClassifier (iris.py)
* ExtraTreesRegressor (facecompletion.py)
* LinearRegression (facecompletion.py)
  
Problems  
* Iris classification problem - 3 way classification
* Face completion - Complete bottom half of face given upper half
  
Just run python filename.py to get output.  

Here is the output from the facecompletion program.

<div><img src = 'Face completion.png' width = '400' align = 'center'/></div>

## Discussion
### Face Completion
We can see the kNN has just picked up the bottom half of another person whose upper face matched with our test subject. 
Extra trees have produced an averaging of results from each tree, thus giving a blurry look to it whereas Linear regression has produced noisy, rough looking images.
  
## Further Readings
* http://scikit-learn.org/stable/modules/tree.html  
* http://scikit-learn.org/stable/auto_examples/plot_multioutput_face_completion.html#sphx-glr-auto-examples-plot-multioutput-face-completion-py  
* https://www.analyticsvidhya.com/blog/2014/06/introduction-random-forest-simplified/
* Olivetti Faces dataset - http://www.cl.cam.ac.uk/research/dtg/attarchive/facedatabase.html

