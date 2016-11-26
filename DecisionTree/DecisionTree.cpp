/*
 * @file DecisionTree.cpp
 * @author Anuraj Kanodia
 *
 * Decision Tree Learning with ID3 Algorithm and 
 * Random Forests.
 * 
 * DataSet used: http://archive.ics.uci.edu/ml/datasets/Adult 
 *
 * Prediction task is to determine whether a person makes over 
 * 50K a year.
 * Tree Evaluates to 'true' if a person makes over 50K a year,
 * else it evaluates to 'false'. 
 *
 * C++11 support required.
 * Read Readme.txt for further details.
 *
 */

#include "RandomForest.cpp"


int main() {
	SetAttributeNames(attributeNames);
	StoreTrainingDataSet();
	StoreTestingDataSet();	
	node* root = new node();
	root = CreateRoot();
	cout << "Building ID3 Decision Tree" << endl;	
	BuildDTree(root); //train the decision tree on given training set
	cout << "Testing the tree built" << endl;
	TestTree(root); //test the decision tree on given testing set
	cout << endl;
	cout << "Building Random Forest" << endl;
	BuildForest();
	cout << "Forest built." << endl;
	cout << "Testing the forest..." << endl;
	
	for(int k=0; k<forestSize; k++) {
		cout << k << endl;
		TestTree(myForest[k]);
	}
	
	ForestAccuracy();

	return 0;
}
