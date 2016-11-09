/*
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

void TestTree(node* root) {
	long i,j;
	double correct = 0.0, incorrect = 0.0;
	double accuracy;
	vector<string> tempValues;
	string attribute;
	
	for(i=0; i<noOfTestingExamples; i++) {
		tempValues = testingDataSet[i].first;
		
		//traverse the decision tree to get predicted output.
		while(root->numberOfChildren) {
			j=0;
			/*
			 * Check whether split attribute is continuous valued.
			 * Set attribute accordingly.
			 */
			if(isContValued(root->splitAttributeIndex)) {
				int threshold = Split(root->splitAttributeIndex);
				if(stoi(tempValues[root->splitAttributeIndex]) <= threshold)
					attribute = "<= " + to_string(threshold);
				else
					attribute = "> " + to_string(threshold);
			}				
			else
				attribute = tempValues[root->splitAttributeIndex];
			
			//search through all the children to find the correct child to traverse.
			while(attribute != root->childNodes[j].first) {
				if(attribute=="")
					break;
				j++;
			}
			
			root = root->childNodes[j].second;	
		}
		
		if(testingDataSet[i].second == root->result)
			correct++;
		else
			incorrect++;	
	}
	
	accuracy = (correct/(correct+incorrect))*100.0;
	cout << "Testing examples correctly classified: " << (int)correct << endl;
	cout << "Testing examples incorrectly classified: " << (int)incorrect << endl;	
	cout << "Accuracy of the Decision Tree is " << accuracy << "%." << endl;
	return;
}

void TestForest() {
	long i,j,k;
	int prediction;
	double correct = 0.0, incorrect = 0.0;
	double accuracy;
	vector<string> tempValues;
	string attribute;
	node* root;
	
	for(i=0; i<noOfTestingExamples; i++) {
		prediction = 0;
		tempValues = testingDataSet[i].first;
		
		for(k=0; k<forestSize; k++) {
			root = myForest[k];
			//traverse the kth tree in the forest to get its predicted output.
			while(root->numberOfChildren) {
				/*
				 * Check whether split attribute is continuous valued.
				 * Set attribute accordingly.
				 */
				if(isContValued(root->splitAttributeIndex)) {
					int threshold = Split(root->splitAttributeIndex);
					if(stoi(tempValues[root->splitAttributeIndex]) <= threshold)
						attribute = "<= " + to_string(threshold);
					else
						attribute = "> " + to_string(threshold);
				}				
				else
					attribute = tempValues[root->splitAttributeIndex];
				
				j=0;
				//search through all the children to find the correct child to traverse.
				while((j<root->numberOfChildren)and(attribute != root->childNodes[j].first)) {
					if(attribute=="")
						break;
					j++;
				}
				
				if(j==root->numberOfChildren)
					j--;
				
				root = root->childNodes[j].second;	
			}
			
			if(root->result)
				prediction++;
			else
				prediction--;	
		}
		
		if(testingDataSet[i].second and prediction>0)
				correct++;
		else if(!testingDataSet[i].second and prediction<=0)
				correct++;
		else
			incorrect++;			
	}
	
	accuracy = (correct/(correct+incorrect))*100.0;
	cout << "Testing examples correctly classified: " << (int)correct << endl;
	cout << "Testing examples incorrectly classified: " << (int)incorrect << endl;	
	cout << "Accuracy of the Decision Tree is " << accuracy << "%." << endl;
	return;
}

int main() {
	SetAttributeNames(attributeNames);
	StoreTrainingDataSet();
	StoreTestingDataSet();	
	node* root = new node();
	root = CreateRoot();
	cout << "Building ID3 Decision Tree" << endl;	
	BuildDTree(root,false); //train the decision tree on given training set
	cout << "Testing the tree built" << endl;
	TestTree(root); //test the decision tree on given testing set
	cout << endl;
	cout << "Building Random Forest" << endl;
	BuildForest();
	cout << "Forest built." << endl;
	cout << "Testing the forest..." << endl;
	TestForest();
	cout << endl;
	return 0;
}
