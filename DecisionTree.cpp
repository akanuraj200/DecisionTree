/*
 * Decision Tree Learning with
 * ID3 Algorithm, Random forests and 
 * AdaBoosting.
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

#include "DataSplit.cpp"

struct node {
	string splitAttributeName;
	int splitAttributeIndex;
	int numberOfChildren;
	long positiveExamples;
	long negativeExamples;
	long totalExamples;
	bool result;
	vector<bool> examplesAvailable;
	vector<pair<string,struct node*> > childNodes;
};

node* CreateRoot() {
	long i;
	vector<bool> setExamplesAvailable(noOfTrainingExamples,true);
	node* root = new node();
	root->positiveExamples = 0;
	root->negativeExamples = 0;
	root->totalExamples = 0;
	root->numberOfChildren = 0;
	root->examplesAvailable = setExamplesAvailable;
	
	for(i=0; i<noOfTrainingExamples; i++) {
		if(trainingDataSet[i].second)
			root->positiveExamples++;
		else
			root->negativeExamples++;
		
		root->totalExamples++;	
	}
	
	return root;			
}

void buildDTree(node* root) {
	if(!root->positiveExamples) {
		root->result = false;
		root->numberOfChildren = 0;
		return;
	}
	if(!root->negativeExamples) {
		root->result = true;
		root->numberOfChildren = 0;
		return;
	}
	
	long i,j;
	int splitAttribute = 0;
	int threshold = 0;
	bool isCont = false;
	string less, more;
	double baseEntropy, newEntropy, infoGain, maxInfoGain = 0.0;
	map<string,pair<long,long> > finalMap;
	map<string,pair<long,long> >::iterator it;
	vector<string> tempValues;
	baseEntropy = NodeEntropy(root->positiveExamples,root->negativeExamples);
	
	for(j=0; j<noOfAttributes; j++) {
		newEntropy = 0.0;
		
		if(attributesAvailable[j])
			continue;
		
		map<string,pair<long,long> > splitMap;
			
		if(isContValued(j)) {
			threshold = Split(j);
			less = "<= " + to_string(threshold);
			splitMap[less] = make_pair(0,0);
			more = "> " + to_string(threshold);
			splitMap[more] = make_pair(0,0);
			isCont = true;
		}
		else 
			isCont = false;
			
		for(i=0; i<noOfTrainingExamples; i++) {
			if(!root->examplesAvailable[i])
				continue;
				
			tempValues = trainingDataSet[i].first;
			
			if(isCont) {
				if(stoi(tempValues[j]) <= threshold) {
					if(trainingDataSet[i].second)
						splitMap[less].first++;
					else
						splitMap[less].second++;
				}
				else {
					if(trainingDataSet[i].second)
						splitMap[more].first++;
					else
						splitMap[more].second++;
				}
			}
			else {
				if(tempValues[j]=="")
					continue;
				it = splitMap.find(tempValues[j]); 
				if(it == splitMap.end())
					splitMap[tempValues[j]] = make_pair(0,0);
					
				if(trainingDataSet[i].second)
					splitMap[tempValues[j]].first++;
				else
					splitMap[tempValues[j]].second++;	
			}		
		}
		
		for(it=splitMap.begin(); it !=splitMap.end(); it++) {
			newEntropy += ((it->second.first + it->second.second)/(double)root->totalExamples)
				* NodeEntropy(it->second.first,it->second.second);
		}
		
		infoGain = baseEntropy - newEntropy;
		if(infoGain>maxInfoGain) {
			finalMap = splitMap;
			maxInfoGain = infoGain;
			splitAttribute = j;
		}
	}
	/*
	if(maxInfoGain==0.0) {
		if(root->positiveExamples > root->negativeExamples)
			root->result = true;
		else
			root->result = false;	
		return;
	}
	*/
	root->splitAttributeName = attributeNames[splitAttribute];
	root->splitAttributeIndex = splitAttribute;
	attributesAvailable[splitAttribute] = false;
	root->numberOfChildren = finalMap.size();
	vector<pair<string,struct node*> > children;

	if(isContValued(splitAttribute))
			isCont = true;
		else 
			isCont = false;
	
	for(it=finalMap.begin(); it!=finalMap.end(); it++) {
		node* newChild = new node();
		newChild->positiveExamples = it->second.first;
		newChild->negativeExamples = it->second.second;
		newChild->totalExamples = newChild->positiveExamples + newChild->negativeExamples;
		newChild->examplesAvailable	= root->examplesAvailable;
		if(isCont) {
			for(i=0; i<noOfTrainingExamples; i++) {
				tempValues = trainingDataSet[i].first;
				if(stoi(tempValues[splitAttribute]) <= threshold) {
					if(it->first != less)
						newChild->examplesAvailable[i] = false;	
				}
				else {
					if(it->first != more)
						newChild->examplesAvailable[i] = false;	
				}
			}
		}
		else {
			for(i=0; i<noOfTrainingExamples; i++) {
				tempValues = trainingDataSet[i].first;
				if(tempValues[splitAttribute] != it->first)
					newChild->examplesAvailable[i] = false;	
			}
		}
		
		children.push_back(make_pair(it->first,newChild));
	}
	
	root->childNodes = children; 
	
	for(i=0; i<root->numberOfChildren; i++) {
		buildDTree(children[i].second);
	}
	
	return;
}

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

int main() {
	SetAttributeNames(attributeNames);
	StoreTrainingDataSet();
	StoreTestingDataSet();	
	node* root = new node();
	root = CreateRoot();	
	buildDTree(root); //train the decision tree on given training set
	TestTree(root); //test the decision tree on given testing set
	return 0;
}
