/*
 * @file RandomForest.cpp
 * @author Anuraj Kanodia
 * 
 * Builds and tests a random forest for data classification.
 *
 * C++11 support required.
 * Read Readme.txt for further details.
 *
 */

#include "ID3.cpp"

vector<node*> myForest; // random forest.
vector<bool> fExamplesAvailable(noOfTrainingExamples,false);
vector<int> prediction(noOfTestingExamples,0);
int forestSize = 50;

void SelectExamples() {
	int i,j=0;
	int subset = noOfTrainingExamples/forestSize;
	
	for(i=0; i<noOfTrainingExamples; i++) {
		fExamplesAvailable[i] = false;
	}
	
	while(j<3000) {
		i = rand() % noOfTrainingExamples;
		if(!fExamplesAvailable[i]) {
			fExamplesAvailable[i] = true;
			j++;
		}	
	}
	
	return;
}

node* CreateFRoot() {
	long i;
	node* root = new node();
	root->positiveExamples = 0;
	root->negativeExamples = 0;
	root->totalExamples = 0;
	root->numberOfChildren = 0;
	root->examplesAvailable = fExamplesAvailable;
	
	for(i=0; i<noOfTrainingExamples; i++) {
		if(!fExamplesAvailable[i])
			continue;
		
		if(trainingDataSet[i].second)
			root->positiveExamples++;
		else
			root->negativeExamples++;
		
		root->totalExamples++;	
	}
	
	return root;			
}


void BuildFTree(node* root) {
	
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
		if(!attributesAvailable[j])
			continue;
			
		newEntropy = 0.0;
		
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
			if(!root->examplesAvailable[i] or !fExamplesAvailable[i])
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
				if(tempValues[j] == "")
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
	
	if(maxInfoGain==0.0) {
		if(root->positiveExamples > root->negativeExamples)
			root->result = true;
		else
			root->result = false;	
		return;
	}
	
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
				if(!fExamplesAvailable[i])
					continue;
				
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
				if(!fExamplesAvailable[i])
					continue;
					
				tempValues = trainingDataSet[i].first;
				if(tempValues[splitAttribute] != it->first)
					newChild->examplesAvailable[i] = false;	
			}
		}
		
		children.push_back(make_pair(it->first,newChild));
	}
	
	root->childNodes = children;
	
	for(i=0; i<root->numberOfChildren; i++) {
		BuildDTree(children[i].second);
	}
	
	myForest.push_back(root);

	return;
}


void BuildForest() {
	long i,j,k;
	vector<map<int,bool> > attributesMap;
	map<int,bool>::iterator it;

	for(i=0; i<forestSize; i++) {
	
		for(j=0; j<noOfAttributes; j++) {
			attributesAvailable[j] = false;
		}
		
		while(true) {
			j=0;
			map<int,bool> tempMap;
			
			while(j!=3) {		
				k = rand() % 14;
				it = tempMap.find(k);
				
				if(it == tempMap.end()) {
					tempMap[k] = true;
					j++;
				}
			}
			
			for(j=0; j<attributesMap.size(); j++) {
				if(attributesMap[j]==tempMap) {
					tempMap.clear();
					break;
				}
			}
			
			if(tempMap.size()) {
				attributesMap.push_back(tempMap);
				for(it=tempMap.begin(); it!=tempMap.end(); it++) {
					attributesAvailable[it->first] = true;
				}
				
				break;
			}
			
		}
		
		node* forestRoot = new node();
		SelectExamples();
		forestRoot = CreateFRoot();
		BuildFTree(forestRoot);	
	}
	
}

void TestForest(node* root) {
	long i,j;
	vector<string> tempValues;
	string attribute;
	node* temp = root;
	
	for(i=0; i<noOfTestingExamples; i++) {
		if((prediction[i]==forestSize/2) or (prediction[i]==-forestSize/2))
			continue;
		
		tempValues = testingDataSet[i].first;
		
		//traverse the tree in the forest to get its predicted output.
		
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
			while((j < root->numberOfChildren) and (attribute != root->childNodes[j].first)) {
				if(attribute=="")
					break;
				j++;
			}
			
			if(j == root->numberOfChildren)
				j--;
		
			root = root->childNodes[j].second;	
		}
		
		if(root->result)
			prediction[i]++;
		else
			prediction[i]--;	
				
		root = temp;	
	
	}
	
	return;
}

void ForestAccuracy() {
	double correct=0.0,incorrect=0.0;
	double accuracy;
	long i;

	for(i=0;i<prediction.size();i++) {
		if(testingDataSet[i].second and prediction[i]>0)
			correct++;
		else if (!testingDataSet[i].second and prediction[i]<=0)	
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

