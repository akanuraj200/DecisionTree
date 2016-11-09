/* 
 * @file StoreData.cpp
 * @author Anuraj Kanodia
 *
 * Stores the training dataset and the testing dataset in memory.
 *
 * DataSet used: http://archive.ics.uci.edu/ml/datasets/Adult 
 * 
 * Read Readme.txt for further details.
 *
 */
 
#include<bits/stdc++.h>
using namespace std;

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

long noOfTrainingExamples = 32561;
long noOfTestingExamples = 16281;
int noOfAttributes = 14;

vector<string> attributeNames(noOfAttributes); // stores the name of the attributes.
vector<pair<vector<string>,bool> > trainingDataSet; // stores the training dataset.
vector<pair<vector<string>,bool> > testingDataSet; // stores the testing data.
vector<bool> attributesAvailable(noOfAttributes,true); // stores the availability of attributes for splitting.
vector<node*> myForest; // random forest.

// stores the attribute names in a global vector for convenience.
void SetAttributeNames(vector<string> &v) {
	v[0] = "age"; v[1] = "workclass"; v[2] = "fnlwgt";
	v[3] = "education"; v[4] = "education-num"; v[5] = "martial-status";
	v[6] = "occupation"; v[7] = "relationship"; v[8] = "race";	
	v[9] = "sex"; v[10] = "capital-gain"; v[11] = "capital-loss";
	v[12] = "hours-per-week"; v[13] = "native-country";
	return;
}

// stores training data.
void StoreTrainingDataSet() {
	long i,j;
	vector<string> example;
	bool output = true;
	string input;
	
	for(i=0;i<noOfTrainingExamples;i++) {
		for(j=0;j<noOfAttributes; j++) {
			cin >> input;
			input.pop_back(); // removes the comma(,) at the end of each attribute.
			if(input[0] == '?') // we ignore missing attributes completely.
				input.clear();
			example.push_back(input);
		}
		// store the output of the training example.
		cin >> input;
		if(input[0] == '>')
			output = true;
		else 
			output = false;
			
		trainingDataSet.push_back(make_pair(example,output)); 	
		example.clear();
	}
	
	return;
}

void StoreTestingDataSet() {
	long i,j;
	vector<string> example;
	bool output = true;
	string input;
	
	for(i=0;i<noOfTestingExamples;i++) {
		for(j=0;j<noOfAttributes; j++) {
			cin >> input;
			input.pop_back();// removes the comma(,) at the end of each attribute.
			if(input[0] == '?') // we ignore missing attributes completely.
				input.clear();
			example.push_back(input);
		}
		// store the output of the testing example.
		cin >> input;
		if(input[0] == '>')
			output = true;
		else 
			output = false;
			
		testingDataSet.push_back(make_pair(example,output)); 	
		example.clear();
	}
	
	return;
}
