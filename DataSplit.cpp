/* 
 * @file DataSplit.cpp
 * @author Anuraj Kanodia
 *
 * Discritises continuous valued attributes in the training examples.
 * Threshold with minimum entropy is selected for the split.
 * 
 * Contains the NodeEntropy() function. 
 *
 */

#include "StoreData.cpp"

// returns the entropy of the node/split. 
double NodeEntropy(double positive, double negative) {
	// base case.
	if(negative == 0 or positive == 0)
		return 0;
	double total = positive + negative;
	double entropy;
	entropy = 1*(((positive/total)*(log2(positive/total)))
		+ ((negative/total)*(log2(negative/total))));
	return -entropy;
}

// returns whether given attribute is continuous valued.
bool isContValued(int i) {
	if(i==0 or i==2 or i==4 or i==10 or i==11 or i==12)
		return true;
	else
		return false;	
}

// returns the threshold to split the attribute on.
int Split(int index) {
	long i;
	int neg, age, pos;
	double cumPos = 0.0;
	double cumNeg = 0.0;	
	double totPos = 0.0;
	double totNeg = 0.0;	
	double minEntropy = 100.0;
	double entropy = 0.0;
	double total;
	int split;
	
	map<int,pair<int,int> > mymap;
	map<int,pair<int,int> >::iterator it;
	vector<pair<int,bool> > splitData;
	vector<string> example;
	
	for(i=0; i<noOfTrainingExamples; i++) {
		example = trainingDataSet[i].first;
		splitData.push_back(make_pair(stoi(example[index]),trainingDataSet[i].second));
	}
	
	for(i=0; i<splitData.size(); i++) {
		it = mymap.find(splitData[i].first);
		if(it == mymap.end())
			mymap[splitData[i].first] = make_pair(0,0);
		
		if(splitData[i].second)
			(mymap[splitData[i].first].first)++;
		else
			(mymap[splitData[i].first].second)++;	
	}
	
	for(it=mymap.begin(); it!=mymap.end();it++) {
		totPos += it->second.first;
		totNeg += it->second.second;
	}

	for(it=mymap.begin();it!=mymap.end();it++) {
		cumPos += it->second.first;
		cumNeg += it->second.second;
		total = cumPos+cumNeg;
		entropy = ((total)/32561.0)*NodeEntropy(cumPos,cumNeg) + ((32561.0-total)/32561.0)*(NodeEntropy(totPos-cumPos,totNeg-cumNeg));
		
		if(entropy < minEntropy) {
			minEntropy = entropy;
			split = it->first;
		}	
	}
	
	return split;
}
