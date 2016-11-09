/*
 * @file RandomForest.cpp
 * @author Anuraj Kanodia
 * 
 * Builds a random forest for data classification.
 *
 * C++11 support required.
 * Read Readme.txt for further details.
 *
 */


#include "ID3.cpp"

vector<map<int,bool> > attributesMap;
map<int,bool>::iterator it;
int forestSize = 10;

void BuildForest() {
	long i,j,k;
	
	for(i=0; i<forestSize; i++) {
	
		for(j=0; j<attributesAvailable.size(); j++) {
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
			else
				delete &tempMap;		
		}
		
		node* forestRoot = new node();
		forestRoot = CreateRoot();
		//cout << "building " << i << endl;
		BuildDTree(forestRoot,true);	
		//cout << forestRoot << endl;
	}
	
}
