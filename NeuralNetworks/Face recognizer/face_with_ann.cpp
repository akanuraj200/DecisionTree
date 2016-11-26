/*
 * @file face_with_ann.cpp
 * @authors Himanshu Bagga, Stuti Sabharwal 
 * 
 * Neural Network for face recognition. 
 * task: given an image, identify the person out of the 20 persons
 * given in dataset.
 *
 */


#include<bits/stdc++.h>

#define no_hidden_unit 30     //Number of units in the hidden layer

using namespace std;

typedef struct _PGMData {      //Structure to store data of one pgm file    
    int row;
    int col;
    int max_gray;
    int matrix[30][32];
} PGMData;
typedef struct input{                  //Structure for one input type object
    double value;
    double weight[no_hidden_unit];
} input;
typedef struct output{               //Structure for one output type object
    double suminput;
    double out;
} output;
typedef struct hidden{                 //Structure of one hidden unit type object
    double suminput;
    double out;
    double weight[20];
} hidden;
double squash(double x)                  //Sigmoid function
{
    return (1/(1+exp(-x)));
}

PGMData* readPGM(const char* file_name, PGMData *data)       //Function to take up the pgm file name
{                                                            //and store it in an object of PGMData*
    
    FILE *pgmFile;
    char version[3];
    int i, j;
    int lo, hi;
 
    pgmFile = fopen(file_name, "rb");
    
    fgets(version, sizeof(version), pgmFile);
    
    fscanf(pgmFile, "%d", &data->col);
    fscanf(pgmFile, "%d", &data->row);
    fscanf(pgmFile, "%d", &data->max_gray);
    fgetc(pgmFile);
        for (i = 0; i < data->row; ++i)
            for (j = 0; j < data->col; ++j) {
                lo = fgetc(pgmFile);
                data->matrix[i][j] = lo;
            }
       
    fclose(pgmFile);
    return data;
 
}
int main()
{
    srand(2);                         // seed value for random weights
    PGMData* pgm=(PGMData*)malloc(sizeof(PGMData));        //object for storing PGM data
    char* file=(char*)malloc(500);
    ifstream myfile;
    PGMData* pgminput[1000];
    double alpha=0.3;                        //constant for momentum
    int i=0;
    double accuracy=0,olddelw[30][32][no_hidden_unit]={0};                  //delta w for previous iteration
    double olddelwhidden[no_hidden_unit][20]={0};           //delta w of hidden unit for previous iteration
    input in[30][32];                             //960 objects for input type, one for each pixel
    hidden h[no_hidden_unit];                      //Hidden units declaration
    output o[20];                               //array of 20 output units
    char name[3];                            //Output unit declaration
    int faceid;                                     
    double target[20],delw,n=0.3,delout[20];    //n is the learning rate, declaration of array of target outputs and delout(error term for output unit)
    int u,v,a,b,z,t,k,j;    //errorinpred stores total number of wrong predictions
    for(z=0;z<no_hidden_unit;z++)             //Initializing to random weights in the range of (-0.5,0.5)
    {
        for(u=0;u<30;u++)
       {
         for(v=0;v<32;v++)
         {
            in[u][v].weight[z]=(rand()%100) + (-50);
            in[u][v].weight[z]/=100;
          
         }
       }
    }
    for(u=0;u<no_hidden_unit;u++)          //Initializing weights of hidden units    
    {
      for(t=0;t<20;t++)
      	{
        	h[u].weight[t]=(rand()%100) + (-50);       //random weights in the range (-0.5,0.5)
        	h[u].weight[t]/=100;
    	}
    }
    int epoch=100,train;          //number of training iterations
    string line[1000],inputline;
    int no_of_inputs=0;
    myfile.open("straighteven_train.list");     //Open the file with training list
    

    while(getline(myfile,inputline))
    {
    	line[no_of_inputs]=inputline;
    	no_of_inputs++;
    }
    cout<<no_of_inputs<<endl;
    

    myfile.close();     //Close the file




for(train=0;train<epoch;train++)
{
	int testinput=0;
    cout << train << endl ;	
	while(testinput<no_of_inputs)  //Read test inputs while this condition holds true
	{    
      
    	int fr=0;
    	while(line[testinput][fr]!='\0')       
        {
            
            file[fr]=line[testinput][fr];
            
            fr++;
        }
       

        file[fr]='\0';
        //cout<<file<<endl;

        for(k=0;k<3;k++)
        {
           name[k]=line[testinput][k+8];         //Storing the three characters after the 8th character in name
           //cout<<name[k];
        }
        //cout<<endl;
        if(!strcmp(name,"an2"))             //Comparing name with each string and setting the faceid if it is true
        	faceid=0;
        else if(!strcmp(name,"at3"))
        	faceid=1;
        else if(!strcmp(name,"bol"))
        	faceid=2;
        else if(!strcmp(name,"bpm"))
        	faceid=3;
        else if(!strcmp(name,"ch4"))
        	faceid=4;
        else if(!strcmp(name,"che"))
        	faceid=5;
        else if(!strcmp(name,"cho"))
        	faceid=6;
        else if(!strcmp(name,"dan"))
        	faceid=7;
        else if(!strcmp(name,"gli"))
        	faceid=8;
        else if(!strcmp(name,"kar"))
        	faceid=9;
        else if(!strcmp(name,"kaw"))
        	faceid=10;
        else if(!strcmp(name,"kk4"))
        	faceid=11;
        else if(!strcmp(name,"meg"))
        	faceid=12;
        else if(!strcmp(name,"mit"))
        	faceid=13;
        else if(!strcmp(name,"nig"))
        	faceid=14;
        else if(!strcmp(name,"pho"))
        	faceid=15;
        else if(!strcmp(name,"saa"))
        	faceid=16;
        else if(!strcmp(name,"ste"))
        	faceid=17;
        else if(!strcmp(name,"sz2"))
        	faceid=18;
        else if(!strcmp(name,"tam"))
        	faceid=19;
        for(u=0;u<20;u++)           
        {   
        	if(u==faceid)                  //Setting the target output to 1 if it matches the faceid
        		target[u]=1;
        	else
        		target[u]=0;
        }
        
        pgm=readPGM(file,pgm);
        

        for(u=0;u<no_hidden_unit;u++)           //Initializing suminput to 0 for each hidden unit
            h[u].suminput=0;
        for(u=0;u<20;u++)                        //Intitalizing suminput to 0 for each output unit
        	o[u].suminput=0;
        int k=0;
        for(t=0;t<30;t++)
        {
            for(j=0;j<32;j++)
            { 
                in[t][j].value=pgm->matrix[t][j];       //Assigning input values
                in[t][j].value/=pgm->max_gray;          //Normalizing input values
            }            
        } 
        
        for(z=0;z<no_hidden_unit;z++)
        {
            
             for(t=0;t<30;t++)
            {
                
                for(j=0;j<32;j++)
                    { 
                       h[z].suminput=h[z].suminput+(in[t][j].weight[z]*in[t][j].value);     //Calculating suminput for each hidden unit
                    }            
            }
            h[z].out=squash(h[z].suminput);         //Squashing fuction to calculate output of hidden unit
         
        }
        for(t=0;t<20;t++)
        {
        	for(z=0;z<no_hidden_unit;z++)
            	{
            		o[t].suminput=o[t].suminput+(h[z].out*h[z].weight[t]);       //Calculating suminput for each Output unit
            	}
            o[t].out=squash(o[t].suminput);         //Calculate output using Squashing function for each unit
        }
        for(t=0;t<20;t++)
        	delout[t]=(o[t].out)*(1-o[t].out)*(target[t]-o[t].out);        //Calculating error for each output unit
        int delh[no_hidden_unit]={0}; 
        for(z=0;z<no_hidden_unit;z++)           
        {
        	int sumofweight=0;
        	for(t=0;t<20;t++)
        	{
            	sumofweight+=h[z].weight[t]*delout[t];
 			}
 			delh[z]=(h[z].out)*(1-h[z].out)*sumofweight;        //Calculating error for each hidden unit       
        }
        for(z=0;z<no_hidden_unit;z++)
        {
             for(t=0;t<30;t++)
            {
                for(j=0;j<32;j++)
                    { 
                       delw=n*delh[z]*in[t][j].value;       //Calculating delw
                       delw=delw+(alpha*olddelw[t][j][z]);  //Adding Momentum
                       olddelw[t][j][z]=delw;
                       in[t][j].weight[z]+=delw;        //Updating weights
                    }            
            }   
        }
        for(t=0;t<20;t++)
        {	
        	for(z=0;z<no_hidden_unit;z++)
        	{

            	delw=n*delout[t]*h[z].out;         //Calculating delw
            	delw=delw+(alpha*olddelwhidden[z][t]);     //Adding Momentum
            	olddelwhidden[z][t]=delw;      
            	h[z].weight[t]+=delw;      //Updating weights for Hidden units     
 		    }    
		} 
		testinput++;   //Incrementing testinput
    }

    
}
        //TESTING
    int testnum=0;                             //Number of test cases
    myfile.open("straighteven_train.list");
    //myfile.open("straighteven_test1.list");     //Open the file with test inputs
   //myfile.open("straighteven_test2.list");
    while ( getline (myfile,inputline) )
    {
        testnum++;
        int prediction[20];     //Prediction for 20 output units
        int trueout[20];        //True output for 20 output units
        int u,t,j;
        for(u=0;u<no_hidden_unit;u++)
            h[u].suminput=0;            //Intitializing suminput to 0 for each hidden unit
        for(u=0;u<20;u++)
        	o[u].suminput=0;       //Intitializing suminput to 0 for each output unit
        int k=0;
        while(inputline[k]!='\0')
        {
            
            file[k]=inputline[k];
            k++;
        }

        file[k]='\0';
        for(k=0;k<3;k++)
        {
           name[k]=inputline[k+8];      //Storing the three characters after the 8th character in name
           //cout<<name[k]<<endl;
        }
        if(!strcmp(name,"an2"))     //Comparing name with the string and setting faceid if it is equal
        	faceid=0;
        else if(!strcmp(name,"at3"))
        	faceid=1;
        else if(!strcmp(name,"bol"))
        	faceid=2;
        else if(!strcmp(name,"bpm"))
        	faceid=3;
        else if(!strcmp(name,"ch4"))
        	faceid=4;
        else if(!strcmp(name,"che"))
        	faceid=5;
        else if(!strcmp(name,"cho"))
        	faceid=6;
        else if(!strcmp(name,"dan"))
        	faceid=7;
        else if(!strcmp(name,"gli"))
        	faceid=8;
        else if(!strcmp(name,"kar"))
        	faceid=9;
        else if(!strcmp(name,"kaw"))
        	faceid=10;
        else if(!strcmp(name,"kk4"))
        	faceid=11;
        else if(!strcmp(name,"meg"))
        	faceid=12;
        else if(!strcmp(name,"mit"))
        	faceid=13;
        else if(!strcmp(name,"nig"))
        	faceid=14;
        else if(!strcmp(name,"pho"))
        	faceid=15;
        else if(!strcmp(name,"saa"))
        	faceid=16;
        else if(!strcmp(name,"ste"))
        	faceid=17;
        else if(!strcmp(name,"sz2"))
        	faceid=18;
        else if(!strcmp(name,"tam"))
        	faceid=19;
        for(u=0;u<20;u++)
        {
        	if(u==faceid)          //if faceid matches set trueout to 1
        		trueout[u]=1;
        	else
        		trueout[u]=0;
        }
        
        pgm=readPGM(file,pgm);
        for(t=0;t<30;t++)
        {
            for(j=0;j<32;j++)
            { 
                in[t][j].value=pgm->matrix[t][j];       //Assigning input values
                in[t][j].value/=pgm->max_gray;          //Normalizing input values
            }            
        }
        
        int z;
        for(z=0;z<no_hidden_unit;z++)
        {
            
             for(t=0;t<30;t++)
            {
                
                for(j=0;j<32;j++)
                    { 
                       h[z].suminput=h[z].suminput+(in[t][j].weight[z]*in[t][j].value);     //Calculating suminput for each Hidden unit
                    }            
            }
            h[z].out=squash(h[z].suminput);     //Calculating output using Squashing function   
        }
        for(t=0;t<20;t++)
        {
        	for(z=0;z<no_hidden_unit;z++)
            	{
            		o[t].suminput=o[t].suminput+(h[z].out*h[z].weight[t]);        //Calculating suminput for each output unit
            	}
            	o[t].out=squash(o[t].suminput);        //Calculating output using Squashing function
        }
         
 			double max=o[0].out;        //Max stores the maximum output value of the units
 			int predfaceid=0;       //predfaceid stores the predicted faceid
        for(t=1;t<20;t++)
        {
        	if(o[t].out>max)
        	{
           	 	max=o[t].out;         //Setting maximum value
        		predfaceid=t;         //Setting predfaceid if the output is maximum
        	}
        }
        switch(predfaceid)               //Printing the predicted userid
        {
            case 0:cout<<"For test case #"<<testnum<<" Max value of All outputs:"<<max<<" Name:an2i"<<endl;
                   break;
            case 1:cout<<"For test case #"<<testnum<<" Max value of All outputs:"<<max<<" Name:at33"<<endl;
                   break;
            case 2:cout<<"For test case #"<<testnum<<" Max value of All outputs:"<<max<<" Name:boland"<<endl;
                   break;
            case 3:cout<<"For test case #"<<testnum<<" Max value of All outputs:"<<max<<" Name:bpm"<<endl;
                   break;
            case 4:cout<<"For test case #"<<testnum<<" Max value of All outputs:"<<max<<" Name:ch4f"<<endl;
                   break;
            case 5:cout<<"For test case #"<<testnum<<" Max value of All outputs:"<<max<<" Name:cheyer"<<endl;
                   break;
            case 6:cout<<"For test case #"<<testnum<<" Max value of All outputs:"<<max<<" Name:choon"<<endl;
                   break;
            case 7:cout<<"For test case #"<<testnum<<" Max value of All outputs:"<<max<<" Name:danieln"<<endl;
                   break;
            case 8:cout<<"For test case #"<<testnum<<" Max value of All outputs:"<<max<<" Name:glickman"<<endl;
                   break;
            case 9:cout<<"For test case #"<<testnum<<" Max value of All outputs:"<<max<<" Name:karyadi"<<endl;
                   break;
            case 10:cout<<"For test case #"<<testnum<<" Max value of All outputs:"<<max<<" Name:kawamura"<<endl;
                   break;
            case 11:cout<<"For test case #"<<testnum<<" Max value of All outputs:"<<max<<" Name:kk49"<<endl;
                   break;
            case 12:cout<<"For test case #"<<testnum<<" Max value of All outputs:"<<max<<" Name:megak"<<endl;
                   break;
            case 13:cout<<"For test case #"<<testnum<<" Max value of All outputs:"<<max<<" Name:mitchell"<<endl;
                   break;
            case 14:cout<<"For test case #"<<testnum<<" Max value of All outputs:"<<max<<" Name:night"<<endl;
                   break;
            case 15:cout<<"For test case #"<<testnum<<" Max value of All outputs:"<<max<<" Name:phoebe"<<endl;
                   break;
            case 16:cout<<"For test case #"<<testnum<<" Max value of All outputs:"<<max<<" Name:saavik"<<endl;
                   break;
            case 17:cout<<"For test case #"<<testnum<<" Max value of All outputs:"<<max<<" Name:steffi"<<endl;
                   break;
            case 18:cout<<"For test case #"<<testnum<<" Max value of All outputs:"<<max<<" Name:sz24"<<endl;
                   break;
            case 19:cout<<"For test case #"<<testnum<<" Max value of All outputs:"<<max<<" Name:tammo"<<endl;
                   break;
               
        }
        if(predfaceid==faceid)
        	accuracy++;            //Checking the accuracy, incrementing it if predicted face id is equal to the actual one
        
        //for(t=0;t<20;t++)
        	//errorinpred+=abs(prediction[t]-trueout[t]);
    }
   accuracy/=testnum;
   accuracy*=100;    
    cout<<"Accuracy for the test file:"<<accuracy<<"%"<<endl; //Printing the accuracy of the test file
    myfile.close();
}
