/*
 * @file pose_with_ann.cpp
 * @authors Himanshu Bagga, Stuti Sabharwal 
 * 
 * Neural Network for pose recognition. 
 * task: given an image, tell the head position of the person.
 * it can have 4 values: straight, left, right and up.
 *
 */

#include<bits/stdc++.h>
#define no_hidden_unit 64   //Number of units in the hidden layer

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
    double weight[4];
} hidden;
double squash(double x)                  //Sigmoid function
{
    return (1/(1+exp(-x)));
}
#define LO(num) ((num) & 0x000000FF)
PGMData* readPGM(const char* file_name, PGMData *data)       //Function to take up the pgm file name
{                                                            //and store it in an object of PGMData*
    
    FILE *pgmFile;
    char version[3];
    int i, j;
    int lo, hi;
 
    pgmFile = fopen(file_name, "rb");
    if (pgmFile == NULL) {
        perror("cannot open file to read");
        exit(EXIT_FAILURE);
    }
 
    fgets(version, sizeof(version), pgmFile);
    if (strcmp(version, "P5")) {
        fprintf(stderr, "Wrong file type!\n");
        exit(EXIT_FAILURE);
    }
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
    srand(2);                         //seed value for random weights
    PGMData* pgm=(PGMData*)malloc(sizeof(PGMData));        //object for storing PGM data
    char* file=(char*)malloc(500);
    ifstream myfile;
    
    double alpha=0.3;                        //constant for momentum
    int i=0;
    double accuracy=0,olddelw[30][32][no_hidden_unit]={0};                  //delta w for previous iteration
    double olddelwhidden[no_hidden_unit][4]={0};        //delta w of hidden uits for previous iteration
    input in[30][32];                             //960 objects for input type, one for each pixel
    hidden h[no_hidden_unit];                      //Hidden units declaration
    output o[4];                                 //array of 4 output units
    char pose;                         //stores the first letter of the pose
    int poseid;                       //poseid, 0 for left,1 for right,2 for straight,3 for up
    double target[4],delw,n=0.3,delout[4];        //n is the learning rate
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
    for(u=0;u<no_hidden_unit;u++)              //Initializing weights for hidden units
    {
      for(t=0;t<4;t++)
        {
            h[u].weight[t]=(rand()%100) + (-50);    //Random weights in the range of (-0.5,0.5)
            h[u].weight[t]/=100;
        }
    }
    int epoch=200,train;        //Number of training iterations
    string line[1000],inputline;    
    int no_of_inputs=0;
    myfile.open("all_train.list");      //Open the file with training list

    while(getline(myfile,inputline))
    {
    	line[no_of_inputs]=inputline;      
    	no_of_inputs++;            //Incrementing Number of Inputs
    }
   
    

    myfile.close();     //Close the file
    for(train=0;train<epoch;train++)
    {
    	int testinput=0;       
        cout << train << endl ;	
    	while(testinput<no_of_inputs)          //Checking if the testinput is less than total number of inputs
    	{    
          
        	int fr=0;
        	while(line[testinput][fr]!='\0')
            {
                
                file[fr]=line[testinput][fr];
                
                fr++;
            }
           

            file[fr]='\0';
            //cout<<file<<endl;
            b=0;
            for(a=0;a<fr;a++)                  
            {
                if(b==2)                       //Storing the first character after the second underscore, which is the first letter of the pose
                {
                    pose=file[a];
                    break;
                }
                if(file[a]=='_')
                    b++;

            }
            if(pose=='l')                   //Comparing the first character of pose to l,r,s,u to assign poseids accordingly
            {
                poseid=0;
                
            }
            else if(pose=='r')
            {
                poseid=1;
                
            }
            else if(pose=='s')
            {
                poseid=2;
                
            }
            else if(pose=='u')
            {
                poseid=3;
                
            }
            for(t=0;t<4;t++)        //setting target output to 1 if t matches the poseid
            {
                if(t==poseid)
                    target[t]=1;
                else
                    target[t]=0;

            }
            pgm=readPGM(file,pgm);
            

            for(u=0;u<no_hidden_unit;u++)
                h[u].suminput=0;            //Initializing suminput of hidden units to 0
            for(t=0;t<4;t++)
                o[t].suminput=0;            //Initializing suminput of output units to 0
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
                           h[z].suminput=h[z].suminput+(in[t][j].weight[z]*in[t][j].value);     //Calculating suminput of hidden units
                        }            
                }
                h[z].out=squash(h[z].suminput);   //Calculating output using Squashing function
            }
           for(t=0;t<4;t++)
            {
            for(z=0;z<no_hidden_unit;z++)
                {
                    o[t].suminput=o[t].suminput+(h[z].out*h[z].weight[t]);      //Calculating suminput of output units
                }
            o[t].out=squash(o[t].suminput);         //Calculating output using Squashing function
            }
            for(t=0;t<4;t++)
                delout[t]=(o[t].out)*(1-o[t].out)*(target[t]-o[t].out);     //Calculating error for each output unit     
            int delh[no_hidden_unit]={0};  
            for(z=0;z<no_hidden_unit;z++)
        {
            int sumofweight=0;
            for(t=0;t<4;t++)
            {
                sumofweight+=h[z].weight[t]*delout[t];      //Calculating error for each hidden unit
            }
            delh[z]=(h[z].out)*(1-h[z].out)*sumofweight;       
        }
            for(z=0;z<no_hidden_unit;z++)
            {
                 for(t=0;t<30;t++)
                {
                    for(j=0;j<32;j++)
                        { 
                           delw=n*delh[z]*in[t][j].value;       //calculating delw
                           delw=delw+(alpha*olddelw[t][j][z]);      //adding momentum
                           olddelw[t][j][z]=delw;       
                           in[t][j].weight[z]+=delw;        //Updating weights
                        }            
                }
            }	
            for(t=0;t<4;t++)
            {   
                for(z=0;z<no_hidden_unit;z++)
                {
                    delw=n*delout[t]*h[z].out;      //Calculating delw
                    delw=delw+(alpha*olddelwhidden[z][t]);      //adding momentum
                    olddelwhidden[z][t]=delw;       
                    h[z].weight[t]+=delw;       //Updating weights
                }
            } 
    		testinput++;      //Incrementing testinput
        }

        
    }
   //Training done

    //TESTING
    int testnum=0;                     //Number of test case
    //myfile.open("all_train.list");
    //myfile.open("all_test1.list");
    myfile.open("all_test2.list");      //Opening the file with test inputs      

    while (getline (myfile,inputline) )
    {
       
        testnum++;
        int u,t,j;
        for(u=0;u<no_hidden_unit;u++)       //Initializing suminput for each hidden unit to 0
            h[u].suminput=0;
        for(t=0;t<4;t++)
            o[t].suminput=0;            //Initializing suminput for each output unit to 0
        int k=0;
        while(inputline[k]!='\0')
        {
            
            file[k]=inputline[k];
            k++;
        }

        file[k]='\0';
        b=0;
            for(a=0;a<k;a++)
            {
                if(b==2)
                {
                    pose=file[a];           //Storing the first character of the pose which appears after the second underscore
                    break;
                }
                if(file[a]=='_')
                    b++;

            }
            if(pose=='l')               //Setting the poseid by comparing the pose with the first character
            {
                poseid=0;
                
            }
            else if(pose=='r')
            {
                poseid=1;
                
            }
            else if(pose=='s')
            {
                poseid=2;
                
            }
            else if(pose=='u')
            {
                poseid=3;
                
            }
            pgm=readPGM(file,pgm);
            
            
            for(t=0;t<30;t++)
            {
                for(j=0;j<32;j++)
                { 
                    in[t][j].value=pgm->matrix[t][j];       //Assigning input values
                    in[t][j].value/=pgm->max_gray;      //Normalizing input values
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
                h[z].out=squash(h[z].suminput);   //Calculating output using Squashing function
           
            }
            for(t=0;t<4;t++)
            {
                for(z=0;z<no_hidden_unit;z++)
                {
                    o[t].suminput=o[t].suminput+(h[z].out*h[z].weight[t]);      //Calculating suminput for each output unit
                }
                o[t].out=squash(o[t].suminput);             //Calculating output using Squashing function
            }
         
 			int predposeid=0;           //Stores the predicted poseid
            double max=o[0].out;        //max stores the maximum value among the outputs
                
            for(t=1;t<4;t++)
            {
                if(o[t].out>max)
                {
                    max=o[t].out;
                    predposeid=t;       //Setting the predposeid to t if it has the maximum output value
                }
            }
        switch(predposeid)
        {
        case 0:cout<<"Test case #"<<testnum<<":Max value:"<<max<<" Pose: Left"<<endl;
                   break;
        case 1:cout<<"Test case #"<<testnum<<":Max value:"<<max<<" Pose: Right"<<endl;
                   break;
        case 2:cout<<"Test case #"<<testnum<<":Max value:"<<max<<" Pose: Straight"<<endl;
                   break;
        case 3:cout<<"Test case #"<<testnum<<":Max value:"<<max<<" Pose: Up"<<endl;
                   break;
        }

        
        if(predposeid==poseid)
        	accuracy++;            // Accuracy of the pose recognizer
        
    }
    myfile.close();
    accuracy/=testnum;
    accuracy*=100;
   cout<<"Accuracy of the test file:"<<accuracy<<endl;
    
    
}




