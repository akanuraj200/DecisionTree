/*
 * @file sunglass_with_ann.cpp
 * @authors Himanshu Bagga, Stuti Sabharwal 
 * 
 * Neural Network for sunglass recognition. 
 * task: When given an image as input, indicate whether the face
 * in the image is wearing sunglasses or not.
 *
 */


#include<bits/stdc++.h>

#define no_hidden_unit 20     //Number of units in the hidden layer
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
    double weight;
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
    double olddelw[30][32][no_hidden_unit]={0};                  //delta w for previous iteration
    double olddelwhidden[no_hidden_unit]={0};           //delta w of hidden units for previous iteration
    input in[30][32];                             //960 objects for input type, one for each pixel
    hidden h[no_hidden_unit];                      //Hidden units declaration
    output o;                               //Output unit declaration
    double target,delw,n=0.3,delout,accuracy=0;        //n is the learning rate
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
      
        h[u].weight=(rand()%100) + (-50);       //random weights in the range of (-0.5,0.5)
        h[u].weight/=100;
    }
    int epoch=100,train;        //number of iterations
    string line[1000],inputline;
    int no_of_inputs=0;
    myfile.open("straightrnd_train.list"); //open the file with training list
    
 
    while(getline(myfile,inputline))
    {
        line[no_of_inputs]=inputline;   
        no_of_inputs++;         //incrementing number of inputs
        
    }
   
    myfile.close();     //close the file

 for(train=0;train<epoch;train++)   
 {      
        int testinput=0;    //initializing testinput
        cout << train << endl ; 
        while(testinput<no_of_inputs)
       {
        for(u=0;u<no_hidden_unit;u++)   //Initializing suminput for each Hidden unit to 0
            h[u].suminput=0;
        o.suminput=0;       //Initializing suminput for the output unit to 0
        int k=0;
        while(line[testinput][k]!='\0')
        {
            
            file[k]=line[testinput][k];
            k++;
        }

        file[k]='\0';
        //cout<<file<<endl;
        if(line[testinput][k-7]=='s') //Checking if the the 7th letter from the end of the input is the letter 's'
            target=1;               //If it is 's' then the person is wearing Sunglasses and target output is 1
        else
            target=0;
        pgm=readPGM(file,pgm);
        for(t=0;t<30;t++)
        {
            for(j=0;j<32;j++)
            { 
                in[t][j].value=pgm->matrix[t][j];   //Assigning input values 
                in[t][j].value/=pgm->max_gray;      //Normalizing the input values
            }            
        } 
        for(z=0;z<no_hidden_unit;z++)
        {
            
             for(t=0;t<30;t++)
            {
                
                for(j=0;j<32;j++)
                    { 
                       h[z].suminput=h[z].suminput+(in[t][j].weight[z]*in[t][j].value); //Calculating suminput of each hidden unit
                    }            
            }
            h[z].out=squash(h[z].suminput);    //Use the squashing function to calculate output of hidden units
        }
        for(z=0;z<no_hidden_unit;z++)
        {   
            o.suminput=o.suminput+(h[z].out*h[z].weight);   //Calculating suminput of the output unit
        }
        o.out=squash(o.suminput);       //Use the squashing function to calculate the output of output unit
        delout=(o.out)*(1-o.out)*(target-o.out);    //Calculating error for the output unit
        int delh[no_hidden_unit]={0};
        for(z=0;z<no_hidden_unit;z++)
        {
            delh[z]=(h[z].out)*(1-h[z].out)*h[z].weight*delout; //Calculating error for each hidden unit
        }
        for(z=0;z<no_hidden_unit;z++)
        {
             for(t=0;t<30;t++)
            {
                for(j=0;j<32;j++)
                    { 
                       delw=n*delh[z]*in[t][j].value;   //Calulating delw
                       delw=delw+(alpha*olddelw[t][j][z]);  //adding the momentum
                       olddelw[t][j][z]=delw;
                       in[t][j].weight[z]+=delw;    //updating the input weights
                    }            
            }
        }


        for(z=0;z<no_hidden_unit;z++)
        {

            delw=n*delout*h[z].out; //Calculating delw
            delw=delw+(alpha*olddelwhidden[z]); //adding Momentum
            olddelwhidden[z]=delw;
            h[z].weight+=delw;  //Updating the hidden unit weights
        }
    testinput++;    //Incrementing test input
    }
    
   } 

  //TESTING
   int testnum=0;
    myfile.open("straightrnd_train.list");
    //myfile.open("straightrnd_test1.list");
    //myfile.open("straightrnd_test2.list");  //Open the file with the list of test inputs
    while ( getline (myfile,inputline) )
    {
    	testnum++;
        int prediction;     //prediction stores the predicted value
        int trueout;        //trueout stores the target output

        for(u=0;u<no_hidden_unit;u++)         //Initializing suminput to 0 for each hidden unit
            h[u].suminput=0;            
        o.suminput=0;               //Initializing suminput of the output unit to 0 
        int k=0;
        while(inputline[k]!='\0')
        {
            
            file[k]=inputline[k];
            k++;
        }

        file[k]='\0';
        if(inputline[k-7]=='s')     //Checking if the 7th letter from the end of the test input is 's'
            trueout=1;          //trueout is 1 if the above condition is true
        else
            trueout=0;
        pgm=readPGM(file,pgm);
        int t,j;
        for(t=0;t<30;t++)
        {
            for(j=0;j<32;j++)
               { 
                in[t][j].value=pgm->matrix[t][j];       //Assigning input values
                in[t][j].value/=pgm->max_gray;                //Normalizing the input values
                }            
        } 
        int z;
        for(z=0;z<no_hidden_unit;z++)
        {
             for(t=0;t<30;t++)
            {
                for(j=0;j<32;j++)
                    { 
                       h[z].suminput+=(in[t][j].weight[z]*in[t][j].value);      //Calculating suminput for each hidden unit
                    }            
                

            }
            h[z].out=squash(h[z].suminput);     //Calculating output of hidden unit using Squashing function
        }
        for(z=0;z<no_hidden_unit;z++)
        {
            o.suminput+=(h[z].out*h[z].weight);     //Calculating suminput for the Output unit
        }
        o.out=squash(o.suminput);       //Output using Squashing function
       // cout<<o.out<<endl;
        if(o.out<0.5)           //Threshold is 0.5
            prediction=0;       //prediction is 0 if output is less than threshold
        else
            prediction=1;       //prediction is 1 if output is greater than threshold
        if(prediction==trueout)
            accuracy++;                      //Calculating accuracy of predictions
        if(prediction==1)              
            cout<<"For test case #"<<testnum<<":Sunglasses"<<endl;
        else
            cout<<"For test case #"<<testnum<<":Open"<<endl;

   
    }
    accuracy/=testnum;
    accuracy*=100;
    cout<<"Accuracy:"<<accuracy<<"%"<<endl;
    myfile.close(); //close the file
}
