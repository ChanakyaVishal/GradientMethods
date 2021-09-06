/***************/
/* C LIBRARIES */
/***************/

#include <stdio.h>
#include <float.h>

/********************/
/* CUSTOM LIBRARIES */
/*******************/

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "Utils.cuh"
#include "GradientDescentGPU.cuh"

/*****************/
/* C++ LIBRARIES */
/*****************/

#include <bits/stdc++.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <iterator>
#include <string>
#include <algorithm>
#include <boost/algorithm/string.hpp>


#define forall(i, a, b) for(int i = a;i<b;i++)
#define SZ(X) ((int)(X.size())) 
#define watch(x) cout << (#x) << " is " << (x) << endl
typedef long long ll;

/***************/
/* FILE READER */
/***************/

class CSVReader
{
    std::string fileName;
    std::string delimeter;
    public:
        CSVReader(std::string filename, std::string delm = ",") :
        fileName(filename), delimeter(delm)
        { }
        // Function to fetch data from a CSV File
        std::vector<std::vector<std::string> > getData();
        };
        /*
        * Parses through csv file line by line and returns the data
        * in vector of vector of strings.
        */
        std::vector<std::vector<std::string> > CSVReader::getData()
        {
        std::ifstream file(fileName);
        std::vector<std::vector<std::string> > dataList;
        std::string line = "";
        // Iterate through each line and split the content using delimeter
        while (getline(file, line))
        {
            std::vector<std::string> vec;
            boost::algorithm::split(vec, line, boost::is_any_of(delimeter));
            dataList.push_back(vec);
        }
        // Close the File
        file.close();
        return dataList;
}


std::vector<std::vector<float> > readCSV(std::string filename){
    // Creating an object of CSVWriter
    CSVReader reader(filename);
    // Get the data from CSV File
    std::vector<std::vector<std::string> > data_list = reader.getData();
    std::vector<std::vector<float> > data_list_double(SZ(data_list),std::vector<float>(SZ(data_list[0]))); 
    for(int i = 0;i<SZ(data_list);i++){
        for(int j = 0;j<SZ(data_list[0]);j++){
            data_list_double[i][j] = std::stod(data_list[i][j]);
        }
    }
    return data_list_double;
} 

/*****************/
/* SIGMOID VALUE */
/*****************/

// --- Calculates and returns the signmoid value of a number
float sigmoid_value_h(float x){
    return 1/(1 + exp(-1 * x));
}

/***********************/
/* PREDICTION FUNCTION */
/***********************/

void predict(int n, int m, std::vector<std::vector<float> > matrix_x, float* weights, float* prediction){
	int h_x = 0;
	for(int j = 0; j < n; j++){
			h_x = 0;
			for(int k = 0; k < m; k++){
				h_x += weights[k] * matrix_x[j][k];
			}
			prediction[j] = round(sigmoid_value_h(h_x));
            if(j%10 == 0)
                printf("Prediction at %d is %f: \n", j, prediction[j]);
		}
}

/***********************/
/* ACCURACY FUNCTION */
/***********************/

float accuracy(int n, std::vector<float> train_y, float* prediction){
	float err = 0;
	for(int i = 0; i < n; i++){
		err += abs(train_y[i] - prediction[i]);
	}
	printf("ERROR: %lf\n", err);
	return ((n - err)/n) * 100;
}

/********/
/* MAIN */
/********/

int main()
{
    /**********************/
    /* INPUT DATA LOADING */
    /**********************/
    std::vector<std::vector<float> > train_x;
    train_x = readCSV("test_data/FMNIST_train_data_x.csv");

    std::vector<std::vector<float> > train_y_temp;
    train_y_temp = readCSV("test_data/FMNIST_train_data_y.csv");
    std::vector<float> train_y = train_y_temp[0];


    /********************/
    /* INPUT PARAMETERS */
    /********************/

    // --- Input data host allocation
    const int n = SZ(train_x);
    const int m = SZ(train_x[0]);

    float *h_x = (float*)malloc(n * m * sizeof(float));

    for(int i = 0;i<n;i++)
        for(int j = 0;j<m;j++)
            h_x[i*m + j] = train_x[i][j];
   
   // --- Output label host allocation
   float *h_y = (float*)malloc(n * sizeof(float));
   for(int i = 0;i<n;i++){
	   h_y[i] = train_y[i];        
    }
    // --- Type of Optimizer to use
    int type;
    printf("Enter Type: \n");
    scanf("%d", &type);
   
    // --- Beta for momentum
    const int beta = 0.8;

    // --- Batch Size
    int no_of_batches;
    printf("Enter Number of Batches: \n");
    scanf("%d", &no_of_batches);

    // --- Termination tolerance
    const float tol = 1.e-6;

    // --- Maximum number of allowed iterations
    int temp;
    printf("Enter Maximum number of iterations: \n");
    scanf("%d", &temp);   
    const int maxiter = temp;

    // --- Learning rate
    float temp2;
    printf("Enter Learning rate: \n");
    scanf("%f", &temp2);   
    const float alpha = temp2;

    // --- Derivative step
    const float h = 0.1f;

    // --- Minimum allowed perturbations
    const float dxmin = 1e-5;

    // --- Intialize weights 
    float *h_x0 = (float*)malloc(m * sizeof(float));
    for (int i=0; i<m; i++)
        h_x0[i] = 1.0;


    /*********************/
    /* OUTPUT PARAMETERS */
    /*********************/

    // --- Optimal point
    float *h_xopt = (float*)malloc(m * sizeof(float));
    for (int i=0; i<m; i++) h_xopt[i] = 0.f;

    // --- Optimal functional
    float fopt = 0.f;

    // --- Number of performed iterations
    int niter = 1;

    // --- Gradient norm at optimal point
    float gnorm = 0.f;

    // --- Distance between last and penultimate solutions found
    float dx = 0.f;

    /***************************/
    /* OPTIMIZATION - GPU CASE */
    /***************************/

    // --- Input data: x
    float *d_inp;    gpuErrchk(cudaMalloc((void**)&d_inp,    n *  m * sizeof(float)));
    gpuErrchk(cudaMemcpy(d_inp, h_x, n * m * sizeof(float), cudaMemcpyHostToDevice));
    // --- Data labels: y
    float *d_lab;    gpuErrchk(cudaMalloc((void**)&d_lab,    n * sizeof(float)));
    gpuErrchk(cudaMemcpy(d_lab, h_y, n * sizeof(float), cudaMemcpyHostToDevice));
    // --- Initial Weights: h_x0
    float *d_x0;    gpuErrchk(cudaMalloc((void**)&d_x0,    m * sizeof(float)));
    gpuErrchk(cudaMemcpy(d_x0, h_x0, m * sizeof(float), cudaMemcpyHostToDevice));
    // --- Optimal point
    float *d_xopt;  gpuErrchk(cudaMalloc((void**)&d_xopt,   m * sizeof(float)));

    GradientDescentGPU(d_inp, d_lab, d_x0, tol, maxiter, alpha, h, dxmin, n, m, d_xopt, &fopt, &niter, &gnorm, &dx, type, beta, no_of_batches);

    printf("Solution found - GPU case:\n");
    printf("fopt = %f; niter = %i; gnorm = %f; dx = %f\n", fopt, niter, gnorm, dx);
    printf("\n\n");


    gpuErrchk(cudaMemcpy(h_xopt, d_xopt, m * sizeof(float), cudaMemcpyDeviceToHost));
    printf("Found minimum - GPU case:\n");
    for (int i=0; i<m; i++) printf("i = %i; h_xopt = %f\n", i, h_xopt[i]);
    printf("\n\n");

    // -- Metrics calculation
    float *predictions = (float*)malloc(n * sizeof(float));
    predict(n, m, train_x, h_xopt, predictions);
    float acc = accuracy(n, train_y, predictions);   
    printf("Accuracy is: %f", acc);
    return 0;
}
