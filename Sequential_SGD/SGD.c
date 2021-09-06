#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include<string.h>

/*
// Parses the data in the buffer 
// @params buff[] buffer data
// @params arr_value[] the array that contains the output parsed data
// @return double matrix of CSV
*/
void getData(char buff[], double arr_value[])
{
   char* rest = buff;

	int counter = 0;
	char* token; 
	double f;
	
   while((token = strtok_r(rest, ",", &rest))) 
   {

   	f = (double)atof(token);
   	arr_value[counter] = f;
 	counter++;

   }
}

int get_feature_count(char buff[]){
  char *token = strtok(buff,",");
  int counter = 0;
 
   while( token != NULL ) 
   {
 	counter++; 
 	token = strtok(NULL,",");
   }

   return counter;	
}


/*
// Reads a CSV file and converts it into a C array
// @params filename the file that contains the data
// @params output_matrix the matrix that will contain the data of the CSV file
// @return double matrix of CSV
*/

void read_x_y(char* x_filename, char* y_filename, int n, int m, double x_matrix[n][m], double y_matrix[n])
{
    char buff[1024];


	FILE *fp_x = fopen(x_filename, "rb");
    double arr_value_x[m];

	FILE *fp_y = fopen(y_filename, "rb");
    double arr_value_y[n];

	//	
	// Dumping X into matrix
	//

	for(int i = 0; fgets(buff, 1024, (FILE*)fp_x) != NULL; i++)
	{

	 getData(buff, arr_value_x);
	 for(int j = 0; j<m; j++){
	 	x_matrix[i][j] = arr_value_x[j];
	 }
	 
	}


	//	
	// Dumping y into matrix
	//
	
	for(int i = 0; fgets(buff, 1024, (FILE*)fp_y) != NULL; i++)
	{

	 getData(buff, arr_value_y);
	 for(int j = 0; j<n ; j++){
	 	y_matrix[j] = arr_value_y[j];
	 }
	
	}

	fclose(fp_x);
	fclose(fp_y);

}

int get_feature_size(char* filename){
	
	FILE *fp = fopen(filename, "rb");

    char buff[1024];
	int feature_iter = 0;	

	while(fgets(buff, 1024, (FILE*)fp) != NULL)
	{

	 if(feature_iter == 0){

	 	char temp[1024];

	 	strcpy(temp, buff);

	 	feature_iter = get_feature_count(temp);
	 }
	}
return feature_iter;
}

int get_sample_size(char* filename)
{
	FILE *fp = fopen(filename, "rb");

    int count=0;
    char buff[1024];	

	while(fgets(buff, 1024, (FILE*)fp) != NULL){
	 count++;
	}

	return count;
}


double sigmoid_value(double x){
	if (x < 0){
	    double a = exp(x);
	    return a / (1.0 + a);
	}
	else 
    	return 1.0 / (1.0 + exp(-x));
}

void sigmoid_matrix(int n, int m, double inp_data[n][m]){
	for(int i=0;i<n;i++){
		for(int j=0;j<m;j++){
				inp_data[i][j] = sigmoid_value(inp_data[i][j]);
		}
	}
}

double cost_function(int n, int m, double train_x[n][m], double y[n], double weights[m]){
	double h_x = 0;
	double eps = 0.001;
	double cost_total = 0.00;
	double square_weights = 0.00; 
	double reg = 0.00;

	for(int i = 0; i < n; i++)
	{
		h_x = 0;
		for(int j = 0; j < m; j++)
		{
			h_x += weights[j] * train_x[i][j];
		}

		h_x = sigmoid_value(h_x);

		if(h_x < eps)
		{
			h_x += eps;
		}

		for(int j = 0; j < m; j++)
		{
			square_weights += pow(weights[j], 2);
		}
		//reg = (0.1/(2 * n)) * square_weights;

		if (y[i] == 1.000)
		{
			cost_total -= abs(log2(h_x));
		}
		else if(y[i] == 0.000)
		{
			cost_total -= abs(log2(1 - h_x)) ;
		}
		// if(abs(log2(1 - h_x)) != 0.00)
		// 	printf("1 - H_X:%d\n", abs(log2(1 - h_x)));
		// if(abs(log2(h_x)) != 0.00)
		// 	printf("H_X:%d\n", abs(log2(h_x)));


	}
	return /*reg + */(cost_total)/n;
}


void print_matrix(int n, int m, double matrix[n][m]){
	for(int i = 0; i < n ; i++){
		for(int j = 0; j < m ; j++){
			printf("%lf ", matrix[i][j]);
		}
		printf("\n");
	}
}

void print_vector(int n, double vector[n]){
	for(int i = 0; i < n ; i++){
		printf("%lf ", vector[i]);
	}
	printf("\n");
}

void random_weight_initializer(int m, double weights[m]){
	for(int i = 0; i < m ; i++){
			weights[i] = 1.00;
	}
}

void constant_weight_initializer(int m, double weights[m], int value){
	for(int i = 0; i < m ; i++){
			weights[i] = (double) value;
	}
}

void pre_process_dataset(int n, double vector[n]){
	for(int i = 0; i < n ; i++){
			if(vector[i] == -1.00){
				vector[i] = 0;
		}
	}
}

void array_copy(int n, double original[n], double copied[n]){
	for(int i = 0; i < n; i++) {
	      copied[i] = original[i];
	}
}

void quantize(int m, double dw[m]){
	for(int i = 0;i<m;i++)
	{
		if(dw[i] > 0.00)
		{
			dw[i] = 1.000;
		}
		else if(dw[i] < 0.00)
		{
			dw[i] = -1.000;
		} 
		else
		{
			dw[i] = 0.00;
		}
	}
}


void full_gradient_matrix(int n, int m, double weights[m], double train_x[n][m], double train_y[n], double dw[m]){
	double sigmoid_h[n];
	double h_x = 0.00;
	for(int j = 0; j < n; j++){
		h_x = 0;
		for(int k = 0; k < m; k++){
			h_x += weights[k] * train_x[j][k];
		}
		sigmoid_h[j] = sigmoid_value(h_x);
	}

	double dz[n];
	for(int j = 0; j < n ; j++){
		dz[j] = sigmoid_h[j] - train_y[j];
	}

	for(int j = 0; j < m; j++){
		dw[j] = 0;
		for(int k = 0; k < n; k++){
			dw[j] += train_x[k][j] * dz[k];
		}
		//dw[j] /= n ;//+ (0.1 * weights[j])/n ;
	}
}


void full_gradient_vector(int n, int m, int idx, double weights[m], double train_x[n][m], double train_y[n], double dw[m]){
		double sigmoid_h;
		double h_x = 0.00;
		for(int i = 0; i < m; i++){
			//printf("w:%lf \n train:%lf \n h:%lf  \n\n", weights[i], train_x[idx][i] ,weights[i] * train_x[idx][i]);
			h_x += weights[i] * train_x[idx][i];
		}
		//printf("hypo: %lf \n", h_x);
		sigmoid_h = sigmoid_value(h_x);
		double dz;
		//printf("sigmoid: %lf \n", sigmoid_h);
		dz = sigmoid_h - train_y[idx];
		//printf("Cost: %lf \n", dz);
		for(int i = 0; i < m; i++){
			dw[i] = (train_x[idx][i] * dz);
		}
}



void predict(int n, int m, double matrix_x[n][m], double weights[m], double prediction[n]){
	int h_x = 0;
	for(int j = 0; j < n; j++){
			h_x = 0;
			for(int k = 0; k < m; k++){
				h_x += weights[k] * matrix_x[j][k];
			}
			prediction[j] = round(sigmoid_value(h_x));
		}
}

double accuracy(int n, double train_y[n], double prediction[n]){
	double err = 0;
	for(int i = 0; i < n; i++){
		err += abs(train_y[i] - prediction[i]);
	}
	//printf("ERROR: %lf\n", err);
	return ((n - err)/n) * 100;
}

double *stochastic_gradient_descent(int max_epoch, int iterate_count, double learning_rate, int n, int m, double train_x[n][m], double train_y[n], double weights[m]){
	double cost;
	double dw[m];
	int idx;

	//double weights[m];
	random_weight_initializer(m, weights);
	for(int i = 0; i < max_epoch; i++){
		for(int j = 0; j < iterate_count; j++){

			idx = rand() % n;
			full_gradient_vector(n, m, idx, weights, train_x, train_y, dw);
			
			for(int k = 0; k < m; k++){
				weights[k] -= learning_rate * dw[k];
			}

		}
		cost = cost_function(n, m, train_x, train_y, weights);
		//printf("Cost: %lf\n", cost);	
	}

	double prediction[n];
	predict(n, m, train_x, weights, prediction);

	double acc = accuracy(n, train_y, prediction);
	//printf("Accuracy is: %lf\n", acc);
	//print_vector(m, weights);
	return weights;
}


double *stochastic_variance_reduced_gradient(int max_epoch, int iterate_count, double learning_rate, int n, int m, double train_x[n][m], double train_y[n], double weights[m]){
	double cost = 0;	
	int idx;

	double w[m];
	double w_s[m];
	double dw_t_1[m];
	double dw_s[m];
	double nu[m];

	random_weight_initializer(m, w);


	for(int i = 0; i < max_epoch; i++){

		full_gradient_matrix(n, m, w_s, train_x, train_y, nu);
	
		for(int j = 0; j < iterate_count; j++){
			idx = rand() % n;
			full_gradient_vector(n, m, idx, w, train_x, train_y, dw_t_1);
			full_gradient_vector(n, m, idx, w_s, train_x, train_y, dw_s);
			for(int k = 0;k<m;k++){
				w[k] = w[k] - learning_rate * (dw_t_1[k] - dw_s[k] +nu[k]);
			}
		}
		array_copy(m, w, w_s);
		cost = cost_function(n, m, train_x, train_y, w);
		//printf("%lf\n", cost);	
	}
	
	array_copy(m, w, weights);
	//print_vector(m, weights);
	return weights;
}

double *SARAH(int max_epoch, int iterate_count, double learning_rate, int n, int m, double train_x[n][m], double train_y[n], double weights[m]){
 	double cost = 0;
	
	int idx;

	double w_1[m];
	double w_2[m];
	double dw_1[m];
	double dw_2[m];
	double v[m];
	//double weights[m];

	random_weight_initializer(m, w_1);

	for(int i = 0; i < max_epoch; i++){

		full_gradient_matrix(n, m, w_1, train_x, train_y, v);

		for(int k = 0;k<m;k++){
			w_2[k] = w_1[k] - learning_rate * v[k];
		}
	
		for(int j = 0; j < iterate_count; j++){
			idx = rand() % n;

			full_gradient_vector(n, m, idx, w_1, train_x, train_y, dw_1);
			full_gradient_vector(n, m, idx, w_2, train_x, train_y, dw_2);
			
			for(int k = 0;k < m;k++){
				v[k] = dw_2[k] - dw_1[k] + v[k];
			}

			array_copy(m, w_2, w_1);

			for(int k = 0;k<m;k++){
				w_2[k] -=  learning_rate * v[k];
			}
		}
		array_copy(m, w_2, w_1);
		cost = cost_function(n, m, train_x, train_y, w_2);
		//printf("%lf\n", cost);	
	}
	
	array_copy(m, w_2, weights);
	//print_vector(m, weights);
	return weights;
}

double *signSGD(int max_epoch, int iterate_count, double learning_rate, double beta, int n, int m, double train_x[n][m], double train_y[n], double weights[m]){
	double cost = 0;
	double dw[m], dw_cpy[m];
	int idx;
	//double weights[m];
	random_weight_initializer(m, weights);
	constant_weight_initializer(m, dw, 0);


	for(int i = 0; i < max_epoch; i++){

		for(int j = 0; j < iterate_count; j++){

			idx = rand() % n;

			array_copy(m, dw, dw_cpy);

			full_gradient_vector(n, m, idx, weights, train_x, train_y, dw);

			for(int i = 0;i<m;i++){
				dw[i] =  dw_cpy[i] * beta + (1 - beta) * dw[i]; 
			}

			quantize(m, dw);
			for(int k = 0; k < m; k++){
				weights[k] -= learning_rate * dw[k];
			}

		}
		cost = cost_function(n, m, train_x, train_y, weights);
		//printf("%lf\n", cost);	
	}
	//print_vector(m, weights);
	return weights;
}


double *ADAM(int max_epoch, int iterate_count, double learning_rate, double beta_1, double beta_2, double epsilon, int n, int m, double train_x[n][m], double train_y[n], double weights[m]){
	double cost = 0;
	double dw[m];
	double momentum[m], v[m];
	double momentum_t[m], v_t[m];
	//double weights[m];

	constant_weight_initializer(m, momentum, 0);
	constant_weight_initializer(m, v, 0);

	int idx;

	constant_weight_initializer(m, dw, 0);

	for(int i = 0; i < max_epoch; i++){

		for(int j = 1; j < iterate_count; j++){

			idx = rand() % n;

			full_gradient_vector(n, m, idx, weights, train_x, train_y, dw);

			
			for(int k = 0;k<m;k++){
				momentum[k] =  momentum[k] * beta_1 + (1 - beta_1) * dw[k]; 
			}

			for(int k = 0;k<m;k++){
				v[k] =  v[k] * beta_2 + (1 - beta_2) * (dw[k] * dw[k]); 
			}

			for(int k = 0;k<m;k++){
				momentum_t[k] =  momentum[k]/(1 - pow(beta_1, j));
			}

			for(int k = 0;k<m;k++){
				v_t[k] = v[k]/(1 - pow(beta_2, j));
			}


			for(int k = 0; k < m; k++){
				weights[k] -= learning_rate * momentum_t[k]/(pow(v_t[k], 0.5) + epsilon);
			}

		}
		cost = cost_function(n, m, train_x, train_y, weights);
		//printf("%lf\n", cost);	
	}
	return weights;
}

