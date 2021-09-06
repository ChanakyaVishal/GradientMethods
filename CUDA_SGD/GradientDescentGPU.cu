#include <thrust/device_ptr.h>
#include <thrust/inner_product.h>
#include "GradientDescentGPU.cuh"
#include "Utils.cuh"
#include <time.h> 
#include <curand.h>
#include <curand_kernel.h>


#define BLOCK_SIZE 256
#define FEATURE_COUNT 784

/*****************/
/* SIGMOID VALUE */
/*****************/

// --- Calculates and returns the signmoid value of a number
__host__ __device__ float sigmoid_value(float x){
    return 1/(1 + exp(-1 * x));
}


/*******************/
/* RANDOM FUNCTION */
/*******************/
__device__ int random(int tid, const int n) {
  /* CUDA's random number library uses curandState_t to keep track of the seed value
     we will store a random state for every thread  */
  curandState_t state;

  /* we have to initialize the state */
  curand_init((unsigned long long)clock(), /* the seed controls the sequence of random values that are produced */
              0, /* the sequence number is only important with multiple cores */
              tid, /* the offset is how much extra we advance in the sequence for each call, can be 0 */
              &state);

  /* curand works like rand - except that it takes a state as a parameter */
  return curand(&state) % n;
}


/*******************************************SGD********************************************/

__device__ void CostFunctionGradientSGDGPU(float * __restrict x, const float * __restrict y, 
float * __restrict d_x, float * __restrict d_g, 
const float h, const int tid, const int n, const int m) {

    float hyp = 0.00;
    for(int i = 0; i < m; i++){
        hyp += d_x[i] * x[tid*m + i];
    }

    float sigmoid_h = sigmoid_value(hyp);
       
    for(int i = 0; i < m; i++){
        float temp = (x[tid*m + i] * (sigmoid_h - y[tid]));
        d_g[i] = temp;
    }
}

__global__ void StepSGDGPU(float * __restrict x, const float * __restrict y, float * __restrict d_x, 
    float * __restrict d_xnew, float * __restrict d_xdiff,
    const float alpha, const float h, const int n, const int m) 
{
    
    const int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < n) 
    {
        //float *d_g = (float*)malloc(m * sizeof(float)); 
        //memset(d_g, 0, m * sizeof(float));
        float d_g[FEATURE_COUNT] = {0};

        //memset(d_g, 0,  m * sizeof(float));
	    int idx = random(tid, n);

        // --- Calculate gradient
        CostFunctionGradientSGDGPU(x, y, d_x, d_g, h, idx, n, m);

        for(int i=0; i<m; i++){
            // --- Take step
            d_xnew[i] -= alpha * d_g[i];

            // --- Update termination metrics
            d_xdiff[i] = d_xnew[i] - d_x[i];

            // --- Update current solution
            d_x[i] = d_xnew[i];
        }
    }
}

/*******************************************SGD********************************************/


/*******************************************signSGD********************************************/

__device__ void CostFunctionGradientsignSGDGPU(float * __restrict x, const float * __restrict y, 
float * __restrict d_x, float * __restrict d_g, 
const float h, const int tid, const int n, const int m) 
{

    if(tid < n)
    {
        float hyp = 0.00;
        for(int i = 0; i < m; i++)
        {
            hyp += d_x[i] * x[tid*m + i];
        }

        float sigmoid_h = sigmoid_value(hyp);
           
        for(int i = 0; i < m; i++)
        {
            float temp = (x[tid*m + i] * (sigmoid_h - y[tid]));
            d_g[i] += temp;
        }
    }
}

__global__ void StepsignSGDGPU(float * __restrict x, const float * __restrict y, float * __restrict d_x, 
 const float alpha, const float h, const int n, const int m, const float beta,
 float * __restrict__ d_momentum, 
 float * __restrict__ d_device_server_v, 
 const int no_of_batches) 
{
    
    const int tid = threadIdx.x + blockIdx.x * blockDim.x;

    //float *d_g = (float*)malloc(m * sizeof(float)); 
    //memset(d_g, 0, m * sizeof(float));
    float d_g[FEATURE_COUNT] = {0};

    for(int i=0; i<((n/no_of_batches)/1.5); i++)
    {
        int idx = (random(tid+i, (n/no_of_batches)) + tid*(n/no_of_batches));
        //printf("(tid:%d, %d) ", tid, idx);
        CostFunctionGradientsignSGDGPU(x, y, d_x, d_g, h, idx/*tid*(n/no_of_batches) + i*/, n, m);
    }

    for(int i=0; i<m; i++)
    {
        d_g[i] /= (n/no_of_batches);
    }

    for(int i=0; i<m; i++)
    {
        d_momentum[tid*m + i] = beta * d_momentum[tid*m + i] + (1 - beta) * d_g[i];
        d_device_server_v[tid*m + i] = ((d_momentum[tid*m + i])>0?1:-1); 
    } 
}

/*******************************************signSGD********************************************/


/*******************************************ADAM********************************************/

__global__ void StepsignADAMGPU(float * __restrict x, const float * __restrict y, float * __restrict d_x, 
    const float alpha, const float h, const int n, const int m,
    const float beta_1, const float beta_2, const int epsilon,
    float * __restrict__ d_momentum, 
    float * __restrict__ d_v, 
    float * __restrict__ d_device_server_v, 
    const int no_of_batches,
    int d_niter) {
    
    const int tid = threadIdx.x + blockIdx.x * blockDim.x;

    if (tid < n) 
    {

        //float *d_g = (float*)malloc(m * sizeof(float)); 
        //memset(d_g, 0, m * sizeof(float));
        float d_g[FEATURE_COUNT] = {0};


        for(int i=0; i<((n/no_of_batches)); i++)
        {
            int idx = (random(tid+i, (n/no_of_batches)) + tid*(n/no_of_batches));
            CostFunctionGradientsignSGDGPU(x, y, d_x, d_g, h, /*tid*(n/no_of_batches) + i*/ idx, n, m);
        }

    	for(int k = 0; k<m; k++)
        {
    		d_momentum[tid*m + k] = d_momentum[tid*m + k] * beta_1 + (1 - beta_1) * d_g[k]; 
    	}

    	for(int k = 0; k<m; k++)
        {
    		d_v[tid*m + k] =  d_v[tid*m + k] * beta_2 + (1 - beta_2) * (d_g[k] * d_g[k]); 
    	}

    	for(int k = 0; k<m; k++)
        {
            d_g[k] += ((d_momentum[tid*m + k]/(1 - pow(beta_1, d_niter)))/(pow(d_v[tid*m + k]/(1 - pow(beta_2, d_niter)), 0.5)) + epsilon);
            d_g[k] /= (n/no_of_batches);
            d_device_server_v[tid*m + k] = ((d_g[k])>0?1:-1); 
        }	
    }
}




/*******************************************SVRG********************************************/


__global__ void StepsignSVRGGPU(float * __restrict x, const float * __restrict y, float * __restrict d_x, 
 const float alpha, const float h, const int n, const int m, float * __restrict d_w_s, float * __restrict d_nu,
 float * __restrict d_device_server_v, const int no_of_batches, float * __restrict__ d_momentum, float beta, int niter) 
{
    
    const int tid = threadIdx.x + blockIdx.x * blockDim.x;

    float d_g[FEATURE_COUNT] = {0};
    float d_g_s[FEATURE_COUNT] = {0};

    int idx = random(tid * niter, n);
    CostFunctionGradientsignSGDGPU(x, y, d_x, d_g, h, idx/*tid*(n/no_of_batches) + i*/, n, m);
    CostFunctionGradientsignSGDGPU(x, y, d_w_s, d_g_s, h, idx/*tid*(n/no_of_batches) + i*/, n, m);

    for(int i=0; i<m; i++)
    {
        //d_momentum[tid*m + i] = beta * d_momentum[tid*m + i] + (1 - beta) * (d_g[i] - d_g_s[i] + d_nu[i]);
        d_device_server_v[tid*m + i] = ((d_g[i] - d_g_s[i] + d_nu[i])/*>0?1:-1*/); 
    } 
}

/*******************************************SVRG********************************************/


__global__ void StepsignSVRGNewGPU(float * __restrict x, const float * __restrict y, float * __restrict d_x, 
 const float alpha, const float h, const int n, const int m, float * __restrict d_w_s, float * __restrict d_nu,
 float * __restrict d_device_server_v, const int no_of_batches, float * __restrict__ d_momentum, float beta, int niter) 
{
    
    const int tid = threadIdx.x + blockIdx.x * blockDim.x;

    float d_g[FEATURE_COUNT] = {0};
    float d_g_s[FEATURE_COUNT] = {0};

    int idx = random(tid * niter, n);
    CostFunctionGradientsignSGDGPU(x, y, d_x, d_g, h, idx/*tid*(n/no_of_batches) + i*/, n, m);
    CostFunctionGradientsignSGDGPU(x, y, d_w_s, d_g_s, h, idx/*tid*(n/no_of_batches) + i*/, n, m);

    for(int i=0; i<m; i++)
    {
        //d_momentum[tid*m + i] = beta * d_momentum[tid*m + i] + (1 - beta) * (d_g[i] - d_g_s[i] + d_nu[i]);
        d_device_server_v[tid*m + i] = ((d_g[i] - d_g_s[i] + d_nu[i])/*>0?1:-1*/); 
    } 
}

__global__ void SVRGFullGradientGPU(int n, int m, int no_of_batches, float * __restrict x, 
    const float * __restrict y, const float * __restrict d_x, float * __restrict d_full_gradient_batch_values){

        const int tid = threadIdx.x + blockIdx.x * blockDim.x;       
        
        float h_x = 0.00;
        int lower_limit = tid * (n/no_of_batches), upper_limit = tid * (n/no_of_batches) + (n/no_of_batches);
        for(int j = min(lower_limit, n); j < min(upper_limit,n); j++){
            h_x = 0;
            for(int k = 0; k < m; k++){
                h_x += d_x[k] * x[j*m + k];
            }
            for(int k=0; k<m; k++)
                d_full_gradient_batch_values[tid * m + k] += x[j*m + k] *  (sigmoid_value(h_x) - y[j]);
        }
}


/****************************************/
/* GRADIENT DESCENT FUNCTION - GPU CASE */
/****************************************/
// x0      - Starting point
// tol     - Termination tolerance
// maxiter - Maximum number of allowed iterations
// alpha   - Step size
// dxmin   - Minimum allowed perturbations

void GradientDescentGPU(float * __restrict__ x, const float * __restrict__ y, const float * __restrict__ d_x0, 
                        const float tol, const int maxiter, const float alpha,
                        const float h, const float dxmin, const int n, 
                        const int m, float * __restrict__ d_xopt, float *fopt, 
                        int *niter, float *gnorm, float *dx, 
                        int type, const float beta, int no_of_batches) {

    // --- Initialize gradient norm, optimization vector, iteration counter, perturbation    
    *niter = 1;
    *gnorm = FLT_MAX; 
    *dx = FLT_MAX;

    // #TODO: Change it to argument

    float lambda = 0.25;
    int svrg_m = 100;

    // Incase of Adam:
    float beta_1=0.9, beta_2=0.999, epsilon=1e-8;

    // #TODO: Change it to early stopping
    float *d_xnew;      
    gpuErrchk(cudaMalloc((void**)&d_xnew, m * sizeof(float)));  

    float *d_xdiff;     
    gpuErrchk(cudaMalloc((void**)&d_xdiff, m * sizeof(float)));
    thrust::device_ptr<float> dev_ptr_xdiff = thrust::device_pointer_cast(d_xdiff);

    float *d_x;         
    gpuErrchk(cudaMalloc((void**)&d_x, m * sizeof(float)));
    gpuErrchk(cudaMemcpy(d_x, d_x0, m * sizeof(float), cudaMemcpyDeviceToDevice));

    float *d_momentum; 
    gpuErrchk(cudaMalloc((void**)&d_momentum, no_of_batches * m * sizeof(float)));   
    gpuErrchk(cudaMemset(d_momentum, 0, no_of_batches * m * sizeof(float))); 

    float *d_v; 
    gpuErrchk(cudaMalloc((void**)&d_v, no_of_batches * m * sizeof(float)));   
    gpuErrchk(cudaMemset(d_v, 0, no_of_batches * m * sizeof(float)));  

    float *d_g;         
    gpuErrchk(cudaMalloc((void**)&d_g, m * sizeof(float))); 
    gpuErrchk(cudaMemset(d_g, 0, m*sizeof(float)));    
    thrust::device_ptr<float> dev_ptr_g     = thrust::device_pointer_cast(d_g);

    float *d_device_server_v;         
    gpuErrchk(cudaMalloc((void**)&d_device_server_v, no_of_batches * m * sizeof(float))); 
    gpuErrchk(cudaMemset(d_device_server_v, 0, no_of_batches * m * sizeof(float)));    

    float *d_nu;         
    gpuErrchk(cudaMalloc((void**)&d_nu,  m * sizeof(float))); 
    gpuErrchk(cudaMemset(d_nu, 0, m * sizeof(float)));   

    float *d_w_s;         
    gpuErrchk(cudaMalloc((void**)&d_w_s,  m * sizeof(float))); 
    gpuErrchk(cudaMemset(d_w_s, 0, m * sizeof(float)));    
    
    float *temp_d_x = (float*)malloc(m * sizeof(float)); 
    float *temp_d_v  = (float*)malloc(no_of_batches * m * sizeof(float));
    float *temp_d_g_sum  = (float*)malloc(m * sizeof(float));
    float *h_w  = (float*)malloc(m * sizeof(float));
    float *nu  = (float*)malloc(m * sizeof(float));
    float *ajax = (float*)malloc(n * m * sizeof(float));  
    float *achilles = (float*)malloc(n * sizeof(float));
    float *full_gradient_batch_values = (float*)malloc(m * no_of_batches * sizeof(float));       
    float *cur_batch_values = (float*)malloc(m * sizeof(float)); 
    gpuErrchk(cudaMemcpy(ajax, x, n * m * sizeof(float), cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(achilles, y, n * sizeof(float), cudaMemcpyDeviceToHost));

    for(int i=0; i<m; i++){
        h_w[i] = 0;
        temp_d_g_sum[i] = 0;
        nu[i] = 0;
        temp_d_x[i] = 0;
        for(int j=0; j<no_of_batches; j++){
            full_gradient_batch_values[j*m + i] = 0;
        }
    }

    float *d_full_gradient_batch_values;         
    gpuErrchk(cudaMalloc((void**)&d_full_gradient_batch_values,  m * no_of_batches * sizeof(float))); 
    gpuErrchk(cudaMemset(d_full_gradient_batch_values, 0, m * no_of_batches * sizeof(float)));    
    

    // --- Gradient Descent iterations
    clock_t t; 
    t = clock(); 

    while ((*niter <= maxiter) && (*dx >= dxmin)) {
        
        // --- Iteration step
        // SGD
        if(type == 1)
        {
            dim3 blocks(iDivUp(n, BLOCK_SIZE));
            dim3 threads(BLOCK_SIZE);
        	StepSGDGPU<<<blocks, threads>>>(x, y, d_x, d_xnew, d_xdiff, alpha, h, n, m);
        }
        // signSGD
    	else if(type == 2)
        {  
            // --- Setting block and threads count for kernel #TODO: Change to a block setting
            dim3 blocks(iDivUp(no_of_batches, BLOCK_SIZE));
            dim3 threads(min(no_of_batches, BLOCK_SIZE));
        	StepsignSGDGPU <<<blocks, threads>>>(x, y, d_x, alpha, h, n, m, beta, d_momentum, d_device_server_v, no_of_batches);


            gpuErrchk(cudaPeekAtLastError());
            gpuErrchk(cudaDeviceSynchronize());

			gpuErrchk(cudaMemcpy(temp_d_x, d_x, m * sizeof(float), cudaMemcpyDeviceToHost));
			gpuErrchk(cudaMemcpy(temp_d_v, d_device_server_v, no_of_batches * m * sizeof(float), cudaMemcpyDeviceToHost));
            
            for(int j=0; j<no_of_batches; j++)  
            {
                for(int k=0; k<m; k++)
                {   
                    temp_d_g_sum[k] += temp_d_v[j*m + k];
                }
            }

            for(int i=0; i<m; i++)
            {   
                temp_d_x[i] -= (alpha * ((temp_d_g_sum[i])>0?1:-1) + lambda * temp_d_x[i]);
            }	
			gpuErrchk(cudaMemcpy(d_x, temp_d_x, m * sizeof(float), cudaMemcpyHostToDevice));
		}
        // signADAM
        else if(type == 3)
        {  
            // --- Setting block and threads count for kernel #TODO: Change to a block setting
            dim3 blocks(iDivUp(no_of_batches, BLOCK_SIZE));
            dim3 threads(min(no_of_batches, BLOCK_SIZE));
            StepsignADAMGPU <<<blocks, threads>>>(x, y, d_x, alpha, h, n, m, beta_1, beta_2, epsilon, d_momentum, d_v, d_device_server_v, no_of_batches, *niter);

            gpuErrchk(cudaPeekAtLastError());
            gpuErrchk(cudaDeviceSynchronize());

            gpuErrchk(cudaMemcpy(temp_d_x, d_x, m * sizeof(float), cudaMemcpyDeviceToHost));
            gpuErrchk(cudaMemcpy(temp_d_v, d_device_server_v, no_of_batches * m * sizeof(float), cudaMemcpyDeviceToHost));
            
            for(int j=0; j<no_of_batches; j++)  
            {
                for(int k=0; k<m; k++)
                {   
                    temp_d_g_sum[k] += temp_d_v[j*m + k];
                }
            }

            for(int i=0; i<m; i++)
            {   
                temp_d_x[i] -= alpha * ((temp_d_g_sum[i])>0?1:-1);
            }   
            gpuErrchk(cudaMemcpy(d_x, temp_d_x, m * sizeof(float), cudaMemcpyHostToDevice));

            //*gnorm  = sqrt(thrust::inner_product(dev_ptr_g,     dev_ptr_g + m,      dev_ptr_g,      0.0f));
            //*dx     = sqrt(thrust::inner_product(dev_ptr_xdiff, dev_ptr_xdiff + m,  dev_ptr_xdiff,  0.0f));
        }
    // signSVRG
    else if(type == 4)
    {
        dim3 blocks(1);
        dim3 threads(min(no_of_batches, BLOCK_SIZE));

        /**Full Gradient Matrix**/
        float sigmoid_h[n];
        float h_x = 0.00;
        for(int j = 0; j < n; j++){
            h_x = 0;
            for(int k = 0; k < m; k++){
                h_x += temp_d_x[k] * ajax[j*m + k];
            }
            sigmoid_h[j] = sigmoid_value(h_x);
        }

        float dz[n];
        for(int j = 0; j < n ; j++){
            dz[j] = sigmoid_h[j] - achilles[j];
        }

        for(int j = 0; j < m; j++){
            nu[j] = 0;
            for(int k = 0; k < n; k++){
                nu[j] += ajax[k*m + j] * dz[k];
            }
        }
        /**Full Gradient Matrix**/

        gpuErrchk(cudaMemcpy(d_nu, nu, m * sizeof(float), cudaMemcpyHostToDevice));
        StepsignSVRGGPU <<<blocks, threads>>>(x, y, d_x, alpha, h, n, m, d_w_s, d_nu, d_device_server_v, no_of_batches,d_momentum, beta, *niter);
        gpuErrchk(cudaMemcpy(temp_d_x, d_x, m * sizeof(float), cudaMemcpyDeviceToHost));

        gpuErrchk(cudaPeekAtLastError());
        gpuErrchk(cudaDeviceSynchronize());

        gpuErrchk(cudaMemcpy(temp_d_x, d_x, m * sizeof(float), cudaMemcpyDeviceToHost));
        gpuErrchk(cudaMemcpy(temp_d_v, d_device_server_v, no_of_batches * m * sizeof(float), cudaMemcpyDeviceToHost));
        
        for(int j=0; j<no_of_batches; j++)  
        {
            for(int k=0; k<m; k++)
            {   
                temp_d_g_sum[k] += temp_d_v[j*m + k];
            }
        }

        for(int i=0; i<m; i++)
        {   
            if(temp_d_g_sum[i] != 0)
                temp_d_x[i] -= alpha * ((temp_d_g_sum[i])/*>0?1:-1*/);

            temp_d_g_sum[i] = 0;
        }   

        gpuErrchk(cudaMemcpy(d_x, temp_d_x, m * sizeof(float), cudaMemcpyHostToDevice));
    }
    // signCumSVRG
    else if(type == 5)
    {
        dim3 blocks(1);
        dim3 threads(min(no_of_batches, BLOCK_SIZE));

        gpuErrchk(cudaMemcpy(d_x, temp_d_x, m * sizeof(float), cudaMemcpyHostToDevice));
        SVRGFullGradientGPU <<<blocks, threads>>>(n, m, no_of_batches, x, y, d_x, d_full_gradient_batch_values);
        gpuErrchk(cudaMemcpy(full_gradient_batch_values, d_full_gradient_batch_values, m * no_of_batches * sizeof(float), cudaMemcpyDeviceToHost));
        gpuErrchk(cudaMemcpy(temp_d_x, d_x, m * sizeof(float), cudaMemcpyDeviceToHost));

        gpuErrchk(cudaPeekAtLastError());
        gpuErrchk(cudaDeviceSynchronize());

        for(int j=0; j<no_of_batches; j++)  
        {
            for(int k=0; k<m; k++)
            {   
                temp_d_g_sum[k] += full_gradient_batch_values[j*m + k]/no_of_batches;
            }
        }

        for(int i=0; i<svrg_m; i++){
            int cur_batch = rand()%no_of_batches;
            float h_x = 0.00;
            int lower_limit = cur_batch * (n/no_of_batches), upper_limit = cur_batch * (n/no_of_batches) + (n/no_of_batches);
            for(int j = lower_limit; j < min(upper_limit,n); j++){
                h_x = 0;
                for(int k = 0; k < m; k++){
                    h_x += temp_d_x[k] * ajax[j*m + k];
                }
                for(int k=0; k<m; k++)
                    cur_batch_values[k] += ajax[j*m + k] *  (sigmoid_value(h_x) - achilles[j]);
            }


            for(int j=0; j<m; j++)
            {      
                temp_d_x[j] -= alpha * ((cur_batch_values[j] - full_gradient_batch_values[cur_batch*m + j] + temp_d_g_sum[j])>0?1:-1);

              //  NOTE: GIVES GOOD RESULT EVEN IF UNCOMMENTED temp_d_g_sum[j] = 0;
            }
        }   

        gpuErrchk(cudaMemcpy(d_x, temp_d_x, m * sizeof(float), cudaMemcpyHostToDevice));
    }
    // signSVRG
    else if(type == 6)
    {
        dim3 blocks(1);
        dim3 threads(min(no_of_batches, BLOCK_SIZE));

        gpuErrchk(cudaMemcpy(d_x, temp_d_x, m * sizeof(float), cudaMemcpyHostToDevice));
        gpuErrchk(cudaMemcpy(d_full_gradient_batch_values, full_gradient_batch_values, m * no_of_batches * sizeof(float), cudaMemcpyHostToDevice));
        SVRGFullGradientGPU <<<blocks, threads>>>(n, m, no_of_batches, x, y, d_x, d_full_gradient_batch_values);
        gpuErrchk(cudaMemcpy(full_gradient_batch_values, d_full_gradient_batch_values, m * no_of_batches * sizeof(float), cudaMemcpyDeviceToHost));
        gpuErrchk(cudaMemcpy(temp_d_x, d_x, m * sizeof(float), cudaMemcpyDeviceToHost));

        gpuErrchk(cudaPeekAtLastError());
        gpuErrchk(cudaDeviceSynchronize());

        for(int j=0; j<no_of_batches; j++)  
        {
            for(int k=0; k<m; k++)
            {   
                temp_d_g_sum[k] += full_gradient_batch_values[j*m + k]/no_of_batches;
            }
        }

        for(int i=0; i<svrg_m; i++){
            int cur_batch = rand()%no_of_batches;
            float h_x = 0.00;
            int lower_limit = cur_batch * (n/no_of_batches), upper_limit = cur_batch * (n/no_of_batches) + (n/no_of_batches);
            for(int j = lower_limit; j < min(upper_limit,n); j++){
                h_x = 0;
                for(int k = 0; k < m; k++){
                    h_x += temp_d_x[k] * ajax[j*m + k];
                }
                for(int k=0; k<m; k++)
                    cur_batch_values[k] += ajax[j*m + k] *  (sigmoid_value(h_x) - achilles[j]);
            }


            for(int j=0; j<m; j++)
            {      
                temp_d_x[j] -= alpha * ((cur_batch_values[j] - full_gradient_batch_values[cur_batch*m + j] + temp_d_g_sum[j])/*>0?1:-1*/);
                cur_batch_values[j] = 0;
            }
        }   

        for(int i=0; i<no_of_batches; i++){
            for(int j=0; j<m; j++){
                full_gradient_batch_values[i*m + j] = 0;
            }
        }

        for(int i=0;i<m;i++){
            temp_d_g_sum[i] = 0;
        }


        gpuErrchk(cudaMemcpy(d_x, temp_d_x, m * sizeof(float), cudaMemcpyHostToDevice));
    }
    *niter  = *niter + 1;
    }

    t = clock() - t; 
    double time_taken = ((double)t)/CLOCKS_PER_SEC;
    printf("fun() took %f seconds to execute \n", time_taken); 
    gpuErrchk(cudaMemcpy(d_xopt, d_x, m * sizeof(float), cudaMemcpyDeviceToDevice));

    *niter = *niter - 1;
}
