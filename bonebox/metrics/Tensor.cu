
//https://stackoverflow.com/questions/18844976/cuda-error-invalid-image-during-cumoduleload

extern "C" 
{
    __global__ void outer(double* a, double* b, double* out, int M, int N, int K)
    {
        /*
        
            Performs outer product for each row on the GPU
            Qian Cao 20220715
            a (K,M)
            b (K,N)
            out (K,M,N)

        */

        // tid indexes k
        unsigned int tid = blockDim.x * blockIdx.x + threadIdx.x;
        unsigned int ind_a, ind_b, ind_out;
        unsigned int k, m, n;

        if (tid < K) {

            k = tid;

            for (m=0; m<M; m++) {
                
                // index for array a
                ind_a = k*M + m;

                for (n=0; n<N; n++) {
                    
                    // index for array b and array out
                    ind_b = k*N + n;
                    ind_out = k*M*N + m*N + n;

                    // compute matrix entry
                    out[ind_out] = a[ind_a]*b[ind_b];
                }
            }
        }
    }
}
