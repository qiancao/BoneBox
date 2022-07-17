
#include <iostream>
#include <../bonebox/metrics/Tensor.cu>

int main() {

    const int K = 2;
    const int M = 3;
    const int N = 3;

    float a[K][M] = {{1.,2.,3.},{4.,5.,6.}};
    float b[K][N] = {{0.1,0.2,0.3},{0.4,0.5,0.6}};
    float out[K][M][N];

    for (int k=0; k<K; k++) {
        for (int m=0; m<M; m++) {
            for (int n=0; n<N; n++) {
                out[k][m][n] = a[k][m]*b[k][n];
            }
        }
    }

    // Print out array
    for (int k=0; k<K; k++) {
        for (int m=0; m<M; m++) {
            for (int n=0; n<N; n++) {
                std::cout << out[k][n][m] << ' ';
            }
        }
        std::cout << std::endl;
    }

}

