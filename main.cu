#include <chrono>
#include <iostream>
#include <vector>
#include <immintrin.h>
#include <stdexcept>
#include <thread>

using namespace std;

__global__ void matrixMulKernel(const float* A, const float* B, float* C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < N; k++) {
            sum += A[row * N + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

class Matrix {
private:
    int rows;
    int cols;
    vector<float> data;
public:
    Matrix(int r, int c, float value=0.0f) {
        rows = r;
        cols = c;
        data.resize(r * c, value);
    }
    float& operator()(int r, int c) {
        return data[r * cols + c];
    }

    float operator()(int r, int c) const {
        return data[r * cols + c];
    }

    void print() const {
        for (int r = 0; r < rows; r++) {
            for (int c = 0; c < cols; c++) {
                cout << data[r * cols + c] << " ";
            }
            cout << endl;
        }
    }

    float* dataPtr() { return data.data(); }
    const float* dataPtr() const { return data.data(); }

    Matrix multiplySIMD(const Matrix& B) const {
        if (this->cols != B.rows) {
            throw invalid_argument("Niepoprawne wymiary macierzy");
        }

        Matrix C(this->rows, B.cols, 0.0f);

        const float* a_ptr = this->dataPtr();
        const float* b_ptr = B.dataPtr();
        float* c_ptr = C.dataPtr();

        int N = this->rows;

        for (int i = 0; i < N; i++) {
            for (int k = 0; k < N; k++) {

                __m256 a_vec = _mm256_set1_ps(a_ptr[i * N + k]);

                for (int j = 0; j < N; j += 8) {

                    __m256 b_vec = _mm256_loadu_ps(&b_ptr[k * N + j]);
                    __m256 c_vec = _mm256_loadu_ps(&c_ptr[i * N + j]);

                    c_vec = _mm256_add_ps(c_vec, _mm256_mul_ps(a_vec, b_vec));

                    _mm256_storeu_ps(&c_ptr[i * N + j], c_vec);
                }
            }
        }
        return C;
    }
    Matrix multiplySIMD_MT(const Matrix& B) const {
        if (this->cols != B.rows) throw invalid_argument("Niepoprawne wymiary macierzy");

        Matrix C(this->rows, B.cols, 0.0f);
        const float* a_ptr = this->dataPtr();
        const float* b_ptr = B.dataPtr();
        float* c_ptr = C.dataPtr();
        int N = this->rows;

        int num_threads = std::thread::hardware_concurrency();
        std::vector<std::thread> threads;

        int chunk = N / num_threads;

        for (int t = 0; t < num_threads; t++) {
            int start_row = t * chunk;
            int end_row = (t == num_threads - 1) ? N : (t + 1) * chunk;

            threads.push_back(std::thread([=]() {


                for (int i = start_row; i < end_row; i++) {
                    for (int k = 0; k < N; k++) {

                        __m256 a_vec = _mm256_set1_ps(a_ptr[i * N + k]);

                        for (int j = 0; j < N; j += 8) {

                            __m256 b_vec = _mm256_loadu_ps(&b_ptr[k * N + j]);
                            __m256 c_vec = _mm256_loadu_ps(&c_ptr[i * N + j]);

                            c_vec = _mm256_add_ps(c_vec, _mm256_mul_ps(a_vec, b_vec));

                            _mm256_storeu_ps(&c_ptr[i * N + j], c_vec);
                        }
                    }
                }
            }));
        }
        for (auto& th : threads) {
            th.join();
        }

        return C;
    }

    Matrix multiplyCUDA(const Matrix& B) const {
        if (this->cols != B.rows) throw invalid_argument("Niepoprawne wymiary");

        int N = this->rows;
        Matrix C(N, N, 0.0f);
        size_t bytes = N * N * sizeof(float);

        float *d_A, *d_B, *d_C;
        cudaMalloc(&d_A, bytes);
        cudaMalloc(&d_B, bytes);
        cudaMalloc(&d_C, bytes);

        cudaMemcpy(d_A, this->dataPtr(), bytes, cudaMemcpyHostToDevice);
        cudaMemcpy(d_B, B.dataPtr(), bytes, cudaMemcpyHostToDevice);

        dim3 threadsPerBlock(16, 16);
        dim3 numBlocks((N + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (N + threadsPerBlock.y - 1) / threadsPerBlock.y);

        matrixMulKernel<<<numBlocks, threadsPerBlock>>>(d_A, d_B, d_C, N);

        cudaMemcpy(C.dataPtr(), d_C, bytes, cudaMemcpyDeviceToHost);

        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);

        return C;
    }
};


int main() {
    int N = 1024;
    cudaFree(0);
    Matrix A(N, N, 1.0f);
    Matrix B(N, N, 2.0f);

    auto startCPU = chrono::high_resolution_clock::now();
    Matrix C_cpu = A.multiplySIMD_MT(B);
    auto endCPU = chrono::high_resolution_clock::now();
    chrono::duration<double> diffCPU = endCPU - startCPU;
    cout << "1. Czas CPU (AVX2 + Wątki): " << diffCPU.count() << " s (Wynik: " << C_cpu(0,0) << ")\n";

    auto startGPU = chrono::high_resolution_clock::now();
    Matrix C_gpu = A.multiplyCUDA(B);
    auto endGPU = chrono::high_resolution_clock::now();
    chrono::duration<double> diffGPU = endGPU - startGPU;
    cout << "2. Czas GPU (CUDA): " << diffGPU.count() << " s (Wynik: " << C_gpu(0,0) << ")\n";

    return 0;
}