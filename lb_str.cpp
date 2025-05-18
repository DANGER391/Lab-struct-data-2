#include <iostream>
#include <vector>
#include <complex>
#include <random>
#include <chrono>
#include <iomanip>
#include <immintrin.h> 
using namespace std;
const int sz = 1024;
const int a = -10;
const int b = 31;

void multiply_avx(const float* A, const float* B, float* C) {
    for (int i = 0; i < sz; ++i) {
        for (int j = 0; j < sz; ++j) {
            __m256 sum = _mm256_setzero_ps(); 
            for (int k = 0; k < sz; k += 8) { 
                __m256 a_vec = _mm256_loadu_ps(&A[i * sz + k]); 
                __m256 b_vec = _mm256_loadu_ps(&B[j * sz + k]); 
                sum = _mm256_fmadd_ps(a_vec, b_vec, sum); 
            }
            
            float sum_arr[8];
            _mm256_storeu_ps(sum_arr, sum);
            C[i * sz + j] = sum_arr[0] + sum_arr[1] + sum_arr[2] + sum_arr[3] +
                sum_arr[4] + sum_arr[5] + sum_arr[6] + sum_arr[7];
        }
    }
}

int main() {
    system("chcp 1251");
    system("cls");
    srand(time(0));

    vector<float> A(sz * sz);
    vector<float> B(sz * sz);
    vector<float> C(sz * sz, 0.0f);

    random_device rd;
    mt19937 gen(rd());
    uniform_real_distribution<float> dis(a, a + b);

    for (int i = 0; i < sz * sz; ++i) {
        A[i] = dis(gen);
        B[i] = dis(gen);
    }

    auto start = chrono::high_resolution_clock::now();
    multiply_avx(A.data(), B.data(), C.data());
    auto end = chrono::high_resolution_clock::now();

    chrono::duration<double> elapsed_seconds = end - start;
    double time = elapsed_seconds.count();

    double complexity = 2.0 * pow(sz, 3);
    double mfl = complexity / time / 1e6;

    cout << fixed << setprecision(2);
    cout << "Размер массива cm: " << (sizeof(complex<float>) * sz * sz) / 1024 << " KB" << endl;
    cout << "Сложность алгоритма: " << complexity << endl;
    cout << "Время выполнения: " << time << " секунд" << endl;
    cout << "Производительность: " << mfl << " MFlops" << endl;
    return 0;
}