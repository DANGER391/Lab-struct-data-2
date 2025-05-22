#include <iostream>
#include <complex>
#include <vector>
#include <random>
#include <chrono>
#include <iomanip>
#ifdef _OPENMP
#include <omp.h>
#endif
using namespace std;
using Complex = std::complex<float>;
constexpr int SIZE = 1024;
const int a = -1;
const int b = 5;
constexpr int BLOCK_SIZE = 32;

void generateMatrix(std::vector<Complex>& mat) {
    mt19937 rng(12345); 
    uniform_real_distribution<float> dist(0.f, 1.f);

    for (int i = 0; i < SIZE * SIZE; ++i) {
        mat[i] = Complex(dist(rng), dist(rng));
    }
}

void multiplyMatricesBlocked(const std::vector<Complex>& A,
    const std::vector<Complex>& B,
    std::vector<Complex>& C) {
#pragma omp parallel for if(SIZE >= 256) schedule(static)
    for (int i = 0; i < SIZE * SIZE; ++i) {
        C[i] = Complex(0.0f, 0.0f);
    }

#pragma omp parallel for collapse(2) schedule(static)
    for (int iBlock = 0; iBlock < SIZE; iBlock += BLOCK_SIZE) {
        for (int jBlock = 0; jBlock < SIZE; jBlock += BLOCK_SIZE) {
            for (int kBlock = 0; kBlock < SIZE; kBlock += BLOCK_SIZE) {
                int iMax = std::min(iBlock + BLOCK_SIZE, SIZE);
                int jMax = std::min(jBlock + BLOCK_SIZE, SIZE);
                int kMax = std::min(kBlock + BLOCK_SIZE, SIZE);

                for (int i = iBlock; i < iMax; ++i) {
                    for (int k = kBlock; k < kMax; ++k) {
                        Complex a_val = A[i * SIZE + k];
                        int a_pos = i * SIZE + k;
                        for (int j = jBlock; j < jMax; ++j) {
                            int b_pos = k * SIZE + j;
                            int c_pos = i * SIZE + j;
                            C[c_pos] += a_val * B[b_pos];
                        }
                    }
                }
            }
        }
    }
}

int main() {
    system("chcp 1251");
    system("cls");
    cout << "Выполнил: Карпенко Денис Иванович\nГруппа: 020303-АИСа-24о\n";

    random_device rd;
    mt19937 gen(rd());
    uniform_real_distribution<float> dis(a, a + b);

    vector<Complex> A(SIZE * SIZE);
    vector<Complex> B(SIZE * SIZE);
    vector<Complex> C(SIZE * SIZE);

    for (int i = 0; i < SIZE * SIZE; ++i) {
        A[i] = complex<float>(dis(gen), dis(gen));
        B[i] = complex<float>(dis(gen), dis(gen));
    }

    cout << "\nВычисление первой матрицы C3:\n";
    auto start = chrono::high_resolution_clock::now();
    
    multiplyMatricesBlocked(A, B, C);

    auto end = chrono::high_resolution_clock::now();
    chrono::duration<double> el_s = end - start;
    double tt = el_s.count();
    double ctt = 2.0 * pow(SIZE, 3);
    double mfl3 = ctt / tt / 1e6;
    cout << fixed << setprecision(2);
    cout << "Размер массива cm: " << (sizeof(complex<float>) * SIZE * SIZE) / 1024 << " KB" << endl;
    cout << "Сложность алгоритма: " << ctt << endl;
    cout << "Время выполнения: " << tt << " секунд" << endl;
    cout << "Производительность: " << mfl3 << " MFlops" << endl;
#ifdef _OPENMP
    std::cout << "Использовано потоков OpenMP: " << omp_get_max_threads() << "\n";
#else
    std::cout << "OpenMP не используется.\n";
#endif
    cout << "Вывод случайных 5 элементов:\n";
    for (int i = 0; i < 5; ++i) {
        int row = rand() % SIZE;
        int col = rand() % SIZE;
        cout << "C1[" << row << "][" << col << "] = " << C[row * SIZE + col] << "\n";
    }
    return 0;
}