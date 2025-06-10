#include <iostream>
#include <vector>
#include <complex>
#include <random>
#include <chrono>
#include <iomanip>
#include <cblas.h> 
const int sz = 1024;
const int a = -10;//минимальное число
const int b = 31;//диапазон
using namespace std;

int main() {
    system("chcp 1251");
    system("cls");
    cout << "Выполнил: Карпенко Денис Иванович\nГруппа: 020303-АИСа-24о\n";
    random_device rd;
    mt19937 gen(rd());
    uniform_real_distribution<float> dis(a, a + b);
    //vector для хранения матриц
    vector<complex<float>> A(sz * sz);
    vector<complex<float>> B(sz * sz);
    vector<complex<float>> C(sz * sz, complex<float>(0.0f, 0.0f));

    
    for (int i = 0; i < sz * sz; ++i) {
        A[i] = complex<float>(dis(gen), dis(gen));
        B[i] = complex<float>(dis(gen), dis(gen));
    }

    
    CBLAS_LAYOUT layout = CblasRowMajor; 
    CBLAS_TRANSPOSE transa = CblasNoTrans;
    CBLAS_TRANSPOSE transb = CblasNoTrans;
    const int m = sz;
    const int n = sz;
    const int k = sz;
    complex<float> alpha(1.0f, 0.0f); // alpha = 1 + 0i
    complex<float> beta(0.0f, 0.0f);   // beta = 0 + 0i
    const int lda = sz; 
    const int ldb = sz; 
    const int ldc = sz; 
    
    auto start = chrono::high_resolution_clock::now();

    cblas_cgemm(layout, transa, transb, m, n, k, &alpha, A.data(), lda, B.data(), ldb, &beta, C.data(), ldc);

    auto end = chrono::high_resolution_clock::now();
    chrono::duration<double> elapsed_seconds = end - start;
    double time = elapsed_seconds.count();

    double complexity = 2.0 * pow(sz, 3); // c = 2 * n^3

    double mflops = complexity / time / 1e6; // p = c / t * 10^-6

    cout << fixed << setprecision(2); 

    cout << "Размер массива C: " << sizeof(complex<float>) * sz * sz / 1024 << " KB" << endl;
    cout << "Сложность алгоритма: " << complexity << endl;
    cout << "Время выполнения (cblas_cgemm): " << time << " секунд" << endl;
    cout << "Производительность (cblas_cgemm): " << mflops << " MFlops" << endl;
    
    return 0;
}
