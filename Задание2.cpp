#include <iostream>
#include <vector>
#include <complex>
#include <random>
#include <chrono>
#include <iomanip>
#include <cblas.h> //  Заменяем mkl.h на cblas.h (стандартный заголовочный файл BLAS)

const int sz = 1024;
const int a = -10;//минимальное число
const int b = 31;//диапазон
using namespace std;

int main() {
    system("chcp 1251");
    system("cls");
    
    random_device rd;
    mt19937 gen(rd());
    uniform_real_distribution<float> dis(a, a + b);
    // Использование std::vector для хранения матриц
    vector<complex<float>> A(sz * sz);
    vector<complex<float>> B(sz * sz);
    vector<complex<float>> C(sz * sz, complex<float>(0.0f, 0.0f)); // Инициализируем нулями

    // Заполнение матриц случайными числами
    for (int i = 0; i < sz * sz; ++i) {
        A[i] = complex<float>(dis(gen), dis(gen));
        B[i] = complex<float>(dis(gen), dis(gen));
    }

    // Параметры для cblas_cgemm
    CBLAS_LAYOUT layout = CblasRowMajor;  // Row-major order (как в C++)
    CBLAS_TRANSPOSE transa = CblasNoTrans;
    CBLAS_TRANSPOSE transb = CblasNoTrans;
    const int m = sz;
    const int n = sz;
    const int k = sz;
    complex<float> alpha(1.0f, 0.0f); // alpha = 1 + 0i
    complex<float> beta(0.0f, 0.0f);   // beta = 0 + 0i
    const int lda = sz; // leading dimension of A
    const int ldb = sz; // leading dimension of B
    const int ldc = sz; // leading dimension of C

    // Засекаем время перед вызовом cblas_cgemm
    auto start = chrono::high_resolution_clock::now();

    // Вызываем cblas_cgemm из BLAS
    cblas_cgemm(layout, transa, transb, m, n, k, &alpha, A.data(), lda, B.data(), ldb, &beta, C.data(), ldc);

    // Засекаем время после вызова cblas_cgemm
    auto end = chrono::high_resolution_clock::now();
    chrono::duration<double> elapsed_seconds = end - start;
    double time = elapsed_seconds.count();

    // Расчет сложности
    double complexity = 2.0 * pow(sz, 3); // c = 2 * n^3

    // Расчет производительности в MFlops
    double mflops = complexity / time / 1e6; // p = c / t * 10^-6

    cout << fixed << setprecision(2); // Фиксируем вывод с двумя знаками после запятой

    cout << "Размер массива C: " << sizeof(complex<float>) * sz * sz << " bytes" << endl;
    cout << "Сложность алгоритма: " << complexity << endl;
    cout << "Время выполнения (cblas_cgemm): " << time << " секунд" << endl;
    cout << "Производительность (cblas_cgemm): " << mflops << " MFlops" << endl;

    // Вывод нескольких случайных элементов (можно закомментировать для больших матриц)
    for (int i = 0; i < 5; ++i) {
        int row = rand() % sz;
        int col = rand() % sz;
        cout << "C[" << row << "][" << col << "] = " << C[row * sz + col] << endl;
    }

    int i;
    cin >> i;
    return 0;
}
