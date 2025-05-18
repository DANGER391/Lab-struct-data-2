#include <iostream>
#include <ctime>
#include <complex>
#include <vector>
#include <chrono>
const int sz = 1024;
const int a = -10;//минимальное число
const int b = 31;//диапазон
using namespace std;

int main() {
    system("chcp 1251");
    system("cls");
    srand(time(0));

    vector<vector<complex<float>>> acm(sz, vector<complex<float>>(sz));
    vector<vector<complex<float>>> bcm(sz, vector<complex<float>>(sz));
    vector<vector<complex<float>>> cm(sz, vector<complex<float>>(sz));

    for (int i = 0; i < sz; i++) {
        for (int j = 0; j < sz; j++) {
            acm[i][j] = complex<float>(a + (rand() % b), a + (rand() % b));
        }
    }
    for (int i = 0; i < sz; i++) {
        for (int j = 0; j < sz; j++) {
            bcm[i][j] = complex<float>(a + (rand() % b), a + (rand() % b));

        }
    }
    auto start = chrono::high_resolution_clock::now();
    for (int i = 0; i < sz; i++) {
        for (int j = 0; j < sz; j++) {
            complex<float> js(0.0, 0.0);
            for (int k = 0; k < sz; k++) {
                js += acm[i][k] * bcm[k][j];
            }
            cm[i][j] = js;
        }
    }
    auto end = chrono::high_resolution_clock::now();
    chrono::duration<double> elapsed_seconds = end - start; 
    double time = elapsed_seconds.count();

    for (int i = 0; i < 5; ++i) {
        int row = rand() % sz;
        int col = rand() % sz;
        cout << "cm[" << row << "][" << col << "] = " << cm[row][col] << endl;
    }
    // Расчет производительности в MFlops
    double complexity = 2.0 * pow(sz, 3);
    double mfl = complexity / time / 1e6; 
    cout << fixed << setprecision(2); // Фиксируем вывод с двумя знаками после запятой
    cout << "Размер массива cm: " << (sizeof(complex<float>) * sz * sz)/1024 << " KB" << endl;
    cout << "Сложность алгоритма: " << complexity << endl;
    cout << "Время выполнения: " << time << " секунд" << endl;
    cout << "Производительность: " << mfl << " MFlops" << endl;
    return 0;
}
