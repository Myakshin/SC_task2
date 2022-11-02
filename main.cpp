#include <iostream>
#include <cmath>
#include <random>
#include "mpi.h"

const double DEFAULT_INTEGRAL = M_PI / 8 * (1 - sin(1)); 

double f(double x, double y, double z) { 
    return y * sin(x * x + z * z);
}

bool G(double x, double y, double z) {
    return x >= 0 && y >= 0 && z >= 0 && x * x + y * y + z * z <= 1;
}

double F(double x, double y, double z) {
    return (G(x, y, z)) ? f(x, y, z) : 0.0;
}

double MonteCarlo(int n) {
    double sum = 0.0;
    for (int i = 0; i < n; ++i) {
        double x = double(rand()) / RAND_MAX;
        double y = double(rand()) / RAND_MAX;
        double z = double(rand()) / RAND_MAX;
        sum += F(x, y, z);
    }
    return sum;
}

double CalculateIntegral(double sum, int n) {
    double volume = 1.0 * 1.0 * 1.0;
    return volume * sum / n;
}
int main(int argc, char** argv) {
    if (argc != 2) {
        std::cout << "Неверное число аргументов" << std::endl;
        return -1;
    }
    double eps = atof(argv[1]);
    MPI_Init(&argc, &argv);
    int N_processes, process_id;
    MPI_Comm_size(MPI_COMM_WORLD, &N_processes);
    MPI_Comm_rank(MPI_COMM_WORLD, &process_id);
    srand(time(NULL) + process_id * N_processes);
    double start_time = MPI_Wtime();
    double integral = 0.0; 
    int n_points = 10000 / N_processes; 
    double local_sum = 0; 
    double global_sum = 0; 
    int cnt_points = 0; 
    do {
        local_sum += MonteCarlo(n_points); 
        MPI_Allreduce(&local_sum, &global_sum, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        cnt_points += n_points * N_processes; 
        integral = CalculateIntegral(global_sum, cnt_points); 
    } while (fabs(integral - DEFAULT_INTEGRAL) > eps); 

    double end_time = MPI_Wtime(); 
    if (process_id == 0) {
        std::cout << "Приближенное значение интеграла:\t" << integral << std::endl;
        std::cout << "Ошибка посчитанного значения:\t" << fabs(integral - DEFAULT_INTEGRAL) << std::endl;
        std::cout << "Количество сгенерированных случайных точек:\t" << cnt_points << std::endl;
        std::cout << "Время работы выполнения:\t" << end_time - start_time << std::endl;
    }
    MPI_Finalize();
    return 0;
}