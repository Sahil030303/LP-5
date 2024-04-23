#include <iostream>
#include <omp.h>
#include <cstdlib>
#include <ctime>
using namespace std;

// void sequentialBubbleSort(int arr[], int n) {
//     bool swapped;
//     for (int i = 0; i < n - 1; i++) {
//         swapped = false;
//         for (int j = 0; j < n - i - 1; j++) {
//             if (arr[j] > arr[j + 1]) {
//                 int temp = arr[j];
//                 arr[j] = arr[j + 1];
//                 arr[j + 1] = temp;
//                 swapped = true;
//             }
//         }
//         // If no swaps occurred, array is already sorted
//         if (!swapped) {
//             break;
//         }
//     }
// }

void sequentialBubbleSort(int arr[], int n)
{
    for (int i = 0; i < n - 1; ++i)
    {
        for (int j = 0; j < n - i - 1; ++j)
        {
            if (arr[j] > arr[j + 1])
            {
                swap(arr[j], arr[j + 1]);
            }
        }
    }
}

// void parallelBubbleSort(int a[], int n) {
//     bool swapped = true;
//     while (swapped) {
//         swapped = false;
//         #pragma omp parallel for shared(a, n, swapped)
//         for (int i = 0; i < n - 1; i++) {
//             if (a[i] > a[i + 1]) {
//                 int temp = a[i];
//                 a[i] = a[i + 1];
//                 a[i + 1] = temp;
//                 swapped = true; // Set swapped flag if any swap occurs
//             }
//         }
//         // Synchronize threads after each iteration to ensure all swaps are complete
//         #pragma omp barrier
//     }
// }

void parallelBubbleSort(int arr[], int n)
{
    for (int i = 0; i < n - 1; ++i)
    {
#pragma omp parallel for
        for (int j = 0; j < n - i - 1; ++j)
        {
            if (arr[j] > arr[j + 1])
            {
                swap(arr[j], arr[j + 1]);
            }
        }
    }
}

int main() {
    const int n = 10000; // Size of the array
    int arr[n];

    // Initialize array with random values
    srand(time(NULL)); // Seed the random number generator
    for (int i = 0; i < n; i++) {
        arr[i] = rand() % 10000; // Random numbers between 0 and 9999
    }

    // Copy the array for sorting
    int arr_copy[n];
    for (int i = 0; i < n; i++) {
        arr_copy[i] = arr[i];
    }

    // Perform sequential bubble sort and measure time
    double start_time_seq = omp_get_wtime();
    sequentialBubbleSort(arr_copy, n);
    double sequential_bubble_sort_time = omp_get_wtime() - start_time_seq;

    // Copy the array for parallel bubble sort
    for (int i = 0; i < n; i++) {
        arr_copy[i] = arr[i];
    }

    // Perform parallel bubble sort and measure time
    double start_time_par = omp_get_wtime();
    parallelBubbleSort(arr_copy, n);
    double parallel_bubble_sort_time = omp_get_wtime() - start_time_par;

    // Output the sorting times
    cout << "Sequential Bubble Sort Time: " << sequential_bubble_sort_time << " seconds" << endl;
    cout << "Parallel Bubble Sort Time: " << parallel_bubble_sort_time << " seconds" << endl;

    return 0;
}
