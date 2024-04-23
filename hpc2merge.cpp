
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <vector>
#include <chrono>
using namespace std;


void merge(int array[], int low, int mid, int high, int size) {
    vector<int> temp(size); // Temporary vector for merging

    int i = low, j = mid + 1, k = 0;
    while (i <= mid && j <= high) {
        if (array[i] <= array[j]) {
            temp[k++] = array[i++];
        } else {
            temp[k++] = array[j++];
        }
    }

    while (i <= mid) {
        temp[k++] = array[i++];
    }

    while (j <= high) {
        temp[k++] = array[j++];
    }

    // Copy elements back to the original array
    for (k = 0; k < temp.size(); k++) {
        array[low + k] = temp[k];
    }
}

void sequentialMergeSort(int array[], int low, int high, int size) {
    if (low < high) {
        int mid = low + (high - low) / 2;
        sequentialMergeSort(array, low, mid, size);
        sequentialMergeSort(array, mid + 1, high, size);
        merge(array, low, mid, high, size);
    }
}

void parallelMergeSort(int arr[], int low, int high, int size) {
    if (low < high) {
        int mid = low + (high - low) / 2;

        #pragma omp parallel sections
        {
            #pragma omp section
            parallelMergeSort(arr, low, mid, size);

            #pragma omp section
            parallelMergeSort(arr, mid + 1, high, size);
        }

        merge(arr, low, mid, high, size);
    }
}

int main() {
    int n = 10000; // Size of the array
    int arr[n], arr_copy[n];

    // Initialize array with random values
    for (int i = 0; i < n; i++) {
        arr[i] = rand() % 10000;
    }


    // Copy the array for sequential Merge Sort
    for (int i = 0; i < n; i++) {
        arr_copy[i] = arr[i];
    }

    // Measure time for sequential Merge Sort
    double start_time2 = omp_get_wtime();
    sequentialMergeSort(arr_copy, 0, n - 1, n);
    double sequential_merge_sort_time = omp_get_wtime() - start_time2;

    // Copy the array for parallel Merge Sort
    for (int i = 0; i < n; i++) {
        arr_copy[i] = arr[i];
    }

    // Measure time for parallel Merge Sort
    double start_time4 = omp_get_wtime();
    parallelMergeSort(arr_copy, 0, n - 1, n);
    double parallel_merge_sort_time = omp_get_wtime() - start_time4;

    // Output the times
    printf("Sequential Merge Sort Time: %f seconds\n", sequential_merge_sort_time);
    printf("Parallel Merge Sort Time: %f seconds\n", parallel_merge_sort_time);

    return 0;
}

