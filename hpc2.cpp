#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <vector>
#include <chrono>
using namespace std;


void sequentialBubbleSort(int arr[], int n) {
    for (int i = 0; i < n-1; i++) {

        for (int j = 0; j < n-i-1; j++) {

            if (arr[j] > arr[j+1]) {
                
                int temp = arr[j];
                arr[j] = arr[j+1];
                arr[j+1] = temp;
            }
        }
    }
}

void parallelBubbleSort(int a[], int n)
{
    for(  int i = 0;  i < n;  i++ ){
        int first = i % 2;

        #pragma omp parallel for shared(a,first)
        for(int j = first;  j < n-1;  j += 2){
            if(a[j] > a[j+1]){

                int temp = a[j];
                a[j] = a[j+1];
                a[j+1] = temp;
            }
        }
    }
}


void merge(int array[],int low, int mid, int high,int size){
    int temp[size];
    int i = low;
    int j = mid + 1;
    int k = 0;
    while((i <= mid) && (j <= high)){
        if(array[i] >= array[j]){
            temp[k] = array[j];
            k++;
            j++;
        }
        else{
            temp[k] = array[i];
            k++;
            i++;
        }
    }
    while(i <= mid){
        temp[k] = array[i];
        k++;
        i++;
    }
    while(j <= high){
        temp[k] = array[j];
        k++;
        j++;
    }

    k = 0;
    for(int i = low;i <= high;i++){
        array[i] = temp[k];
        k++;
    }
}

void sequentialMergeSort(int array[],int low,int high,int size){
    if(low < high){
        int mid = low + (high - low) / 2;
        sequentialMergeSort(array,low,mid,size);
        sequentialMergeSort(array,mid+1,high,size);
        merge(array,low,mid,high,size);
    }
}


void parallelMergeSort(int arr[], int low, int high,int size) {
    if (low < high) {
        int mid = low + (high - low) / 2;

        #pragma omp parallel sections
        {
            #pragma omp section
            parallelMergeSort(arr, low, mid,size);
            #pragma omp section
            parallelMergeSort(arr, mid + 1, high,size);
        }

        merge(arr, low, mid, high,size);
    }
}

int main() {
    int n = 10000; // Size of the array
    int arr[10000], arr_copy[10000];

    // Initialize array with random values
    for (int i = 0; i < n; i++) {
        arr[i] = rand() % 10000;
    }

    // Copy the array for sequential Bubble Sort
    for (int i = 0; i < n; i++) {
        arr_copy[i] = arr[i];
    }

    // Measure time for sequential Bubble Sort
    double start_time1 = omp_get_wtime();
    sequentialBubbleSort(arr_copy, n);
    double sequential_bubble_sort_time = omp_get_wtime() - start_time1;

    // Copy the array for sequential Merge Sort
    for (int i = 0; i < n; i++) {
        arr_copy[i] = arr[i];
    }

    // Measure time for sequential Merge Sort
    double start_time2 = omp_get_wtime();
    sequentialMergeSort(arr_copy, 0, n - 1,n);
    double sequential_merge_sort_time = omp_get_wtime() - start_time2;

    // Measure time for parallel Bubble Sort
    double start_time3 = omp_get_wtime();
    parallelBubbleSort(arr, n);
    double parallel_bubble_sort_time = omp_get_wtime() - start_time3;

    // Measure time for parallel Merge Sort
    double start_time4 = omp_get_wtime();
    parallelMergeSort(arr, 0, n - 1,n);
    double parallel_merge_sort_time = omp_get_wtime() - start_time4;

    // Output the times
    printf("Sequential Bubble Sort Time: %f seconds\n", sequential_bubble_sort_time);
    printf("Sequential Merge Sort Time: %f seconds\n", sequential_merge_sort_time);
    printf("Parallel Bubble Sort Time: %f seconds\n", parallel_bubble_sort_time);
    printf("Parallel Merge Sort Time: %f seconds\n", parallel_merge_sort_time);

    return 0;

}