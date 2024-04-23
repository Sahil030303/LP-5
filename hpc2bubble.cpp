#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <vector>
#include<iostream>
#include<bits/stdc++.h>
using namespace std;

void sequentialBubbleSort(int arr[], int n){

    for(int i = 0; i<n-1; i++){
        
        for(int j =0; j<n-i-1; j++){

            if(arr[j] > arr[j+1]){
                swap(arr[j],arr[j+1]);
            }
        }
    }
}

void parallelBubbleSort(int arr[], int n){

    for(int i = 0; i<n-1; i++){

        int first = i%2;

        #pragma omp parallel for shared(arr,first)
        for(int j= first; j<n-1; j += 2){
            if(arr[j] > arr[j+1]){
                swap(arr[j],arr[j+1]);
            }
        }
    }
}

int main(){
    int size = 10000;
    int arr[size];
    int arr_copy[size];

    for(int i = 0; i<size; i++){
        arr[i] = rand()%10000;
    }

    for(int i = 0; i < size; i++){
        arr_copy[i] = arr[i];
    }

    double start_time = omp_get_wtime();
    sequentialBubbleSort(arr_copy,size);
    double sequential_bubble_sort_time = omp_get_wtime() - start_time;
    cout<<endl;

     for(int i = 0; i< size; i++){
        arr_copy[i] = arr[i];
    }
    double start_time1 = omp_get_wtime();
    parallelBubbleSort(arr_copy,size);
    double parallel_time = omp_get_wtime() - start_time1;

    cout<<" sequential time "<<sequential_bubble_sort_time<<endl;
    cout<<"parallel time "<<parallel_time<<endl;
}
