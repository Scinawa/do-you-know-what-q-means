#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <string.h>
#include <vector>
#include <bits/stdc++.h>
#include <random>
#include <iostream>
#include <fstream>
#include <armadillo>
#include <fstream>
#include "omp.h"

using namespace std;
using namespace arma;

int main (void) {

	auto t0 = chrono::high_resolution_clock::now();

	int d = 784, K = 10;
	int length_eps = 5;
    	double eps[length_eps] = {0.5, 1., 1.5, 2., 2.5};
    	double threshold = 0.15;
    	double delta = 0.01;
    	
    	int n_rows = 28;
    	int n_cols = 28;
    	int n_images = 60000;

	int n_iter_C = 6;
	int n_iter_N = 6;
	int length_N = 7;
	int N[length_N] = {20000, 25000, 30000, 35000, 40000, 45000, 50000};
		
    	srand(time(0));
	random_device rd;
	mt19937 gen(rd());
	default_random_engine generator;
    	
    	unsigned long int time_EK_Kmeans[length_N][length_eps] = {0};
  	 	
    	// Read MNIST dataset
    	vector<vector<double>> mnist_dataset(n_images, vector<double>(d));
    	
    	ifstream file ("t10k-images.idx3-ubyte", ios::binary);
    	for (int i = 0; i < n_images; ++i) {
		for (int r = 0; r < n_rows; ++r) {
	        	for (int c = 0; c < n_cols; ++c) {
	            		unsigned char temp = 0;
	            		file.read((char*)&temp,sizeof(temp));
	            		mnist_dataset[i][(n_rows*r)+c] = ((double)temp) / 255;
	        	}
	    	}
	}
        file.close();
    	
    	for (int n = 0; n < length_N; n++) {
    	
  		uniform_int_distribution<int> unif_distribution(0, N[n]-1);
  		
  		for (int iter_N = 0; iter_N < n_iter_N; iter_N++) {
    	
	    		vector<vector<double>> V(N[n], vector<double>(d));
	    		mat V_aux(N[n], d);
	  
	  		// Create the input
	  		unordered_set<int> elems;
			for (int r = n_images - N[n]; r < n_images; ++r) {
				int v = uniform_int_distribution<>(0, r)(gen);
				
				if (!elems.insert(v).second) 
				    elems.insert(r);
			}
	  		
	  		int counter = 0;
	    		for (auto x : elems) {
	    			for (int h = 0 ; h < d; h++) {
	    				V[counter][h] = mnist_dataset[x][h];
	    				V_aux(counter, h) = mnist_dataset[x][h];
	    			}
	    			counter++;
	    		}		    		    		
	    		
	    		// Repeating several initial centroids
	    		for (int iter_C = 0; iter_C < n_iter_C; iter_C++) { 	    	    		
	    				    								    		
 //////////////////////////////////////////////////////////////////////////////////////
		    				    													    					   	    			   							
				for (int e = 0; e < length_eps; e++) {
				
				    	// Initial time
				    	auto begin = chrono::high_resolution_clock::now();
				    	
				    	// Compute the norms
			    		double spectral_norm = norm2est(V_aux);
			    		vector<double> V_norms(N[n], 0);
			    		double V21_norm = 0;
			    		for (int i = 0; i < N[n]; i++) {
			    			for (int h = 0; h < d; h++)
			    				V_norms[i] += V[i][h] * V[i][h];

			    			V_norms[i] = sqrt(V_norms[i]);
			    			V21_norm += V_norms[i];
			    		}
			    		
			    		// Create the distribution for Q
				    	vector<double> Q_distr(N[n]);
				    	for (int i = 0; i < N[n]; i++)
						Q_distr[i] = V_norms[i] / V21_norm;
						
					discrete_distribution<> Q_dist(Q_distr.begin(), Q_distr.end());
				    	
				    	// Sample initial centroids uniformly from the dataset
				    	vector<vector<double>> C_eps(K, vector<double>(d));
				    	for (int i = 0; i < K; i++) {
				    		int j = unif_distribution(generator);
						C_eps[i] = V[j];
					}
				
			    		long int p = ceil( spectral_norm * spectral_norm / N[n] * K * K / (eps[e] * eps[e]) * log(K/delta) );
			    		long int q = ceil( (V21_norm / N[n])  * (V21_norm / N[n]) * K * K / (eps[e] * eps[e]) * log(K/delta) );
			    		
			 //   		cout << "p: " << p << endl;
			 //		cout << "q: " << q << endl;
				    	
					// Sample p vectors
					vector<vector<double>> V_P_sampled(p, vector<double>(d));
				    	for (int i = 0; i < p; i++) {
				    		int j = unif_distribution(generator);
				 		V_P_sampled[i] = V[j];
					}
					
					// Sample q vectors
					vector<vector<double>> V_Q_sampled(q, vector<double>(d));
					vector<vector<double>> V_Q_sampled_normalised(q, vector<double>(d));
					for (int i = 0; i < q; i++) {
						int j = Q_dist(gen);
						V_Q_sampled[i] = V[j];
						for (int l = 0; l < d; l++)
							V_Q_sampled_normalised[i][l] = V[j][l] / V_norms[j];
					}				    	
					    				
				    	// Main EKK-means loop
				    	double distance_centroids = 2 * K * threshold;
				    	int iterations = 0.;
					while (distance_centroids > K * threshold and iterations < 50) {
//					while (distance_centroids > K * threshold) {
					
					    	// Compute cluster sizes
					    	vector<int> C_size(K, 0);
					    	for (int i = 0; i < p; i++) {
					    	
							int min_index = -1;
						    	double min_distance = DBL_MAX;
						    	for (int j = 0; j < K; j++) {
						    	
								double new_dist = 0;
								for (int h = 0; h < d; h++) 
							    		new_dist += abs((C_eps[j][h] - V_P_sampled[i][h]) * (C_eps[j][h] - V_P_sampled[i][h]));
						 
								if (new_dist < min_distance) {
							    		min_distance = new_dist;
							    		min_index = j;
								}
						    	}
							C_size[min_index]++;
					    	}
					    	
					    	
					    	// Compute new centroids
					    	vector<vector<double>> C_new(K, vector<double>(d,0));
					    	for (int i = 0; i < q; i++) {
					    	
					    		int min_index = -1;
						    	double min_distance = DBL_MAX;
						    	for (int j = 0; j < K; j++) {
						    	
								double new_dist = 0;
								for (int h = 0; h < d; h++) 
							    		new_dist += abs((C_eps[j][h] - V_Q_sampled[i][h]) * (C_eps[j][h] - V_Q_sampled[i][h]));
						 
								if (new_dist < min_distance) {
							    		min_distance = new_dist;
							    		min_index = j;
								}
						    	}

							for (int h = 0; h < d; h++)
						    		C_new[min_index][h] += V_Q_sampled_normalised[i][h];
					    	}
					    	
					    	for (int j = 0; j < K; j++) {
					    		if (C_size[j] == 0) 
					    			C_new[j] = C_eps[j];			    
					    		else {
					    			double coeff = (((V21_norm / N[n]) / q) * p) / C_size[j];
					    			for (int h = 0; h < d; h++)
					    				C_new[j][h] = coeff * C_new[j][h];
					    		}
					    	}
					    	
					    	// Compute distance between centroids
					    	distance_centroids = 0;	
					    	for (int j = 0; j < K; j++) {
					    	
					    		double distance_aux = 0;
							for (int h = 0; h < d; h++) 
						    		distance_aux += abs((C_eps[j][h] - C_new[j][h]) * (C_eps[j][h] - C_new[j][h]));
						  	distance_centroids += sqrt(abs(distance_aux));
					    	}

					    	C_eps = C_new;
					    	iterations += 1;					 
					}

				    	// Compute duration
				    	auto end = chrono::high_resolution_clock::now();
				    	auto dur = chrono::duration_cast<chrono::milliseconds>(end - begin).count();				    	
				    	time_EK_Kmeans[n][e] += dur;
				}
				cout << n << ' ' << iter_N << ' ' << iter_C << endl;								
			}
		}
    	}
    	
    	// Write the output onto a separate file
	ofstream MyFile("data_n_preprocessing.txt");
	MyFile << "iterations" << ' ' << n_iter_C * n_iter_N << endl;
	
	for (int n = 0; n < length_N; n++) {
		cout << "N: " << N[n] << endl;
		for (int e = 0; e < length_eps; e++) {
			MyFile << N[n] << ' ' << eps[e] << ' ' << time_EK_Kmeans[n][e] / ((double) (n_iter_N * n_iter_C)) << endl;		
			cout << "Miliseconds EKK: " << time_EK_Kmeans[n][e]  / ((double) (n_iter_N * n_iter_C)) << endl;
    		}
	}
	MyFile.close();
	
	auto t1 = chrono::high_resolution_clock::now();
	cout <<  chrono::duration_cast<chrono::seconds>(t1 - t0).count() << endl;
    	
    	return 0;

}
