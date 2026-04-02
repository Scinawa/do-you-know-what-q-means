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
#include "omp.h"

using namespace std;
using namespace arma;

int main (void) {

	auto t0 = chrono::high_resolution_clock::now();

	int d = 30, K = 5;
	int length_eps = 5;
    	double eps[length_eps] = {0.2, 0.4, 0.6, 0.8, 1.0};
    	double threshold = 0.1;
    	double delta = 0.01;

	int n_iter_C = 30;
	int n_iter_N = 25;
	int length_N = 9;
	int N[length_N] = {100000, 150000, 200000, 250000, 300000, 350000, 400000, 450000, 500000};
		
    	srand(time(0));
	random_device rd;
	mt19937 gen(rd());
	default_random_engine generator;
    	
    	unsigned long int time_EK_Kmeans[length_N][length_eps] = {0};   
    	 
    	
    	for (int n = 0; n < length_N; n++) {
    	
  		uniform_int_distribution<int> unif_distribution(0, N[n]-1);
  		
  		for (int iter_N = 0; iter_N < n_iter_N; iter_N++) {
    	
	    		vector<vector<double>> C_aux(K, vector<double>(d));
	    		vector<vector<double>> V(N[n], vector<double>(d));
	    		mat V_aux(N[n], d);
	  
	  		// Create the input
	    		for (int j = 0; j < K; j++) {
	    			for (int h = 0; h < d; h++) 
	    				C_aux[j][h] = 2*((double)rand()) / RAND_MAX - 1;
	    		}
	    		
	    		for (int j = 0; j < K; j++) {
	    			for (int l = 0; l < N[n]/K; l++) {
	    				for (int h = 0 ; h < d; h++) {
	    					V[j * N[n] / K + l][h] = C_aux[j][h] + 2* ((double)rand()) / RAND_MAX - 1;
	    					V_aux(j * N[n] / K + l, h) = V[j * N[n] / K + l][h];
	    				}
	    			}
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
					while (distance_centroids > K * threshold and iterations < 20) {
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
