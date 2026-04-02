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

	int n_iter_C = 5;
	int n_iter_N = 5;
	int length_N = 7;
	int N[length_N] = {20000, 25000, 30000, 35000, 40000, 45000, 50000};
		
    	srand(time(0));
	random_device rd;
	mt19937 gen(rd());
	default_random_engine generator;
    	
    	vector<double> time_standard_Kmeans(length_N, 0);
    	vector<vector<double>> time_EK_means(length_N, vector<double>(length_eps, 0));
    	vector<vector<double>> time_mini_batch(length_N, vector<double>(length_eps, 0));
    	
    	vector<double> iterations_standard_Kmeans(length_N, 0);
    	vector<vector<double>> iterations_EK_means(length_N, vector<double>(length_eps, 0));
    	vector<vector<double>> iterations_mini_batch(length_N, vector<double>(length_eps, 0));
    	   	
    	vector<vector<double>> RSS_EK_means(length_N, vector<double>(length_eps, 0));
    	vector<vector<double>> ARI_EK_means(length_N, vector<double>(length_eps, 0));
    	vector<vector<double>> NMI_EK_means(length_N, vector<double>(length_eps, 0));
    	
    	vector<vector<double>> RSS_mini_batch(length_N, vector<double>(length_eps, 0));
    	vector<vector<double>> ARI_mini_batch(length_N, vector<double>(length_eps, 0));
    	vector<vector<double>> NMI_mini_batch(length_N, vector<double>(length_eps, 0));
    	 	
    	 	
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
	    		
	    		
	    		// Repeating several initial centroids
	    		for (int iter_C = 0; iter_C < n_iter_C; iter_C++) { 
	    		
		    		// Sample initial centroids uniformly from the dataset
			    	vector<vector<double>> C_initial(K, vector<double>(d));
			    	for (int i = 0; i < K; i++) {
			    		int j = unif_distribution(generator);
					C_initial[i] = V[j];
				}
							    		
 //////////////////////////////////////////////////////////////////////////////////////
		    		
		    		auto begin = chrono::high_resolution_clock::now();
				
				// Sample initial centroids uniformly from the dataset
				vector<vector<double>> C_standard(K, vector<double>(d));
				C_standard = C_initial;		
				
				// Main standard k-means loop
				double distance_centroids = 2 * K * threshold;
				double iterations = 0;
				while (distance_centroids > K*threshold) {
					
				    	// Compute cluster sizes
				    	vector<int> C_size(K, 0);
				    	for (int i = 0; i < N[n]; i++) {
				    	
						int min_index = -1;
					    	double min_distance = DBL_MAX;
					    	for (int j = 0; j < K; j++) {
					    	
							double new_dist = 0;
							for (int h = 0; h < d; h++) 
						    		new_dist += abs((C_standard[j][h] - V[i][h]) * (C_standard[j][h] - V[i][h]));
					 
							if (new_dist < min_distance) {
						    		min_distance = new_dist;
						    		min_index = j;
							}
					    	}
						C_size[min_index]++;
				    	}
				    	
				    	// Compute new centroids
				    	vector<vector<double>> C_new(K, vector<double>(d,0));
				    	for (int i = 0; i < N[n]; i++) {
				    	
				    		int min_index = -1;
					    	double min_distance = DBL_MAX;
					    	for (int j = 0; j < K; j++) {
					    	
							double new_dist = 0;
							for (int h = 0; h < d; h++) 
						    		new_dist += abs((C_standard[j][h] - V[i][h]) * (C_standard[j][h] - V[i][h]));
					 
							if (new_dist < min_distance) {
						    		min_distance = new_dist;
						    		min_index = j;
							}
					    	}

						for (int h = 0; h < d; h++)  
					    		C_new[min_index][h] += V[i][h];
				    	}
				    	
				    	for (int j = 0; j < K; j++) {
				    		if (C_size[j] == 0) {
				    			for (int h = 0; h < d; h++)
				    				C_new[j][h] = 0;
				    		}
				    		else {
				    			for (int h = 0; h < d; h++)
				    				C_new[j][h] = C_new[j][h] / C_size[j];
				    		}
				    	}
				    	
				    	// Compute distance between centroids
				    	distance_centroids = 0;	
				    	for (int j = 0; j < K; j++) {
				    	
				    		double distance_aux = 0;
						for (int h = 0; h < d; h++) 
					    		distance_aux += abs((C_standard[j][h] - C_new[j][h]) * (C_standard[j][h] - C_new[j][h]));
					  	distance_centroids += sqrt(abs(distance_aux));
				    	}
				//    	cout << distance_centroids << endl;
				  				    	
				    	C_standard = C_new;
				    	iterations += 1;
				}
				iterations_standard_Kmeans[n] += iterations / (n_iter_C * n_iter_N);

			    	// Compute duration
			    	auto end = chrono::high_resolution_clock::now();
			    	double dur = chrono::duration_cast<chrono::milliseconds>(end - begin).count();
			    	time_standard_Kmeans[n] += dur / (n_iter_N * n_iter_C);
			//    	cout << "Iterations standard: " << iterations << endl;		    					   
		    				    		
//////////////////////////////////////////////////////////////////////////////////////
		   			
				
				for (int e = 0; e < length_eps; e++) {
				
				    	// Initial time
				    	begin = chrono::high_resolution_clock::now();
				
			    		long int p = ceil( spectral_norm * spectral_norm / N[n] * K * K / (eps[e] * eps[e]) * log(K/delta) );
			    		long int q = ceil( (V21_norm / N[n])  * (V21_norm / N[n]) * K * K / (eps[e] * eps[e]) * log(K/delta) );
			    		
//				    	cout << "p: " << p << endl;
//				 	cout << "q: " << q << endl;
				    	
					// Sample p vectors
					vector<vector<double>> V_P_sampled(p, vector<double>(d));
				    	for (int i = 0; i < p; i++) {
				    		int j = unif_distribution(generator);
				 		V_P_sampled[i] = V[j];
					}
					
					// Sample q vectors EK_means
					vector<vector<double>> V_Q_EK_means(q, vector<double>(d));
					vector<vector<double>> V_Q_EK_means_normalised(q, vector<double>(d));
					for (int i = 0; i < q; i++) {
						int j = Q_dist(gen);
						V_Q_EK_means[i] = V[j];
						for (int l = 0; l < d; l++)
							V_Q_EK_means_normalised[i][l] = V[j][l] / V_norms[j];
					}
													    	
				    	// Sample initial centroids uniformly from the dataset
				    	vector<vector<double>> C_EK_means(K, vector<double>(d));
				    	C_EK_means = C_initial;
					    		
						
				    	// Main EKK-means loop
				    	distance_centroids = 2 * K * threshold;
				    	iterations = 0.;
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
							    		new_dist += abs((C_EK_means[j][h] - V_P_sampled[i][h]) * (C_EK_means[j][h] - V_P_sampled[i][h]));
						 
								if (new_dist < min_distance) {
							    		min_distance = new_dist;
							    		min_index = j;
								}
						    	}
							C_size[min_index]++;
					    	}
					    	
					    	
					    	// Compute new centroids
					    	vector<vector<double>> C_new_EK_means(K, vector<double>(d, 0));
					    	for (int i = 0; i < q; i++) {
					    	
					    		int min_index = -1;
						    	double min_distance = DBL_MAX;
						    	for (int j = 0; j < K; j++) {
						    	
								double new_dist = 0;
								for (int h = 0; h < d; h++) 
							    		new_dist += abs((C_EK_means[j][h] - V_Q_EK_means[i][h]) * (C_EK_means[j][h] - V_Q_EK_means[i][h]));
						 
								if (new_dist < min_distance) {
							    		min_distance = new_dist;
							    		min_index = j;
								}
						    	}

							for (int h = 0; h < d; h++)
						    		C_new_EK_means[min_index][h] += V_Q_EK_means_normalised[i][h];
					    	}
					    	
					    	for (int j = 0; j < K; j++) {
					    		if (C_size[j] == 0) 
					    			C_new_EK_means[j] = C_EK_means[j];			    
					    		else {
					    			double coeff = (((V21_norm / N[n]) / q) * p) / C_size[j];
					    			for (int h = 0; h < d; h++)
					    				C_new_EK_means[j][h] = coeff * C_new_EK_means[j][h];
					    		}
					    	}
					    	
					    	// Compute distance between centroids
					    	distance_centroids = 0;	
					    	for (int j = 0; j < K; j++) {
					    	
					    		double distance_aux = 0;
							for (int h = 0; h < d; h++) 
						    		distance_aux += abs((C_EK_means[j][h] - C_new_EK_means[j][h]) * (C_EK_means[j][h] - C_new_EK_means[j][h]));
						  	distance_centroids += sqrt(abs(distance_aux));
					    	}
					    	
			//		    	cout << distance_centroids << endl;

					    	C_EK_means = C_new_EK_means;
					    	iterations += 1;					 
					}					
					iterations_EK_means[n][e] += iterations / (n_iter_C * n_iter_N);
					
					// Compute duration
				    	end = chrono::high_resolution_clock::now();
				    	dur = chrono::duration_cast<chrono::milliseconds>(end - begin).count();
				    	time_EK_means[n][e] += dur / (n_iter_N * n_iter_C);
			
			//		cout << "Iterations EKmeans: " << iterations << endl;
					
										
//////////////////////////////////////////////////////////////////////////////////////
				    	
				    	
				    	// Initial time
				    	begin = chrono::high_resolution_clock::now();
					
					p = ceil( spectral_norm * spectral_norm / N[n] * K * K / (eps[e] * eps[e]) * log(K/delta) );
			    		q = ceil( (V21_norm / N[n])  * (V21_norm / N[n]) * K * K / (eps[e] * eps[e]) * log(K/delta) );
			    		
//				    	cout << "p: " << p << endl;
//				 	cout << "q: " << q << endl;
				    	
					// Sample p vectors
					vector<vector<double>> V_P_mini_batch(p, vector<double>(d));
				    	for (int i = 0; i < p; i++) {
				    		int j = unif_distribution(generator);
				 		V_P_mini_batch[i] = V[j];
					}
					
					// Sample q vectors mini_batch
					vector<vector<double>> V_Q_mini_batch(q, vector<double>(d));
					for (int i = 0; i < q; i++) {
						int j = unif_distribution(generator);
				 		V_Q_mini_batch[i] = V[j];
					}
					
					vector<vector<double>> C_mini_batch(K, vector<double>(d));
				    	C_mini_batch = C_initial;
					
					// Main mini batch loop
				    	distance_centroids = 2 * K * threshold;
				    	iterations = 0.;
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
							    		new_dist += abs((C_mini_batch[j][h] - V_P_sampled[i][h]) * (C_mini_batch[j][h] - V_P_sampled[i][h]));
						 
								if (new_dist < min_distance) {
							    		min_distance = new_dist;
							    		min_index = j;
								}
						    	}
							C_size[min_index]++;
					    	}
					    	
					    	
					    	// Compute new centroids
					    	vector<vector<double>> C_new_mini_batch(K, vector<double>(d, 0));
					    	for (int i = 0; i < q; i++) {
					    	
					    		int min_index = -1;
						    	double min_distance = DBL_MAX;
						    	for (int j = 0; j < K; j++) {
						    	
								double new_dist = 0;
								for (int h = 0; h < d; h++) 
							    		new_dist += abs((C_mini_batch[j][h] - V_Q_mini_batch[i][h]) * (C_mini_batch[j][h] - V_Q_mini_batch[i][h]));
						 
								if (new_dist < min_distance) {
							    		min_distance = new_dist;
							    		min_index = j;
								}
						    	}
						    	
						    	for (int h = 0; h < d; h++)
						    		C_new_mini_batch[min_index][h] += V_Q_mini_batch[i][h];						
					    	}
					    	
					    	for (int j = 0; j < K; j++) {
					    		if (C_size[j] == 0) 
					    			C_new_mini_batch[j] = C_mini_batch[j];			    
					    		else {
					    			double coeff = ((double) p) / q / C_size[j];
					    			for (int h = 0; h < d; h++)
					    				C_new_mini_batch[j][h] = coeff * C_new_mini_batch[j][h];
					    		}
					    	}
					    	
					    	// Compute distance between centroids
					    	distance_centroids = 0;	
					    	for (int j = 0; j < K; j++) {
					    	
					    		double distance_aux = 0;
							for (int h = 0; h < d; h++) 
						    		distance_aux += abs((C_mini_batch[j][h] - C_new_mini_batch[j][h]) * (C_mini_batch[j][h] - C_new_mini_batch[j][h]));
						  	distance_centroids += sqrt(abs(distance_aux));
					    	}
					    	
			//		    	cout << distance_centroids << endl;

					    	C_mini_batch = C_new_mini_batch;
					    	iterations += 1;					 
					}					
					iterations_mini_batch[n][e] += iterations / (n_iter_C * n_iter_N);					

				    	// Compute duration
				    	end = chrono::high_resolution_clock::now();
				    	dur = chrono::duration_cast<chrono::milliseconds>(end - begin).count();
				    	time_mini_batch[n][e] += dur / (n_iter_N * n_iter_C);
				    	
			//	    	cout << "Iterations mini batch: " << iterations << endl;


//////////////////////////////////////////////////////////////////////////////////////
				    	
				    	// Compute RSS, ARI, NMI
				    	long double RSS_standard_aux = 0., RSS_EK_means_aux = 0., RSS_mini_batch_aux = 0.;
				    	long double ARI_standard_aux = 0., ARI_EK_means_aux = 0., ARI_mini_batch_aux = 0.;
				    	long double NMI_standard_aux = 0., NMI_EK_means_aux = 0., NMI_mini_batch_aux = 0.;
				    	vector<vector<double>> overlap_matrix_EK_means(K, vector<double>(K, 0));
				    	vector<vector<double>> overlap_matrix_mini_batch(K, vector<double>(K, 0));
				    	
				    	for (int i = 0; i < N[n]; i++) {
				    	
					    	long double min_distance_standard = DBL_MAX, min_distance_EK_means = DBL_MAX, min_distance_mini_batch = DBL_MAX;
					    	int min_index_standard, min_index_EK_means, min_index_mini_batch;
					    	for (int j = 0; j < K; j++) {
					    	
							long double new_dist_standard = 0., new_dist_EK_means = 0., new_dist_mini_batch = 0.;
							for (int h = 0; h < d; h++) {
								new_dist_standard += abs((C_standard[j][h] - V[i][h]) * (C_standard[j][h] - V[i][h]));
						    		new_dist_EK_means += abs((C_EK_means[j][h] - V[i][h]) * (C_EK_means[j][h] - V[i][h]));
						    		new_dist_mini_batch += abs((C_mini_batch[j][h] - V[i][h]) * (C_mini_batch[j][h] - V[i][h]));
						    	}
					 
							if (new_dist_standard < min_distance_standard) {
						    		min_distance_standard = new_dist_standard;
						    		min_index_standard = j;
						    	}
						    	if (new_dist_EK_means < min_distance_EK_means) {
						    		min_distance_EK_means = new_dist_EK_means;
						    		min_index_EK_means = j;
						    	}
						    	if (new_dist_mini_batch < min_distance_mini_batch) {
						    		min_distance_mini_batch = new_dist_mini_batch;
						    		min_index_mini_batch = j;
						    	}
					    	}
					    	RSS_standard_aux += sqrt(abs(min_distance_standard));
					    	RSS_EK_means_aux += sqrt(abs(min_distance_EK_means));
					    	RSS_mini_batch_aux += sqrt(abs(min_distance_mini_batch));
					    	overlap_matrix_EK_means[min_index_standard][min_index_EK_means] += 1;
					    	overlap_matrix_mini_batch[min_index_standard][min_index_mini_batch] += 1;
				    	}
				    	
				    	vector<double> marginal_distribution_standard(K, 0);
				    	vector<double> marginal_distribution_EK_means(K, 0);
				    	vector<double> marginal_distribution_mini_batch(K, 0);
				    	for (int i = 0; i < K; i++) {
				    		for (int j = 0; j < K; j++) {
				    		     	marginal_distribution_standard[i] += overlap_matrix_EK_means[i][j];
				    		     	marginal_distribution_EK_means[i] += overlap_matrix_EK_means[j][i];
				    		     	marginal_distribution_mini_batch[i] += overlap_matrix_mini_batch[j][i];				    		     	
				    		}
				    	}
				    			
				    	RSS_EK_means[n][e] += 100 * (RSS_EK_means_aux - RSS_standard_aux) / RSS_standard_aux / (n_iter_C * n_iter_N);
			//	    	cout << "RSS EK means: " << RSS_EK_means[n][e] << endl;
				    	
				    	RSS_mini_batch[n][e] += 100 * (RSS_mini_batch_aux - RSS_standard_aux) / RSS_standard_aux / (n_iter_C * n_iter_N);
			//	    	cout << "RSS mini batch: " << RSS_mini_batch[n][e] << endl;    			    				    	
				    	
				    	long double Nij_EK_means = 0, Nij_mini_batch = 0, ai = 0, bj_EK_means = 0, bj_mini_batch = 0;
				    	long double MI_EK_means = 0, MI_mini_batch = 0, H_standard = 0, H_EK_means = 0, H_mini_batch = 0;
				    	for (int i = 0; i < K; i++) {
				    		for (int j = 0; j < K; j++) {
				    		
				    		     	Nij_EK_means += overlap_matrix_EK_means[i][j] * (overlap_matrix_EK_means[i][j] - 1)/2;
				    		     	Nij_mini_batch += overlap_matrix_mini_batch[i][j] * (overlap_matrix_mini_batch[i][j] - 1)/2;
				    		     	
				    		     	if (overlap_matrix_EK_means[i][j] > 0)
				    		     		MI_EK_means += overlap_matrix_EK_means[i][j] * log( N[n] * overlap_matrix_EK_means[i][j] / marginal_distribution_standard[i] / marginal_distribution_EK_means[j]);     	
				    		     	if (overlap_matrix_mini_batch[i][j] > 0)
				    		     		MI_mini_batch += overlap_matrix_mini_batch[i][j] * log( N[n] * overlap_matrix_mini_batch[i][j] / marginal_distribution_standard[i] / marginal_distribution_mini_batch[j]);
				    		}
				    		ai += marginal_distribution_standard[i] * (marginal_distribution_standard[i] - 1)/2;
				    		bj_EK_means += marginal_distribution_EK_means[i] * (marginal_distribution_EK_means[i] - 1)/2;
				    		bj_mini_batch += marginal_distribution_mini_batch[i] * (marginal_distribution_mini_batch[i] - 1)/2;
				    		
				    		if (marginal_distribution_standard[i] > 0)
				    			H_standard += marginal_distribution_standard[i] * log( N[n] / marginal_distribution_standard[i]);
				    		if (marginal_distribution_EK_means[i] > 0)
				    			H_EK_means += marginal_distribution_EK_means[i] * log( N[n] / marginal_distribution_EK_means[i]);
				    		if (marginal_distribution_mini_batch[i] > 0)
				    			H_mini_batch += marginal_distribution_mini_batch[i] * log( N[n] / marginal_distribution_mini_batch[i]);
				    	}					   	    	
				    		     
				    	ARI_EK_means[n][e] +=  (Nij_EK_means - 2 * ai * bj_EK_means / N[n] / (N[n]-1) ) / (ai/2 + bj_EK_means/2 - 2 * ai * bj_EK_means / N[n] / (N[n]-1) ) / (n_iter_C * n_iter_N);
				    	ARI_mini_batch[n][e] += (Nij_mini_batch - 2 * ai * bj_mini_batch / N[n] / (N[n]-1) ) / (ai/2 + bj_mini_batch/2 - 2 * ai * bj_mini_batch / N[n] / (N[n]-1) ) / (n_iter_C * n_iter_N);
				    	NMI_EK_means[n][e] += 2 * MI_EK_means / (H_standard + H_EK_means) / (n_iter_C * n_iter_N);				   				    	
				    	NMI_mini_batch[n][e] += 2 * MI_mini_batch / (H_standard + H_mini_batch) / (n_iter_C * n_iter_N);
				    	
			//	    	cout << "ARI EKmeans: " << ARI_EK_means[n][e] << endl;
			//	    	cout << "ARI mini batch: " << ARI_mini_batch[n][e] << endl;
			//	    	cout << "NMI EKmeans: " << NMI_EK_means[n][e] << endl;				 				 
			//	    	cout << "NMI mini batch: " << NMI_mini_batch[n][e] << endl;
				}
				cout << n << ' ' << iter_N << ' ' << iter_C << endl; 						
			}	
		}
    	}
    	
    	// Write the output into a separate file
	ofstream MyFile;
	
	MyFile.open("data_n_EK_means.txt");
	MyFile << "iterations" << ' ' << n_iter_C * n_iter_N << endl;
	
	for (int n = 0; n < length_N; n++) {
		MyFile << N[n] << ' ' << 0.0 << ' ' << time_standard_Kmeans[n] << ' ' << iterations_standard_Kmeans[n] << endl;
		cout << "N: " << N[n] << endl;
		cout << "Milliseconds standard: " << time_standard_Kmeans[n] << endl;
    		cout << "Iterations standard: " << iterations_standard_Kmeans[n] << endl << endl;
    		
		for (int e = 0; e < length_eps; e++) {
			MyFile << N[n] << ' ' << eps[e] << ' ' << time_EK_means[n][e] << ' ' << iterations_EK_means[n][e] << ' ' << RSS_EK_means[n][e] << ' ' << ARI_EK_means[n][e] << ' ' << NMI_EK_means[n][e] << endl;
			cout << "Miliseconds EKmeans: " << time_EK_means[n][e] << endl;
    			cout << "Iterations EKmeans: " << iterations_EK_means[n][e] << endl;
    			cout << "RSS EKmeans (%): " << RSS_EK_means[n][e] << endl;
    			cout << "ARI EKmeans: " << ARI_EK_means[n][e] << endl;
    			cout << "NMI EKmeans: " << NMI_EK_means[n][e] << endl << endl;
    		}
	}
	MyFile.close();
	
	MyFile.open("data_n_mini_batch.txt");
	MyFile << "iterations" << ' ' << n_iter_C * n_iter_N << endl;
	
	for (int n = 0; n < length_N; n++) {
		MyFile << N[n] << ' ' << 0.0 << ' ' << time_standard_Kmeans[n] << ' ' << iterations_standard_Kmeans[n] << endl;
		cout << "N: " << N[n] << endl;
		cout << "Milliseconds standard: " << time_standard_Kmeans[n] << endl;
    		cout << "Iterations standard: " << iterations_standard_Kmeans[n] << endl << endl;
    		
		for (int e = 0; e < length_eps; e++) {
			MyFile << N[n] << ' ' << eps[e] << ' ' << time_mini_batch[n][e] << ' ' << iterations_mini_batch[n][e] << ' ' << RSS_mini_batch[n][e] << ' ' << ARI_mini_batch[n][e] << ' ' << NMI_mini_batch[n][e] << endl;
			cout << "Miliseconds mini batch: " << time_mini_batch[n][e] << endl;
    			cout << "Iterations mini batch: " << iterations_mini_batch[n][e] << endl;
    			cout << "RSS mini batch (%): " << RSS_mini_batch[n][e] << endl;
    			cout << "ARI mini batch: " << ARI_mini_batch[n][e] << endl;
    			cout << "NMI mini batch: " << NMI_mini_batch[n][e] << endl << endl;
    		}
	}
	MyFile.close();
	
	auto t1 = chrono::high_resolution_clock::now();
	cout <<  chrono::duration_cast<chrono::seconds>(t1 - t0).count() << endl;
    	
    	return 0;

}
