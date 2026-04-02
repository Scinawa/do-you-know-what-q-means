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

	int d = 30, N = 300000;
	int length_eps = 5;
    	double eps[length_eps] = {0.2, 0.4, 0.6, 0.8, 1.0};
    	double threshold = 0.1;
    	double delta = 0.01;

	int n_iter_C = 20;
	int n_iter_K = 10;
	int length_K = 6;
	int n_parallel = 4;
	int K[length_K] = {2, 4, 6, 8, 10, 12};
		
    	srand(time(0));
	random_device rd;
	mt19937 gen(rd());
	default_random_engine generator;
    	
    	double time_standard_Kmeans[n_parallel][length_K] = {0};
    	double time_EK_Kmeans[n_parallel][length_K][length_eps] = {0};
    	
    	double iterations_standard_Kmeans[n_parallel][length_K] = {0};
    	double iterations_EK_Kmeans[n_parallel][length_K][length_eps] = {0};
    	   	
    	double RSS[n_parallel][length_K][length_eps] = {0};
    	double ARI[n_parallel][length_K][length_eps] = {0};
    	double NMI[n_parallel][length_K][length_eps] = {0};
    	
    	#pragma omp parallel for
    	for (int parallel = 0; parallel < n_parallel; parallel++) {
    	
    		uniform_int_distribution<int> unif_distribution(0, N-1);
    		
	    	for (int k = 0; k < length_K; k++) {
	  		
	  		for (int iter_K = 0; iter_K < n_iter_K; iter_K++) {
	    	
		    		vector<vector<double>> C_aux(K[k], vector<double>(d));
		    		vector<vector<double>> V(N, vector<double>(d));
		    		mat V_aux(N, d);
		  
		  		// Create the input
		    		for (int j = 0; j < K[k]; j++) {
		    			for (int h = 0; h < d; h++) 
		    				C_aux[j][h] = 2*((double)rand()) / RAND_MAX - 1;
		    		}
		    		
		    		for (int j = 0; j < K[k]; j++) {
		    			for (int l = 0; l < N/K[k]; l++) {
		    				for (int h = 0 ; h < d; h++) {
		    					V[j * N / K[k] + l][h] = C_aux[j][h] + 2* ((double)rand()) / RAND_MAX - 1;
		    					V_aux(j * N / K[k] + l, h) = V[j * N / K[k] + l][h];
		    				}
		    			}
		    		}
		    		
		    		// Compute the norms
		    		double spectral_norm = norm2est(V_aux);
		    		vector<double> V_norms(N, 0);
		    		double V21_norm = 0;
		    		for (int i = 0; i < N; i++) {
		    			for (int h = 0; h < d; h++)
		    				V_norms[i] += V[i][h] * V[i][h];

		    			V_norms[i] = sqrt(V_norms[i]);
		    			V21_norm += V_norms[i];
		    		}
		    		
		    		// Create the distribution for Q
			    	vector<double> Q_distr(N);
			    	for (int i = 0; i < N; i++)
					Q_distr[i] = V_norms[i] / V21_norm;
					
				discrete_distribution<> Q_dist(Q_distr.begin(), Q_distr.end());
		    		
		    		
		    		// Repeating several initial centroids
		    		for (int iter_C = 0; iter_C < n_iter_C; iter_C++) { 
		    		
			    		// Sample initial centroids uniformly from the dataset
				    	vector<vector<double>> C_initial(K[k], vector<double>(d));
				    	for (int i = 0; i < K[k]; i++) {
				    		int j = unif_distribution(generator);
						C_initial[i] = V[j];
					}
								    		
	 //////////////////////////////////////////////////////////////////////////////////////
			    		
			    		auto begin = chrono::high_resolution_clock::now();
					
					// Sample initial centroids uniformly from the dataset
					vector<vector<double>> C_standard(K[k], vector<double>(d));
					C_standard = C_initial;		
					
					// Main standard k-means loop
					double distance_centroids = 2 * K[k] * threshold;
					double iterations = 0;
					while (distance_centroids > K[k] * threshold) {
						
					    	// Compute cluster sizes
					    	vector<int> C_size(K[k], 0);
					    	for (int i = 0; i < N; i++) {
					    	
							int min_index = -1;
						    	double min_distance = DBL_MAX;
						    	for (int j = 0; j < K[k]; j++) {
						    	
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
					    	vector<vector<double>> C_new(K[k], vector<double>(d,0));
					    	for (int i = 0; i < N; i++) {
					    	
					    		int min_index = -1;
						    	double min_distance = DBL_MAX;
						    	for (int j = 0; j < K[k]; j++) {
						    	
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
					    	
					    	for (int j = 0; j < K[k]; j++) {
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
					    	for (int j = 0; j < K[k]; j++) {
					    	
					    		double distance_aux = 0;
							for (int h = 0; h < d; h++) 
						    		distance_aux += abs((C_standard[j][h] - C_new[j][h]) * (C_standard[j][h] - C_new[j][h]));
						  	distance_centroids += sqrt(abs(distance_aux));
					    	}
					  				    	
					    	C_standard = C_new;
					    	iterations += 1;
					}
					iterations_standard_Kmeans[parallel][k] += iterations / (n_iter_C * n_iter_K);

				    	// Compute duration
				    	auto end = chrono::high_resolution_clock::now();
				    	double dur = chrono::duration_cast<chrono::milliseconds>(end - begin).count();
				    	time_standard_Kmeans[parallel][k] += dur / (n_iter_K * n_iter_C);
			//	    	cout << "Iterations: " << iterations << endl;
				    		    		
//////////////////////////////////////////////////////////////////////////////////////			   			
					
					for (int e = 0; e < length_eps; e++) {
					
					    	// Initial time
					    	begin = chrono::high_resolution_clock::now();
					
				    		long int p = ceil( spectral_norm * spectral_norm / N * K[k] * K[k] / (eps[e] * eps[e]) * log(K[k]/delta) );
				    		long int q = ceil( (V21_norm / N)  * (V21_norm / N) * K[k] * K[k] / (eps[e] * eps[e]) * log(K[k]/delta) );
				    		
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
					    	
					    	// Sample initial centroids uniformly from the dataset
					    	vector<vector<double>> C_eps(K[k], vector<double>(d));
						C_eps = C_initial;	
						    		
							
					    	// Main EKK-means loop
					    	distance_centroids = 2 * K[k] * threshold;
					    	iterations = 0;
						while (distance_centroids > K[k] * threshold and iterations < 20) {
						
						    	// Compute cluster sizes
						    	vector<int> C_size(K[k], 0);
						    	for (int i = 0; i < p; i++) {
						    	
								int min_index = -1;
							    	double min_distance = DBL_MAX;
							    	for (int j = 0; j < K[k]; j++) {
							    	
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
						    	vector<vector<double>> C_new(K[k], vector<double>(d,0));
						    	for (int i = 0; i < q; i++) {
						    	
						    		int min_index = -1;
							    	double min_distance = DBL_MAX;
							    	for (int j = 0; j < K[k]; j++) {
							    	
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
						    	
						    	for (int j = 0; j < K[k]; j++) {
						    		if (C_size[j] == 0) 
						    			C_new[j] = C_eps[j];			    
						    		else {
						    			double coeff = (((V21_norm / N) / q) * p) / C_size[j];
						    			for (int h = 0; h < d; h++)
						    				C_new[j][h] = coeff * C_new[j][h];
						    		}
						    	}
						    	
						    	// Compute distance between centroids
						    	distance_centroids = 0;	
						    	for (int j = 0; j < K[k]; j++) {
						    	
						    		double distance_aux = 0;
								for (int h = 0; h < d; h++) 
							    		distance_aux += abs((C_eps[j][h] - C_new[j][h]) * (C_eps[j][h] - C_new[j][h]));
							  	distance_centroids += sqrt(abs(distance_aux));
						    	}

						    	C_eps = C_new;
						    	iterations += 1;					 
						}					
						iterations_EK_Kmeans[parallel][k][e] += iterations / (n_iter_C * n_iter_K);

					    	// Compute duration
					    	end = chrono::high_resolution_clock::now();
					    	dur = chrono::duration_cast<chrono::milliseconds>(end - begin).count();
					    	time_EK_Kmeans[parallel][k][e] += dur / (n_iter_K * n_iter_C);
					    	
				//	    	cout << "Iterations: " << iterations << endl;
					    	
					    	// Compute RSS, ARI, NMI
					    	long double RSS_standard = 0., RSS_eps = 0.;
					    	long double ARI_standard = 0., ARI_eps = 0.;
					    	long double NMI_standard = 0., NMI_eps = 0.;
					    	vector<vector<double>> overlap_matrix(K[k], vector<double>(K[k], 0));
					    	
					    	for (int i = 0; i < N; i++) {
					    	
						    	long double min_distance_standard = DBL_MAX, min_distance_eps = DBL_MAX;
						    	int min_index_standard, min_index_eps;
						    	for (int j = 0; j < K[k]; j++) {
						    	
								long double new_dist_standard = 0., new_dist_eps = 0.;
								for (int h = 0; h < d; h++) {
									new_dist_standard += abs((C_standard[j][h] - V[i][h]) * (C_standard[j][h] - V[i][h]));
							    		new_dist_eps += abs((C_eps[j][h] - V[i][h]) * (C_eps[j][h] - V[i][h]));
							    	}
						 
								if (new_dist_standard < min_distance_standard) {
							    		min_distance_standard = new_dist_standard;
							    		min_index_standard = j;
							    	}
							    	if (new_dist_eps < min_distance_eps) {
							    		min_distance_eps = new_dist_eps;
							    		min_index_eps = j;
							    	}
						    	}
						    	RSS_standard += sqrt(abs(min_distance_standard));
						    	RSS_eps += sqrt(abs(min_distance_eps));
						    	overlap_matrix[min_index_standard][min_index_eps] += 1;
					    	}
					    	
					    	vector<double> marginal_distribution_standard(K[k], 0);
					    	vector<double> marginal_distribution_eps(K[k], 0);
					    	for (int i = 0; i < K[k]; i++) {
					    		for (int j = 0; j < K[k]; j++) {
					    		     	marginal_distribution_standard[i] += overlap_matrix[i][j];
					    		     	marginal_distribution_eps[i] += overlap_matrix[j][i];					    		     	
					    		}
					    	}
					    			
					    	RSS[parallel][k][e] += 100 * (RSS_eps - RSS_standard) / RSS_standard / (n_iter_C * n_iter_K);
				//	    	cout << "RSS: " << RSS[parallel][k][e] << endl;
					    			    				    	
					    	
					    	long double Nij = 0, ai = 0, bj = 0;
					    	long double MI = 0, H_standard = 0, H_eps = 0;
					    	for (int i = 0; i < K[k]; i++) {
					    		for (int j = 0; j < K[k]; j++) {
					    		
					    		     	Nij += overlap_matrix[i][j] * (overlap_matrix[i][j] - 1)/2;
					    		     	
					    		     	if (overlap_matrix[i][j] > 0)
					    		     		MI += overlap_matrix[i][j] * log( N * overlap_matrix[i][j] / marginal_distribution_standard[i] / marginal_distribution_eps[j]);     	
					    		}
					    		ai += marginal_distribution_standard[i] * (marginal_distribution_standard[i] - 1)/2;
					    		bj += marginal_distribution_eps[i] * (marginal_distribution_eps[i] - 1)/2;
					    		
					    		if (marginal_distribution_standard[i] > 0)
					    			H_standard += marginal_distribution_standard[i] * log( N / marginal_distribution_standard[i]);
					    		if (marginal_distribution_eps[i] > 0)
					    			H_eps += marginal_distribution_eps[i] * log( N / marginal_distribution_eps[i]);
					    	}					   	    	
					    		     
					    	ARI[parallel][k][e] += (Nij - 2 * ai * bj / N / (N-1) ) / (ai/2 + bj/2 - 2 * ai * bj / N / (N-1) ) / (n_iter_C * n_iter_K);
					    	NMI[parallel][k][e] += 2 * MI / (H_standard + H_eps) / (n_iter_C * n_iter_K);
					//    	cout << "ARI: " << ARI[parallel][k][e] << endl;
					//    	cout << "NMI: " << NMI[parallel][k][e] << endl;
					}
					cout << k << ' ' << iter_K << ' ' << iter_C << endl;											
				}
			}
	    	}
	}
 
     	// Write the output onto a separate file   	
	ofstream MyFile("data_k.txt");
	MyFile << "iterations" << ' ' << n_parallel * n_iter_C * n_iter_K << endl;
	
	for (int k = 0; k < length_K; k++) {
		double time = 0;
		double iterations = 0;
		for (int parallel = 0; parallel < n_parallel; parallel++) {
			time += time_standard_Kmeans[parallel][k] / n_parallel;
			iterations += iterations_standard_Kmeans[parallel][k] / n_parallel;
		}
		MyFile << K[k] << ' ' << 0.0 << ' ' << time << ' ' << iterations << endl;
		cout << "K: " << K[k] << endl;
		cout << "Milliseconds standard: " << time << endl;
    		cout << "Iterations standard: " << iterations << endl << endl;
    		
		for (int e = 0; e < length_eps; e++) {
			time = 0;
			iterations = 0;
	    		double RSS_aux = 0, ARI_aux = 0, NMI_aux = 0;
	    		for (int parallel = 0; parallel < n_parallel; parallel++) {
				time += time_EK_Kmeans[parallel][k][e] / n_parallel;
				iterations += iterations_EK_Kmeans[parallel][k][e] / n_parallel;
				RSS_aux += RSS[parallel][k][e] / n_parallel;
				ARI_aux += ARI[parallel][k][e] / n_parallel;
				NMI_aux += NMI[parallel][k][e] / n_parallel;
			}
			MyFile << K[k] << ' ' << eps[e] << ' ' << time << ' ' << iterations << ' ' << RSS_aux << ' ' << ARI_aux << ' ' << NMI_aux << endl;
			cout << "Miliseconds EKK: " << time << endl;
    			cout << "Iterations EKK: " << iterations << endl;
    			cout << "RSS EKK (%): " << RSS_aux << endl;
    			cout << "ARI EKK: " << ARI_aux << endl;
    			cout << "NMI EKK: " << NMI_aux << endl << endl;
    		}
	}
	MyFile.close();
	
	auto t1 = chrono::high_resolution_clock::now();
	cout <<  chrono::duration_cast<chrono::seconds>(t1 - t0).count() << endl;
    	
    	return 0;

}
