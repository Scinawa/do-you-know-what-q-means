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

	int K = 5, N = 300000;
	int length_eps = 5;
    	double eps[length_eps] = {0.2, 0.4, 0.6, 0.8, 1.0};
    	double threshold = 0.1;
    	double delta = 0.01;

	int n_iter_C = 15;
	int n_iter_d = 10;
	int length_d = 6;
	int n_parallel = 4;
	int d[length_d] = {10, 20, 30, 40, 50, 60};
		
    	srand(time(0));
	random_device rd;
	mt19937 gen(rd());
	default_random_engine generator;
    	
    	double time_standard_Kmeans[n_parallel][length_d] = {0};
    	double time_EK_Kmeans[n_parallel][length_d][length_eps] = {0};
    	
    	double iterations_standard_Kmeans[n_parallel][length_d] = {0};
    	double iterations_EK_Kmeans[n_parallel][length_d][length_eps] = {0};
    	   	
    	double RSS[n_parallel][length_d][length_eps] = {0};
    	double ARI[n_parallel][length_d][length_eps] = {0};
    	double NMI[n_parallel][length_d][length_eps] = {0};
    	
    	#pragma omp parallel for
    	for (int parallel = 0; parallel < n_parallel; parallel++) {
    	
    		uniform_int_distribution<int> unif_distribution(0, N-1);
    		
	    	for (int dim = 0; dim < length_d; dim++) {
	  		
	  		for (int iter_d = 0; iter_d < n_iter_d; iter_d++) {
	    	
		    		vector<vector<double>> C_aux(K, vector<double>(d[dim]));
		    		vector<vector<double>> V(N, vector<double>(d[dim]));
		    		mat V_aux(N, d[dim]);
		  
		  		// Create the input
		    		for (int j = 0; j < K; j++) {
		    			for (int h = 0; h < d[dim]; h++) 
		    				C_aux[j][h] = 2*((double)rand()) / RAND_MAX - 1;
		    		}
		    		
		    		for (int j = 0; j < K; j++) {
		    			for (int l = 0; l < N/K; l++) {
		    				for (int h = 0 ; h < d[dim]; h++) {
		    					V[j * N / K + l][h] = C_aux[j][h] + 2* ((double)rand()) / RAND_MAX - 1;
		    					V_aux(j * N / K + l, h) = V[j * N / K + l][h];
		    				}
		    			}
		    		}
		    		
		    		// Compute the norms
		    		double spectral_norm = norm2est(V_aux);
		    		vector<double> V_norms(N, 0);
		    		double V21_norm = 0;
		    		for (int i = 0; i < N; i++) {
		    			for (int h = 0; h < d[dim]; h++)
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
				    	vector<vector<double>> C_initial(K, vector<double>(d[dim]));
				    	for (int i = 0; i < K; i++) {
				    		int j = unif_distribution(generator);
						C_initial[i] = V[j];
					}
								    		
	 //////////////////////////////////////////////////////////////////////////////////////
			    		
			    		auto begin = chrono::high_resolution_clock::now();
					
					// Sample initial centroids uniformly from the dataset
					vector<vector<double>> C_standard(K, vector<double>(d[dim]));
					C_standard = C_initial;		
					
					// Main standard k-means loop
					double distance_centroids = 2 * K * threshold;
					double iterations = 0.;
					while (distance_centroids > K * threshold) {
						
					    	// Compute cluster sizes
					    	vector<int> C_size(K, 0);
					    	for (int i = 0; i < N; i++) {
					    	
							int min_index = -1;
						    	double min_distance = DBL_MAX;
						    	for (int j = 0; j < K; j++) {
						    	
								double new_dist = 0;
								for (int h = 0; h < d[dim]; h++) 
							    		new_dist += abs((C_standard[j][h] - V[i][h]) * (C_standard[j][h] - V[i][h]));
						 
								if (new_dist < min_distance) {
							    		min_distance = new_dist;
							    		min_index = j;
								}
						    	}
							C_size[min_index]++;
					    	}
					    	
					    	// Compute new centroids
					    	vector<vector<double>> C_new(K, vector<double>(d[dim],0));
					    	for (int i = 0; i < N; i++) {
					    	
					    		int min_index = -1;
						    	double min_distance = DBL_MAX;
						    	for (int j = 0; j < K; j++) {
						    	
								double new_dist = 0;
								for (int h = 0; h < d[dim]; h++) 
							    		new_dist += abs((C_standard[j][h] - V[i][h]) * (C_standard[j][h] - V[i][h]));
						 
								if (new_dist < min_distance) {
							    		min_distance = new_dist;
							    		min_index = j;
								}
						    	}

							for (int h = 0; h < d[dim]; h++)  
						    		C_new[min_index][h] += V[i][h];
					    	}
					    	
					    	for (int j = 0; j < K; j++) {
					    		if (C_size[j] == 0) {
					    			for (int h = 0; h < d[dim]; h++)
					    				C_new[j][h] = 0;
					    		}
					    		else {
					    			for (int h = 0; h < d[dim]; h++)
					    				C_new[j][h] = C_new[j][h] / C_size[j];
					    		}
					    	}
					    	
					    	// Compute distance between centroids
					    	distance_centroids = 0;	
					    	for (int j = 0; j < K; j++) {
					    	
					    		double distance_aux = 0;
							for (int h = 0; h < d[dim]; h++) 
						    		distance_aux += abs((C_standard[j][h] - C_new[j][h]) * (C_standard[j][h] - C_new[j][h]));
						  	distance_centroids += sqrt(abs(distance_aux));
					    	}
					  				    	
					    	C_standard = C_new;
					    	iterations += 1;
					}
					iterations_standard_Kmeans[parallel][dim] += iterations / (n_iter_C * n_iter_d);

				    	// Compute duration
				    	auto end = chrono::high_resolution_clock::now();
				    	double dur = chrono::duration_cast<chrono::milliseconds>(end - begin).count();
				    	time_standard_Kmeans[parallel][dim] += dur / (n_iter_d * n_iter_C);
				//    	cout << "Iterations: " << iterations << endl;
				    				    				    				    		
//////////////////////////////////////////////////////////////////////////////////////			   			
					
					for (int e = 0; e < length_eps; e++) {
					
					    	// Initial time
					    	begin = chrono::high_resolution_clock::now();
					
				    		long int p = ceil( spectral_norm * spectral_norm / N * K * K / (eps[e] * eps[e]) * log(K/delta) );
				    		long int q = ceil( (V21_norm / N)  * (V21_norm / N) * K * K / (eps[e] * eps[e]) * log(K/delta) );
				    		
				 //   		cout << "p: " << p << endl;
				 //		cout << "q: " << q << endl;
					    	
						// Sample p vectors
						vector<vector<double>> V_P_sampled(p, vector<double>(d[dim]));
					    	for (int i = 0; i < p; i++) {
					    		int j = unif_distribution(generator);
					 		V_P_sampled[i] = V[j];
						}
						
						// Sample q vectors
						vector<vector<double>> V_Q_sampled(q, vector<double>(d[dim]));
						vector<vector<double>> V_Q_sampled_normalised(q, vector<double>(d[dim]));
						for (int i = 0; i < q; i++) {
							int j = Q_dist(gen);
							V_Q_sampled[i] = V[j];
							for (int l = 0; l < d[dim]; l++)
								V_Q_sampled_normalised[i][l] = V[j][l] / V_norms[j];
						}
					    	
					    	// Sample initial centroids uniformly from the dataset
					    	vector<vector<double>> C_eps(K, vector<double>(d[dim]));
				    		C_eps = C_initial;
						    		
							
					    	// Main EKK-means loop
					    	distance_centroids = 2 * K * threshold;
					    	iterations = 0;
						while (distance_centroids > K * threshold) {
						
						    	// Compute cluster sizes
						    	vector<int> C_size(K, 0);
						    	for (int i = 0; i < p; i++) {
						    	
								int min_index = -1;
							    	double min_distance = DBL_MAX;
							    	for (int j = 0; j < K; j++) {
							    	
									double new_dist = 0;
									for (int h = 0; h < d[dim]; h++) 
								    		new_dist += abs((C_eps[j][h] - V_P_sampled[i][h]) * (C_eps[j][h] - V_P_sampled[i][h]));
							 
									if (new_dist < min_distance) {
								    		min_distance = new_dist;
								    		min_index = j;
									}
							    	}
								C_size[min_index]++;
						    	}
						    	
						    	
						    	// Compute new centroids
						    	vector<vector<double>> C_new(K, vector<double>(d[dim],0));
						    	for (int i = 0; i < q; i++) {
						    	
						    		int min_index = -1;
							    	double min_distance = DBL_MAX;
							    	for (int j = 0; j < K; j++) {
							    	
									double new_dist = 0;
									for (int h = 0; h < d[dim]; h++) 
								    		new_dist += abs((C_eps[j][h] - V_Q_sampled[i][h]) * (C_eps[j][h] - V_Q_sampled[i][h]));
							 
									if (new_dist < min_distance) {
								    		min_distance = new_dist;
								    		min_index = j;
									}
							    	}

								for (int h = 0; h < d[dim]; h++)
							    		C_new[min_index][h] += V_Q_sampled_normalised[i][h];
						    	}
						    	
						    	for (int j = 0; j < K; j++) {
						    		if (C_size[j] == 0) 
						    			C_new[j] = C_eps[j];			    
						    		else {
						    			double coeff = (((V21_norm / N) / q) * p) / C_size[j];
						    			for (int h = 0; h < d[dim]; h++)
						    				C_new[j][h] = coeff * C_new[j][h];
						    		}
						    	}
						    	
						    	// Compute distance between centroids
						    	distance_centroids = 0;	
						    	for (int j = 0; j < K; j++) {
						    	
						    		double distance_aux = 0;
								for (int h = 0; h < d[dim]; h++) 
							    		distance_aux += abs((C_eps[j][h] - C_new[j][h]) * (C_eps[j][h] - C_new[j][h]));
							  	distance_centroids += sqrt(abs(distance_aux));
						    	}

						    	C_eps = C_new;
						    	iterations += 1;					 
						}					
						iterations_EK_Kmeans[parallel][dim][e] += iterations / (n_iter_C * n_iter_d);

					    	// Compute duration
					    	end = chrono::high_resolution_clock::now();
					    	dur = chrono::duration_cast<chrono::milliseconds>(end - begin).count();
					    	time_EK_Kmeans[parallel][dim][e] += dur / (n_iter_d * n_iter_C);
					    	
					  //  	cout << "Iterations: " << iterations << endl;
					    	
					    	// Compute RSS, ARI, NMI
					    	long double RSS_standard = 0., RSS_eps = 0.;
					    	long double ARI_standard = 0., ARI_eps = 0.;
					    	long double NMI_standard = 0., NMI_eps = 0.;
					    	vector<vector<double>> overlap_matrix(K, vector<double>(K, 0));
					    	
					    	for (int i = 0; i < N; i++) {
					    	
						    	long double min_distance_standard = DBL_MAX, min_distance_eps = DBL_MAX;
						    	int min_index_standard, min_index_eps;
						    	for (int j = 0; j < K; j++) {
						    	
								long double new_dist_standard = 0., new_dist_eps = 0.;
								for (int h = 0; h < d[dim]; h++) {
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
					    	
					    	vector<double> marginal_distribution_standard(K, 0);
					    	vector<double> marginal_distribution_eps(K, 0);
					    	for (int i = 0; i < K; i++) {
					    		for (int j = 0; j < K; j++) {
					    		     	marginal_distribution_standard[i] += overlap_matrix[i][j];
					    		     	marginal_distribution_eps[i] += overlap_matrix[j][i];					    		     	
					    		}
					    	}
					    			
					    	RSS[parallel][dim][e] += 100 * (RSS_eps - RSS_standard) / RSS_standard / (n_iter_C * n_iter_d);
					 //   	cout << "RSS: " << RSS[parallel][dim][e] << endl;
					    			    				    	
					    	
					    	long double Nij = 0, ai = 0, bj = 0;
					    	long double MI = 0, H_standard = 0, H_eps = 0;
					    	for (int i = 0; i < K; i++) {
					    		for (int j = 0; j < K; j++) {
					    		
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
					    		     
					    	ARI[parallel][dim][e] += (Nij - 2 * ai * bj / N / (N-1) ) / (ai/2 + bj/2 - 2 * ai * bj / N / (N-1) ) / (n_iter_C * n_iter_d);
					    	NMI[parallel][dim][e] += 2 * MI / (H_standard + H_eps) / (n_iter_C * n_iter_d);
					//    	cout << "ARI: " << ARI[parallel][dim][e] << endl;
					//    	cout << "NMI: " << NMI[parallel][dim][e] << endl;
					}
					cout << dim << ' ' << iter_d << ' ' << iter_C << endl;											
				}
			}
	    	}
	}

    	// Write the output onto a separate file    	
	ofstream MyFile("data_d.txt");
	MyFile << "iterations" << ' ' << n_parallel * n_iter_C * n_iter_d << endl;
	
	for (int dim = 0; dim < length_d; dim++) {
		double time = 0;
		double iterations = 0;
		for (int parallel = 0; parallel < n_parallel; parallel++) {
			time += time_standard_Kmeans[parallel][dim] / n_parallel;
			iterations += iterations_standard_Kmeans[parallel][dim] / n_parallel;
		}
		MyFile << d[dim] << ' ' << 0.0 << ' ' << time << ' ' << iterations << endl;
		cout << "d: " << d[dim] << endl;
		cout << "Milliseconds standard: " << time << endl;
    		cout << "Iterations standard: " << iterations << endl << endl;
    		
		for (int e = 0; e < length_eps; e++) {
			time = 0;
			iterations = 0;
	    		double RSS_aux = 0, ARI_aux = 0, NMI_aux = 0;
	    		for (int parallel = 0; parallel < n_parallel; parallel++) {
				time += time_EK_Kmeans[parallel][dim][e] / n_parallel;
				iterations += iterations_EK_Kmeans[parallel][dim][e] / n_parallel;
				RSS_aux += RSS[parallel][dim][e] / n_parallel;
				ARI_aux += ARI[parallel][dim][e] / n_parallel;
				NMI_aux += NMI[parallel][dim][e] / n_parallel;
			}
			MyFile << d[dim] << ' ' << eps[e] << ' ' << time << ' ' << iterations << ' ' << RSS_aux << ' ' << ARI_aux << ' ' << NMI_aux << endl;
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
