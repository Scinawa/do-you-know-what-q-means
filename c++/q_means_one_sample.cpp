#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <cfloat>
#include <random>
#include <chrono>
#include <algorithm>
#include <armadillo>

using namespace std;
using namespace arma;

// Function to read binary MNIST data file
bool read_mnist_binary(const char* filename, vector<vector<double>>& V_full, int& d, int& N_total) {
    ifstream file(filename, ios::binary);
    if (!file.is_open()) {
        cerr << "Error: Cannot open file " << filename << endl;
        return false;
    }
    
    // Read header: N (int), d (int)
    int N, d_read;
    file.read(reinterpret_cast<char*>(&N), sizeof(int));
    file.read(reinterpret_cast<char*>(&d_read), sizeof(int));
    
    if (file.fail() || N <= 0 || d_read <= 0) {
        cerr << "Error: Invalid file format or corrupted header" << endl;
        file.close();
        return false;
    }
    
    // Allocate space for full dataset
    V_full.resize(N, vector<double>(d_read));
    
    // Read data: N * d doubles in row-major order
    for (int i = 0; i < N; i++) {
        file.read(reinterpret_cast<char*>(V_full[i].data()), d_read * sizeof(double));
        if (file.fail()) {
            cerr << "Error: Failed to read data at sample " << i << endl;
            file.close();
            return false;
        }
    }
    
    file.close();
    d = d_read;
    N_total = N;
    
    cout << "Successfully loaded " << N << " samples with " << d << " dimensions from " << filename << endl;
    return true;
}

int main (int argc, char* argv[]) {

	auto t0 = chrono::high_resolution_clock::now();

	// Parse command-line arguments
	const char* data_file = nullptr;
	for (int i = 1; i < argc; i++) {
		if ((strcmp(argv[i], "--data-file") == 0 || strcmp(argv[i], "-f") == 0) && i + 1 < argc) {
			data_file = argv[i + 1];
			i++; // Skip next argument as it's the filename
		} else if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
			cout << "Usage: " << argv[0] << " [--data-file|-f <filename>]" << endl;
			cout << "  --data-file, -f: Path to binary MNIST data file" << endl;
			cout << "  --help, -h: Show this help message" << endl;
			cout << "If no data file is specified, synthetic data will be generated." << endl;
			return 0;
		}
	}

	// ========================================================================
	// DATA SOURCE CONFIGURATION: Separate initialization for file vs synthetic
	// ========================================================================
	int d, K;
	int N_total = 0; // Total samples in loaded dataset (only used for file data)
	vector<vector<double>> V_full; // Full dataset if loaded from file
	bool use_file_data = false;
	
	if (data_file != nullptr) {
		// ====================================================================
		// CASE 1: Load data from binary file (e.g., MNIST)
		// ====================================================================
		cout << "=== Loading data from file ===" << endl;
		if (read_mnist_binary(data_file, V_full, d, N_total)) {
			use_file_data = true;
			// For MNIST, typically use K=10 (for 10 digit classes)
			// Adjust K based on your needs
			K = 10;
			cout << "Using file data: d=" << d << ", K=" << K << ", N_total=" << N_total << endl;
		} else {
			cerr << "Error: Failed to load data file. Exiting." << endl;
			return 1;
		}
	} else {
		// ====================================================================
		// CASE 2: Generate synthetic data (original behavior)
		// ====================================================================
		cout << "=== Generating synthetic data ===" << endl;
		d = 50;
		K = 5;
		cout << "Using synthetic data: d=" << d << ", K=" << K << endl;
	}
	
	// Constants for experiment configuration
	const int length_eps = 5;
    double eps[length_eps] = {0.2, 0.4, 0.6, 0.8, 1.0};
    double threshold = 0.1;
    double delta = 0.01;

	int n_iter_C = 20;
	int n_iter_N = 10;
	const int length_N = 9;
	int N[length_N] = {100000, 150000, 200000, 250000, 300000, 350000, 400000, 450000, 500000};
		
    srand(time(0));
	random_device rd;
	mt19937 gen(rd());
	default_random_engine generator;
    	
    vector<double> time_standard_Kmeans(length_N, 0);
    vector<vector<double>> time_EK_Kmeans(length_N, vector<double>(length_eps, 0));
    	
    vector<double> iterations_standard_Kmeans(length_N, 0);
    vector<vector<double>> iterations_EK_Kmeans(length_N, vector<double>(length_eps, 0));
    	   	
    vector<vector<double>> RSS_ratio(length_N, vector<double>(length_eps, 0));
    vector<double> preprocessing_time_ms(length_N, 0);
    volatile double preprocessing_guard = 0.0;
    	
    for (int n = 0; n < length_N; n++) {
    	
  		uniform_int_distribution<int> unif_distribution(0, N[n]-1);
  		
  		for (int iter_N = 0; iter_N < n_iter_N; iter_N++) {
    	
	    	vector<vector<double>> C_aux(K, vector<double>(d));
	    	vector<vector<double>> V(N[n], vector<double>(d));
	    	mat V_aux(N[n], d);
	  
	  		// ====================================================================
	  		// DATA GENERATION: Two distinct paths based on data source
	  		// ====================================================================
	    	if (use_file_data) {
	    		// --------------------------------------------------------------------
	    		// PATH 1: Sample from loaded file data (e.g., MNIST)
	    		// --------------------------------------------------------------------
	    		uniform_int_distribution<int> file_distribution(0, N_total - 1);
	    		for (int i = 0; i < N[n]; i++) {
	    			int idx = file_distribution(generator);
	    			V[i] = V_full[idx];
	    			for (int h = 0; h < d; h++) {
	    				V_aux(i, h) = V[i][h];
	    			}
	    		}
	    		// C_aux not needed when using file data, but initialize for consistency
	    		for (int j = 0; j < K; j++) {
	    			for (int h = 0; h < d; h++) 
	    				C_aux[j][h] = 0; // Will be overwritten by initial centroids
	    		}
	    	} else {
	    		// --------------------------------------------------------------------
	    		// PATH 2: Generate synthetic data (original behavior)
	    		// --------------------------------------------------------------------
	    		// Generate K cluster centers
	    		for (int j = 0; j < K; j++) {
	    			for (int h = 0; h < d; h++) 
	    				C_aux[j][h] = 2*((double)rand()) / RAND_MAX - 1;
	    		}
	    		
	    		// Generate N[n] points around the K cluster centers
	    		for (int j = 0; j < K; j++) {
	    			for (int l = 0; l < N[n]/K; l++) {
	    				for (int h = 0 ; h < d; h++) {
	    					V[j * N[n] / K + l][h] = C_aux[j][h] + 2* ((double)rand()) / RAND_MAX - 1;
	    					V_aux(j * N[n] / K + l, h) = V[j * N[n] / K + l][h];
	    				}
	    			}
	    		}
	    	}
	    		
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

		    auto end = chrono::high_resolution_clock::now();
		    double dur = chrono::duration_cast<chrono::milliseconds>(end - begin).count();
		    preprocessing_time_ms[n] += dur / n_iter_N;

		    // Keep the measured preprocessing work alive even under optimization.
		    int sample_index = Q_dist(gen);
		    preprocessing_guard += spectral_norm + V21_norm + Q_distr[sample_index];
		    continue;

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
				vector<vector<double>> C(K, vector<double>(d));
				C = C_initial;		
				
				// Main standard k-means loop
				double distance_centroids = 2 * K * threshold;
				int iterations = 0;
				while (distance_centroids > K*threshold) {
					
				    // Compute cluster sizes
				    vector<int> C_size(K, 0);
				    for (int i = 0; i < N[n]; i++) {
				    	
						int min_index = -1;
					    double min_distance = DBL_MAX;
					    for (int j = 0; j < K; j++) {
					    	
							double new_dist = 0;
							for (int h = 0; h < d; h++) 
						    	new_dist += (C[j][h] - V[i][h]) * (C[j][h] - V[i][h]);
					 
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
						    	new_dist += (C[j][h] - V[i][h]) * (C[j][h] - V[i][h]);
					 
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
					    	distance_aux += (C[j][h] - C_new[j][h]) * (C[j][h] - C_new[j][h]);
					  	distance_centroids += sqrt(abs(distance_aux));
				    }
				  				    	
				    C = C_new;
				    iterations++;
				}
				iterations_standard_Kmeans[n] += (double (iterations)) / (n_iter_C * n_iter_N);

			    // Compute duration
			    auto end = chrono::high_resolution_clock::now();
			    double dur = chrono::duration_cast<chrono::milliseconds>(end - begin).count();
			    time_standard_Kmeans[n] += dur / (n_iter_N * n_iter_C);
			    	
			    // Computes RSS
			    double RSS_standard = 0.;
			    for (int i = 0; i < N[n]; i++) {
			    	
				    double min_distance = DBL_MAX;
				    for (int j = 0; j < K; j++) {
				    	
						double new_dist = 0;
						for (int h = 0; h < d; h++) 
					    		new_dist += (C[j][h] - V[i][h]) * (C[j][h] - V[i][h]);
				 
						if (new_dist < min_distance) 
					    		min_distance = new_dist;
				    }
				    RSS_standard += sqrt(abs(min_distance));
			    }
		    				    		
//////////////////////////////////////////////////////////////////////////////////////
		   					
				for (int e = 0; e < length_eps; e++) {
				
				    // Initial time
				    auto begin = chrono::high_resolution_clock::now();
				
			    	int p = ceil( spectral_norm * spectral_norm / N[n] * K * K / (eps[e] * eps[e]) * log(K/delta) );
			    	int q = ceil( (V21_norm / N[n])  * (V21_norm / N[n]) * K * K / (eps[e] * eps[e]) * log(K/delta) );
			    		
			 //   	cout << "p: " << p << endl;
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
				    C = C_initial;
					    		
						
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
								for (int h = 0; h < d; h++) 
							    	new_dist += (C[j][h] - V_P_sampled[i][h]) * (C[j][h] - V_P_sampled[i][h]);
						 
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
							    	new_dist += (C[j][h] - V_Q_sampled[i][h]) * (C[j][h] - V_Q_sampled[i][h]);
						 
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
					    		C_new[j] = C[j];			    
					    	else {
					    		double coeff = V21_norm / N[n] * p / q / C_size[j];
					    		for (int h = 0; h < d; h++)
					    			C_new[j][h] = coeff * C_new[j][h];
					    	}
					    }
					    	
					    // Compute distance between centroids
					    distance_centroids = 0;	
					    for (int j = 0; j < K; j++) {
					    	
					    	double distance_aux = 0;
							for (int h = 0; h < d; h++) 
						    	distance_aux += (C[j][h] - C_new[j][h]) * (C[j][h] - C_new[j][h]);
						  	distance_centroids += sqrt(abs(distance_aux));
					    }

					    C = C_new;
					    iterations++;					 
					}					
					iterations_EK_Kmeans[n][e] += (double (iterations)) / (n_iter_C * n_iter_N);

				    // Compute duration
				    auto end = chrono::high_resolution_clock::now();
				    double dur = chrono::duration_cast<chrono::milliseconds>(end - begin).count();
				    time_EK_Kmeans[n][e] += dur / (n_iter_N * n_iter_C);
				    	
				    // Compute RSS
				    double RSS = 0.;
				    for (int i = 0; i < N[n]; i++) {
				    	
					    double min_distance = DBL_MAX;
					    for (int j = 0; j < K; j++) {
					    	
							double new_dist = 0;
							for (int h = 0; h < d; h++) 
						    	new_dist += (C[j][h] - V[i][h]) * (C[j][h] - V[i][h]);
					 
							if (new_dist < min_distance) 
						    	min_distance = new_dist;
					    }
					    RSS += sqrt(abs(min_distance));
				    }
				    RSS_ratio[n][e] += 100 * (RSS - RSS_standard) / RSS_standard / (n_iter_C * n_iter_N);
				    	
				    //	cout << n << ' ' << iter_N << ' ' << iter_C << ' ' << e << endl;				    	  	
				}								
			}
		}
    }
    	
	ofstream MyFile("preprocessing_data.txt");
	MyFile << "# dataset_size preprocessing_ms" << endl;
	
	for (int n = 0; n < length_N; n++) {
		MyFile << N[n] << ' ' << preprocessing_time_ms[n] << endl;
		cout << "N: " << N[n] << endl;
		cout << "Milliseconds preprocessing: " << preprocessing_time_ms[n] << endl << endl;
	}
	MyFile.close();

	if (preprocessing_guard < 0) {
		cerr << "Preprocessing guard: " << preprocessing_guard << endl;
	}
	
	auto t1 = chrono::high_resolution_clock::now();
	cout << "Total seconds: " << chrono::duration_cast<chrono::seconds>(t1 - t0).count() << endl;
    	
    return 0;
}
