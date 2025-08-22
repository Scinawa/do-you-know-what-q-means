#include<iostream>
#include<vector>
#include<math.h>
#include<random>
#include<time.h>
#include<float.h>
#include<chrono>
using namespace std;

int compute_label(vector<double>& v, vector<vector<double>>& centers, int k, int d) {
    int min_index = -1;
    double min_distance = DBL_MAX;
    for (int j = 0; j < k; j++) {
        double new_dist = 0;
        for (int h = 0; h < d; h++) {
            new_dist = (centers[j][h] - v[h]) * (centers[j][h] - v[h]);
        }
        if (new_dist < min_distance) {
            min_distance = new_dist;
            min_index = j;
        }
    }
    return min_index;
}

int main() {
    srand(time(0));
    random_device rd;
    mt19937 gen(rd());

    // Read the input
    int N,d,k,p,q;
    cin >> N >> d >> k >> p >> q;
    vector<vector<double>> V(N,vector<double>(d,0));
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < d; j++) {
            cin >> V[i][j];
        }
    }

    // Compute the norms
    vector<double> V_norms(N,0);
    double V21_norm = 0;
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < d; j++) {
            V_norms[i] += V[i][j] * V[i][j];
        }
        V_norms[i] = sqrt(V_norms[i]);
        V21_norm += V_norms[i];
    }

    // Initial time
    auto begin = chrono::high_resolution_clock::now();

    // Sample initial centroids uniformly from the dataset
    vector<vector<double>> C(k, vector<double>(d,0));
    for (int i = 0; i < k; i++) {
        C[i] = V[rand()%N];
    }

    // Compute cluster sizes
    vector<int> C_sizes(k,0);
    for (int i = 0; i < p; i++) {
        int j = rand()%N;
        int label = compute_label(V[j], C, k, d);
        C_sizes[label]++;
    }

    // Create the distribution for Q
    vector<double> Q_distr(N);
    for (int i = 0; i < N; i++) {
        Q_distr[i] = V_norms[i] / V21_norm;
    }
    discrete_distribution<> Q_dist(Q_distr.begin(), Q_distr.end());

    // Compute new centroids
    vector<vector<double>> new_centroids(k,vector<double>(d,0));
    for (int i = 0; i < q; i++) {
        int j = Q_dist(gen);
        int label = compute_label(V[j], C, k, d);
        double f = (V21_norm * p) / (q * N * C_sizes[label] * V_norms[j]);
        for (int h = 0; h < d; h++) {
            new_centroids[label][h] += f * V[j][h];
        }
    }

    // Compute duration
    auto end = chrono::high_resolution_clock::now();    
    auto dur = end - begin;
    auto ms = std::chrono::duration_cast<std::chrono::microseconds>(dur).count();
    cout << "Microseconds: " << ms << endl;

    // Output the new centroids
    for (int i = 0; i < k; i++) {
        for (int j = 0; j < d; j++) {
            cout << new_centroids[i][j] << ' ';
        }
        cout << endl;
    }
    
    return 0;
}