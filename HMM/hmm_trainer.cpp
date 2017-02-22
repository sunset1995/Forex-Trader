#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <ctime>
#include <vector>
#include <algorithm>
#define EPS 1e-12
#define N 6
#define M 11
#define INPUT_FILE_NAME "./datas/in-2016-02"
using namespace std;

typedef vector<vector<long double>> LF_2D;

// Training datas
vector<long double> input;
long double input_max;
long double input_min;

// Model
vector<int> obs;
LF_2D A(N, vector<long double>(N, 0.0));
LF_2D log_A(N, vector<long double>(N, 0.0));
LF_2D B(N, vector<long double>(M, 0.0));
LF_2D log_B(N, vector<long double>(M, 0.0));
vector<long double> pi(N, 0);

// Init function
void read_input() {
	FILE *in = fopen(INPUT_FILE_NAME, "r");
	if( !in ) {
		fprintf(stderr, "%s\n", "Training datas not found.");
		exit(0);
	}

	long double rate, last_rate;
	fscanf(in, "%Lf", &last_rate);
	while( fscanf(in, "%Lf", &rate) != EOF ) {
		input.emplace_back(rate - last_rate);
		last_rate = rate;
	}

	fclose(in);

	input_max = input_min = input[0];
	for(auto r : input) {
		input_max = max(input_max, r);
		input_min = min(input_min, r);
	}

	long double avg = 0;
	for(auto r : input)
		avg += r;
	avg /= input.size();

	long double stdev = 0;
	for(auto r : input)
		stdev += (r-avg) * (r-avg);
	stdev = sqrt(stdev / input.size());

	obs = vector<int>(input.size());
	long double rng = stdev / (M>>1);
	long double base = rng / 2.0;
	for(int i=0; i<input.size(); ++i)
		if( fabs(input[i]) < base )
			obs[i] = 0;
		else if( input[i] > 0 )
			obs[i] = min(int((input[i]-base)/rng) + 1, (M>>1));
		else if( input[i] < 0 )
			obs[i] = min(int((-input[i]-base)/rng) + 5, M-1);
}

void random_distribution(vector<long double> &vec) {
	long double sum = 0;
	for(auto &v : vec) {
		v = rand();
		sum += v;
	}
	for(auto &v : vec)
		v /= sum;
}

void random_init_model() {
	for(auto &ai : A)
		random_distribution(ai);

	for(auto &bi : B)
		random_distribution(bi);

	random_distribution(pi);
}

// Training
inline long double mul(long double a, long double b) {
	return a + b;
}
inline long double div(long double a, long double b) {
	return a - b;
}
inline long double add(long double a, long double b) {
	if( b > a ) swap(a, b);
	return a + log(1.0 + exp(b-a));
}

LF_2D fwd() {
	LF_2D alpha(obs.size(), vector<long double>(N));
	for(int i=0; i<N; ++i)
		alpha[0][i] = log(pi[i] * B[i][obs[0]]);
	for(int t=1; t<obs.size(); ++t)
		for(int j=0; j<N; ++j) {
			alpha[t][j] = mul(alpha[t-1][0], log_A[0][j]);
			for(int i=1; i<N; ++i)
				alpha[t][j] = add(alpha[t][j], mul(alpha[t-1][i], log_A[i][j]));
			alpha[t][j] = mul(alpha[t][j], log_B[j][obs[t]]);
		}
	return alpha;
}

LF_2D bwd() {
	LF_2D beta(obs.size(), vector<long double>(N));
	for(int i=0; i<N; ++i)
		beta[obs.size()-1][i] = log(1.0);
	for(int t=obs.size()-2; t>=0; --t)
		for(int i=0; i<N; ++i) {
			beta[t][i] = mul(log_A[i][0], mul(log_B[0][obs[t+1]], beta[t+1][0]));
			for(int j=1; j<N; ++j)
				beta[t][i] = add(beta[t][i], mul(log_A[i][j], mul(log_B[j][obs[t+1]], beta[t+1][j])));
		}
	return beta;
}

LF_2D fwd_bwd(const LF_2D &alpha, const LF_2D &beta) {
	LF_2D gamma(obs.size(), vector<long double>(N));
	for(int t=0; t<obs.size(); ++t) {
		long double sum = 0;
		for(int i=0; i<N; ++i) {
			gamma[t][i] = mul(alpha[t][i], beta[t][i]);
			sum = (i==0) ? gamma[t][i] : add(sum, gamma[t][i]);
		}
		for(int i=0; i<N; ++i)
			gamma[t][i] = div(gamma[t][i], sum);
	}
	return gamma;
}

LF_2D xi(int t, const LF_2D &alpha, const LF_2D &beta) {
	LF_2D xi(N, vector<long double>(N));
	long double sum = -1;
	for(int i=0; i<N; ++i)
		for(int j=0; j<N; ++j) {
			xi[i][j] = mul(alpha[t][i], mul(log_A[i][j], mul(log_B[j][obs[t+1]], beta[t+1][j])));
			if( i==0 && j==0 ) sum = xi[i][j];
			else sum = add(sum, xi[i][j]);
		}
	for(auto &xi_ti : xi)
		for(auto &xi_tij : xi_ti)
			xi_tij = div(xi_tij, sum);
	return xi;
}

void optimize() {
	for(int i=0; i<N; ++i)
		for(int j=0; j<N; ++j)
			log_A[i][j] = log(A[i][j]);
	for(int i=0; i<N; ++i)
		for(int j=0; j<M; ++j)
			log_B[i][j] = log(B[i][j]);

	LF_2D alpha = fwd();
	LF_2D beta = bwd();
	LF_2D gamma = fwd_bwd(alpha, beta);

	// Coculate new better model
	LF_2D A_son(N, vector<long double>(N));
	vector<long double> A_mom(N);
	LF_2D B_son(N, vector<long double>(M));
	vector<bool> B_son_visited(M, false);
	vector<long double> B_mom(N);
	vector<long double> n_pi(N, 0.0);

	// Count transition probability
	for(int t=0; t<obs.size()-1; ++t) {
		LF_2D xi_t = xi(t, alpha, beta);
		for(int i=0; i<N; ++i) {
			for(int j=0; j<N; ++j)
				if( t==0 ) A_son[i][j] = xi_t[i][j];
				else A_son[i][j] = add(A_son[i][j], xi_t[i][j]);
			if( t==0 ) A_mom[i] = gamma[t][i];
			else A_mom[i] = add(A_mom[i], gamma[t][i]);
		}
	}
	
	// Count emmision probability
	for(int t=0; t<obs.size(); ++t) {
		for(int j=0; j<N; ++j) {
			if( !B_son_visited[obs[t]] ) B_son[j][obs[t]] = gamma[t][j];
			else B_son[j][obs[t]] = add(B_son[j][obs[t]], gamma[t][j]);

			if( t==0 ) B_mom[j] = gamma[t][j];
			else B_mom[j] = add(B_mom[j], gamma[t][j]);
		}
		B_son_visited[obs[t]] = true;
	}

	// Count initial state distribution
	for(int i=0; i<N; ++i)
		n_pi[i] = gamma[0][i];

	// Assign new better model
	for(int i=0; i<N; ++i)
		for(int j=0; j<N; ++j)
			A[i][j] = exp(div(A_son[i][j], A_mom[i]));
	for(int i=0; i<N; ++i)
		for(int j=0; j<M; ++j)
			B[i][j] = exp(div(B_son[i][j], B_mom[i]));
	for(int i=0; i<N; ++i)
		pi[i] = exp(n_pi[i]);
}

// Helper function
void show_model() {
	puts("A: transition probability");
	for(auto &ai : A) {
		for(auto &aij : ai)
			printf(" %Lf", aij);
		puts("");
	}
	puts("");

	puts("B: emission probability");
	for(auto &bi : B) {
		for(auto &bij : bi)
			printf(" %Lf", bij);
		puts("");
	}
	puts("");

	puts("pi: initial state distribution");
	for(auto &p : pi)
		printf(" %Lf", p);
	puts("");
}

void checksum_model() {
	for(auto &ai : A) {
		long double sum = 0;
		for(auto &aij : ai)
			sum += aij;
		if( fabs(sum - 1.0) > 1e-9 )
			fprintf(stderr, "checksum failed\n");
	}

	for(auto &bi : B) {
		long double sum = 0;
		for(auto &bij : bi)
			sum += bij;
		if( fabs(sum - 1.0) > 1e-9 )
			fprintf(stderr, "checksum failed\n");
	}

	long double sum = 0;
	for(auto &p : pi)
		sum += p;
	if( fabs(sum - 1.0) > 1e-9 )
		fprintf(stderr, "checksum failed\n");
}



int main() {

	srand(850311);
	read_input();
	random_init_model();

	for(int i=0; i<1; ++i) {
		optimize();
	}

	show_model();
	checksum_model();

	return 0;

}
