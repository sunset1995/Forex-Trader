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

// Training datas
vector<long double> input;
long double input_max;
long double input_min;

// Model
vector<int> obs;
vector< vector<long double> > A(N, vector<long double>(N, 0.0));
vector< vector<long double> > B(N, vector<long double>(M, 0.0));
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
inline long double add(long double a, long double b) {
	static long double logEPS = -log(EPS);
	if( b > a )
		swap(a, b);
	long double diff = b - a;
	if( diff < logEPS )
		return a;
	return a + log(1.0 + exp(diff));
}

vector<vector<long double>> fwd() {
	vector<vector<long double>> alpha(obs.size(), vector<long double>(N, 0.0));
	for(int i=0; i<N; ++i)
		alpha[0][i] = pi[i] * B[i][obs[0]];
	for(int t=1; t<obs.size(); ++t)
		for(int j=0; j<N; ++j) {
			for(int i=0; i<N; ++i)
				alpha[t][j] = alpha[t][j] + alpha[t-1][i] * A[i][j];
			alpha[t][j] = alpha[t][j] * B[j][obs[t]];
		}
	return alpha;
}

vector<vector<long double>> bwd() {
	vector< vector<long double> > beta(obs.size(), vector<long double>(N, 0.0));
	for(int i=0; i<N; ++i)
		beta[obs.size()-1][i] = 1.0;
	for(int t=obs.size()-2; t>=0; --t)
		for(int i=0; i<N; ++i)
			for(int j=0; j<N; ++j)
				beta[t][i] = beta[t][i] + A[i][j] * B[j][obs[t+1]] * beta[t+1][j];
	return beta;
}

void optimize() {
	vector< vector<long double> > alpha = fwd();
	vector< vector<long double> > beta = bwd();
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

	srand(time(NULL));
	read_input();
	random_init_model();

	for(int i=0; i<10; ++i) {
		optimize();
	}

	show_model();
	checksum_model();

	return 0;

}
