#include <cstdio>
#include <cstdlib>
#include <cassert>
#include <cmath>
#include <ctime>
#include <vector>
#include <algorithm>
#define EPS 1e-12
#define N 6
#define M 11
#define SKIP 100
#define INPUT_AVG 0
#define INPUT_STDEV 0.000054
#define INPUT_FILE_NAME "./datas/in-2016-03"
using namespace std;

typedef vector<vector<long double>> LF_2D;

// Validation data
vector<long double> input;

// Model
vector<int> obs;
vector<vector<long double>> A = {
	{0.472186, 0.457760, 0.024973, 0.030245, 0.001582, 0.013255},
	{0.327511, 0.482830, 0.010714, 0.146532, 0.004318, 0.028094},
	{0.008557, 0.037933, 0.017366, 0.159998, 0.466406, 0.309740},
	{0.036507, 0.379769, 0.204269, 0.083674, 0.295753, 0.000027},
	{0.001313, 0.009821, 0.180939, 0.070567, 0.642823, 0.094537},
	{0.041117, 0.059538, 0.506746, 0.000028, 0.392570, 0.000001}
};
vector<vector<long double>> B = {
	{0.173692, 0.036986, 0.062285, 0.005971, 0.064391, 0.019038, 0.099327, 0.060000, 0.040266, 0.052519, 0.385525},
	{0.132989, 0.000740, 0.010869, 0.123603, 0.000022, 0.495285, 0.004300, 0.086149, 0.080186, 0.002288, 0.063568},
	{0.002339, 0.000000, 0.284820, 0.071449, 0.000003, 0.493148, 0.025067, 0.000004, 0.096574, 0.023177, 0.003419},
	{0.042663, 0.392380, 0.000321, 0.006804, 0.261050, 0.030001, 0.008505, 0.223642, 0.000019, 0.000739, 0.033876},
	{0.607476, 0.023053, 0.000464, 0.081841, 0.019646, 0.098846, 0.009982, 0.087264, 0.020930, 0.000001, 0.050498},
	{0.000038, 0.307573, 0.030426, 0.000004, 0.024091, 0.005441, 0.534159, 0.019837, 0.000003, 0.050173, 0.028255}
};
vector<long double> pi = {
	0.000000, 0.000000, 0.000000, 0.000000, 1.000000, 0.000000
};

// Init function
void read_input() {
	FILE *in = fopen(INPUT_FILE_NAME, "r");
	if( !in ) {
		fprintf(stderr, "%s\n", "Validation datas not found.");
		exit(0);
	}

	long double rate, last_rate;
	fscanf(in, "%Lf", &last_rate);
	while( fscanf(in, "%Lf", &rate) != EOF ) {
		input.emplace_back(rate - last_rate);
		last_rate = rate;
	}

	fclose(in);

	long double avg = INPUT_AVG;
	long double stdev = INPUT_STDEV;

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


// Helper function
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

bool checksum_model() {
	for(auto &ai : A) {
		long double sum = 0;
		for(auto &aij : ai)
			sum += aij;
		if( fabs(sum - 1.0) > 1e-4 )
			return false;
	}

	for(auto &bi : B) {
		long double sum = 0;
		for(auto &bij : bi)
			sum += bij;
		if( fabs(sum - 1.0) > 1e-4 )
			return false;
	}

	long double sum = 0;
	for(auto &p : pi)
		sum += p;
	if( fabs(sum - 1.0) > 1e-4 )
		return false;
	return true;
}

vector<long double> given_one(int observation, const vector<long double> &probability) {
	vector<long double> next_p(N, 0.0);
	long double total = 0;
	for(int j=0; j<N; ++j) {
		for(int i=0; i<N; ++i) {
			next_p[j] += probability[i] * A[i][j];
		}
		next_p[j] *= B[j][observation];
		total += next_p[j];
	}
	long double checksum = 0.0;
	for(auto &p : next_p) {
		p /= total;
		checksum += p;
	}
	assert(fabs(checksum - 1.0) < 1e-4);
	return next_p;
}

vector<long double> infer(const vector<long double> &probability) {
	vector<long double> next_hidden(N, 0.0);
	vector<long double> belief(M, 0.0);
	for(int i=0; i<N; ++i)
		for(int j=0; j<N; ++j)
			next_hidden[i] += probability[j] * A[j][i];
	for(int k=0; k<M; ++k)
		for(int i=0; i<N; ++i)
			belief[k] += next_hidden[i] * B[i][k];
	long double checksum = 0;
	for(auto p : belief)
		checksum += p;
	assert(fabs(checksum - 1.0) < 1e-4);
	return belief;
}

void validate() {
	// P(qt=Sj | O[1:t])
	vector<long double> probability = pi;

	for(int t=0; t<SKIP && t<obs.size(); ++t)
		probability = given_one(obs[t], probability);

	long long correct = 0;
	long long wrong = 0;
	long long effect_correct = 0;
	long long effect_wrong = 0;
	for(int t=SKIP; t<obs.size(); ++t) {
		vector<long double> belief = infer(probability);
		int guess = 0;
		for(int i=0; i<belief.size(); ++i) {
			if( belief[i] > belief[guess] )
				guess = i;
		}
		printf("guess %2d vs. correct %2d\n", guess, obs[t]);
		if( guess == obs[t] ) ++correct;
		else ++wrong;
		if( guess!=0 || obs[t]!=0 ) {
			if( (guess>M/2) ^ (obs[t]>M/2) ) ++effect_wrong;
			else ++effect_correct;
		}
		probability = given_one(obs[t], probability);
	}
	printf("\ncorrect %lld vs. wrong %lld\n", correct, wrong);
	printf("correct rate: %.2f\n", 100.0 * correct / (correct + wrong));
	
	printf("\neffective_correct %lld vs. effective_wrong %lld\n", effect_correct, effect_wrong);
	printf("effective_correct rate: %.2f\n", 100.0 * effect_correct / (effect_correct + effect_wrong));
}



int main() {

	read_input();
	show_model();
	printf("\nchecksum=%s\n\n", checksum_model() ? "PASS" : "FAIL" );

	validate();

	return 0;

}
