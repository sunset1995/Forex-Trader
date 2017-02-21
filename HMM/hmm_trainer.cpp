#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <ctime>
#include <vector>
#include <algorithm>
#define N 6
#define M 10
#define INPUT_FILE_NAME "./datas/in-2016-02"
using namespace std;

// Training datas
vector<long double> input;
long double input_max;
long double input_min;

// Model
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

// Helper function
void show_model() {
	puts("A");
	for(auto &ai : A) {
		for(auto &aij : ai)
			printf("%Lf ", aij);
		puts("");
	}
	puts("");

	puts("B");
	for(auto &bi : B) {
		for(auto &bij : bi)
			printf("%Lf ", bij);
		puts("");
	}
	puts("");

	puts("pi");
	for(auto &p : pi)
		printf("%Lf ", p);
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

	show_model();
	checksum_model();

	return 0;

}
