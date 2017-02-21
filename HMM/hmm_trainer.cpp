#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <algorithm>
#define N 6
#define M 10
#define INPUT_FILE_NAME "./datas/in-2016-02"
using namespace std;

// Training datas
vector<double> input;

// Model
vector< vector<double> > A(N, vector<double>(N, 0.0));
vector< vector<double> > B(N, vector<double>(M, 0.0));
vector<double> pi(N, 0);

// Init function
void read_input() {
	
	FILE *in = fopen(INPUT_FILE_NAME, "r");
	if( !in ) {
		fprintf(stderr, "%s\n", "Training datas not found.");
		exit(0);
	}

	double rate, last_rate;
	fscanf(in, "%lf", &last_rate);
	while( fscanf(in, "%lf", &rate) != EOF ) {
		input.emplace_back(rate - last_rate);
		last_rate = rate;
	}

	fclose(in);

}



int main() {

	read_input();

	return 0;

}
