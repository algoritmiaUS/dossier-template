#include <bits/stdc++.h>

using namespace std;

#define rep(i, a, b) for (int i = (a); i < (b); ++i)
#define all(x) begin(x), end(x)
#define sz(x) (int)(x).size()
typedef long long ll;
typedef vector<int> vi;

// Incluimos tu implementación desde la carpeta de docs
#include "../../../docs/cpp/graphs/2sat.h"

static bool litValue(int lit, const vector<int>& assignment) {
	int var = max(lit, ~lit);
	return assignment[var] == (lit >= 0 ? 1 : 0);
}

static bool clauseSatisfied(int a, int b, const vector<int>& assignment) {
	return litValue(a, assignment) || litValue(b, assignment);
}

static bool bruteForceSat(int n, const vector<pair<int, int>>& clauses, vector<int>* witness = nullptr) {
	vector<int> assignment(n);
	for (int mask = 0; mask < (1 << n); ++mask) {
		for (int i = 0; i < n; ++i) assignment[i] = (mask >> i) & 1;
		bool ok = true;
		for (auto [a, b] : clauses) {
			if (!clauseSatisfied(a, b, assignment)) {
				ok = false;
				break;
			}
		}
		if (ok) {
			if (witness) *witness = assignment;
			return true;
		}
	}
	return false;
}

static void checkAssignment(const TwoSat& ts, const vector<pair<int, int>>& clauses) {
	for (auto [a, b] : clauses) {
		bool va = ts.values[max(a, ~a)] == (a >= 0);
		bool vb = ts.values[max(b, ~b)] == (b >= 0);
		assert(va || vb);
	}
}

static int randomLit(mt19937& rng, int n) {
	int var = uniform_int_distribution<int>(0, n - 1)(rng);
	return uniform_int_distribution<int>(0, 1)(rng) ? var : ~var;
}

static void test_unsat_case() {
	TwoSat ts(1);
	ts.setValue(0);
	ts.setValue(~0);
	assert(!ts.solve());
}

static void test_basic_case() {
	// (a | b) & (a | ~b) & (~c | ~d)
	TwoSat ts(4);
	ts.either(0, 1);
	ts.either(0, ~1);
	ts.either(~2, ~3);
	assert(ts.solve());
	assert(ts.values[0] == 1);
	assert(ts.values[2] == 0);
	assert(ts.values[3] == 0);
}

static void test_at_most_one() {
	mt19937 rng(123456);
	for (int it = 0; it < 200; ++it) {
		int n = uniform_int_distribution<int>(1, 7)(rng);
		int k = uniform_int_distribution<int>(1, n)(rng);
		vector<int> vars(n);
		iota(vars.begin(), vars.end(), 0);
		shuffle(vars.begin(), vars.end(), rng);
		vector<int> lits;
		for (int i = 0; i < k; ++i) {
			int var = vars[i];
			lits.push_back(uniform_int_distribution<int>(0, 1)(rng) ? var : ~var);
		}

		TwoSat ts(n);
		ts.atMostOne(lits);
		assert(ts.solve());
		int count = 0;
		for (int lit : lits) count += (ts.values[max(lit, ~lit)] == (lit >= 0));
		assert(count <= 1);
	}
}

static void test_random_bruteforce() {
	mt19937 rng(20260726);
	for (int it = 0; it < 250; ++it) {
		int n = uniform_int_distribution<int>(1, 8)(rng);
		int m = uniform_int_distribution<int>(1, 20)(rng);
		vector<pair<int, int>> clauses;
		clauses.reserve(m);

		TwoSat ts(n);
		for (int i = 0; i < m; ++i) {
			int a = randomLit(rng, n);
			int b = randomLit(rng, n);
			clauses.push_back({a, b});
			ts.either(a, b);
		}

		vector<int> witness;
		bool brute = bruteForceSat(n, clauses, &witness);
		bool solved = ts.solve();
		assert(solved == brute);
		if (solved) checkAssignment(ts, clauses);
	}
}

int main() {
	ios_base::sync_with_stdio(false);
	cin.tie(NULL);

	cout << "Iniciando stress test para 2-SAT (C++)..." << endl;

	test_unsat_case();
	test_basic_case();
	test_at_most_one();
	test_random_bruteforce();

	cout << "¡Todos los tests de C++ para 2-SAT pasaron con éxito!" << endl;
	return 0;
}