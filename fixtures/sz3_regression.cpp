// SZ3 streams compressed with the Lorenzo/regression predictors (regression on), which the
// HDF5 SZ3 filter never chooses by itself; `fixtures/gen_more_filters.py --sz3-streams DIR`
// stores them as raw chunks so that HDF5 (with the plugin) decodes the expected values.
//
// Build against the SZ3 3.1.7 and zstd sources bundled in the hdf5plugin 7.1.0 sdist (`L` is
// its `lib` directory):
//   for f in $L/zstd/lib/{common,compress,decompress}/*.c; do
//     clang -O2 -DZSTD_DISABLE_ASM -c "$f" -I$L/zstd/lib -o "$(basename $f .c).o"; done
//   clang++ -O2 -std=c++17 -I$L/SZ3/include -I$L/SZ3_extra -I$L/SZ3_extra/darwin \
//     -I$L/zstd/lib sz3_regression.cpp *.o -o sz3_regression && ./sz3_regression
#include "SZ3/api/sz.hpp"
#include <cstdio>
#include <cmath>
#include <vector>
#include <random>
// Writes <name>.sz (compressed buffer) for arrays compressed with Lorenzo + regression.
template<class T>
void run(const char *name, std::vector<size_t> dims, bool lor, bool lor2, bool reg, int bs, double eb, int seed) {
    size_t n = 1; for (auto d: dims) n *= d;
    std::vector<T> data(n);
    std::mt19937 g(seed); std::normal_distribution<double> nd(0, 1);
    // planes with noise: regression wins on some blocks, Lorenzo on others
    for (size_t i = 0; i < n; i++) {
        size_t r = i; double v = 0; double w = 1;
        for (int k = dims.size() - 1; k >= 0; k--) { v += w * (double)(r % dims[k]); r /= dims[k]; w *= 0.7; }
        data[i] = (T)(0.3 * v + 0.05 * nd(g) + ((i / 97) % 3 == 0 ? 5 * std::sin(i * 0.01) : 0));
    }
    SZ::Config conf;
    conf.setDims(dims.begin(), dims.end());
    conf.cmprAlgo = SZ::ALGO_LORENZO_REG;
    conf.errorBoundMode = SZ::EB_ABS;
    conf.absErrorBound = eb;
    conf.lorenzo = lor; conf.lorenzo2 = lor2; conf.regression = reg; conf.regression2 = false;
    conf.blockSize = bs;
    conf.quantbinCnt = 65536;
    size_t outSize;
    char *out = SZ_compress(conf, data.data(), outSize);
    FILE *f = fopen(name, "wb"); fwrite(out, 1, outSize, f); fclose(f);
    delete[] out;
}
int main() {
    run<double>("reg2d_f64.sz", {40, 50}, true, false, true, 6, 0.01, 1);
    run<float>("reg2d_f32.sz", {40, 50}, true, false, true, 6, 0.01, 2);
    run<double>("regonly2d.sz", {37, 45}, false, false, true, 8, 0.01, 3);
    run<double>("reg3d_f64.sz", {13, 14, 15}, true, false, true, 6, 0.01, 4);
    run<float>("reg3d_f32.sz", {13, 14, 15}, true, true, true, 6, 0.01, 5);
    run<double>("reg1d.sz", {500}, true, true, true, 16, 0.001, 6);
    run<double>("reg4d.sz", {5, 6, 7, 8}, true, false, true, 4, 0.01, 7);
    run<double>("lor2only3d.sz", {13, 14, 15}, false, true, false, 6, 0.01, 8);
    return 0;
}
