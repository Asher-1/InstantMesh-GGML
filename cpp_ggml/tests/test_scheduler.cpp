// Parity test: EulerAncestralScheduler (C++) vs diffusers reference trace.
//
// Fixtures are produced by convert/parity_scheduler.py into
// ${CMAKE_SOURCE_DIR}/benchmarks/fixtures/scheduler/. When missing the test
// exits with SKIP_RETURN_CODE 77 (counts as skipped in ctest).
//
//   python3 convert/parity_scheduler.py   # then ctest
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "models/scheduler.hpp"

#ifndef SCHEDULER_FIXTURE_DIR
#define SCHEDULER_FIXTURE_DIR "benchmarks/fixtures/scheduler"
#endif

namespace {

std::vector<float> read_bin(const std::string & path, size_t expect) {
    std::FILE * f = std::fopen(path.c_str(), "rb");
    if (!f) return {};
    std::fseek(f, 0, SEEK_END);
    long bytes = std::ftell(f);
    std::fseek(f, 0, SEEK_SET);
    if (bytes < 0 || (size_t) bytes != expect * sizeof(float)) { std::fclose(f); return {}; }
    std::vector<float> out(expect);
    if (std::fread(out.data(), sizeof(float), expect, f) != expect) { std::fclose(f); return {}; }
    std::fclose(f);
    return out;
}

} // namespace

int main() {
    const std::string dir = SCHEDULER_FIXTURE_DIR;
    const int steps = 75;
    const int n = 4 * 80 * 120;

    auto ts_ref   = read_bin(dir + "/timesteps.bin", steps);
    auto sig_ref  = read_bin(dir + "/sigmas.bin", steps + 1);
    auto ac_ref   = read_bin(dir + "/alphas_cumprod.bin", 1000);
    auto x0_ref   = read_bin(dir + "/x0.bin", n);
    auto ni_ref   = read_bin(dir + "/noise_init.bin", n);
    auto xini_ref = read_bin(dir + "/x_init.bin", n);
    auto xsc0_ref = read_bin(dir + "/x_scaled0.bin", n);
    auto eps_ref  = read_bin(dir + "/eps.bin", (size_t) steps * n);
    auto anc_ref  = read_bin(dir + "/anc_noise.bin", (size_t) steps * n);
    auto xfin_ref = read_bin(dir + "/x_final.bin", n);

    if (ts_ref.empty() || sig_ref.empty() || xfin_ref.empty() || ac_ref.empty()) {
        std::printf("SKIP: fixtures missing under %s — run convert/parity_scheduler.py\n", dir.c_str());
        return 77;
    }

    EulerAncestralScheduler sched;
    sched.set_alphas_cumprod(ac_ref.data(), (int) ac_ref.size());
    sched.set_timesteps(steps);
    int fails = 0;

    // 1) timesteps + sigmas tables.
    double max_ts = 0, max_sg = 0;
    for (int i = 0; i < steps; ++i) {
        max_ts = std::max(max_ts, (double) std::fabs(sched.timesteps()[i] - ts_ref[i]));
        max_sg = std::max(max_sg, (double) std::fabs(sched.sigmas()[i] - sig_ref[i]));
    }
    max_sg = std::max(max_sg, (double) std::fabs(sched.sigmas()[steps] - sig_ref[steps]));
    std::printf("%s timesteps/sigmas tables: max_ts=%.3g max_sg=%.3g\n",
                (max_ts < 1e-6 && max_sg < 1e-6) ? "PASS" : "FAIL", max_ts, max_sg);
    if (!(max_ts < 1e-6 && max_sg < 1e-6)) ++fails;

    // 2) add_noise at t0.
    {
        std::vector<float> x(n);
        sched.add_noise(x0_ref.data(), ni_ref.data(), n, ts_ref[0], x.data());
        double m = 0;
        for (int k = 0; k < n; ++k) m = std::max(m, (double) std::fabs(x[k] - xini_ref[k]));
        std::printf("%s add_noise t0: max_abs=%.3g\n", m < 1e-5 ? "PASS" : "FAIL", m);
        if (!(m < 1e-5)) ++fails;
    }

    // 3) scale_model_input at step 0.
    {
        std::vector<float> xs(n);
        sched.scale_model_input(xini_ref.data(), n, xs.data());
        double m = 0;
        for (int k = 0; k < n; ++k) m = std::max(m, (double) std::fabs(xs[k] - xsc0_ref[k]));
        std::printf("%s scale_model_input step0: max_abs=%.3g\n", m < 1e-5 ? "PASS" : "FAIL", m);
        if (!(m < 1e-5)) ++fails;
    }

    // 4) full 75-step walk with fixed eps/ancestral noise -> x_final.
    {
        std::vector<float> x = xini_ref, nxt(n);
        for (int i = 0; i < steps; ++i) {
            const float * eps = eps_ref.data() + (size_t) i * n;
            const float * anc = anc_ref.data() + (size_t) i * n;
            sched.step(eps, x.data(), anc, n, nxt.data());
            std::copy(nxt.begin(), nxt.end(), x.begin());
        }
        double m = 0;
        for (int k = 0; k < n; ++k) m = std::max(m, (double) std::fabs(x[k] - xfin_ref[k]));
        std::printf("%s 75-step walk -> x_final: max_abs=%.3g\n", m < 1e-4 ? "PASS" : "FAIL", m);
        if (!(m < 1e-4)) ++fails;
    }

    std::printf("%s\n", fails == 0 ? "ALL PASS" : "FAILURES PRESENT");
    return fails == 0 ? 0 : 1;
}
