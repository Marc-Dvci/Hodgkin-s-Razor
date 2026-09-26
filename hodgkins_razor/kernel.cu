// Batched simulator for cultured neuronal networks on a microelectrode array.
//
// One CUDA block simulates one network; one thread integrates one neuron.
// Neuron state lives in shared memory for the whole run, so the time loop
// never leaves the SM. Synaptic arrivals go through a shared ring buffer,
// which is why conduction delays are quantised to RB_STEP simulation steps.
//
// Units: mV, ms, nS, pA, pF.

#define NN 256          // neurons per network (16 x 16 grid)
#define NELEC 16        // electrodes per network (4 x 4 grid)
#define NPARAM 16       // columns of theta, see params.PARAMS
#define RB_BINS 16      // ring-buffer slots
#define RB_STEP 8       // simulation steps per slot
#define NWARP (NN / 32)
#define PER_ELEC 16384  // event capacity per electrode

// Fixed membrane constants, following Traub-Miles as used by Doorn et al. 2025.
#define CM 3.0f         // pF
#define GL 0.9f         // nS
#define EL (-39.2f)     // mV
#define EK (-80.0f)     // mV
#define ENA 70.0f       // mV
#define VT (-30.4f)     // mV
#define EEXC 0.0f       // mV
#define EINH (-70.0f)   // mV
#define TAU_AMPA 2.0f   // ms
#define TAU_NMDA 100.0f // ms
#define TAU_GABA 6.0f   // ms
#define TAU_CA 6000.0f  // ms
#define ALPHA_CA 0.00035f
#define REFRAC 2.0f     // ms
#define DEADTIME 0.2f   // ms, per electrode; each recording system adds its own, see simulator.VIEWS
#define TAU_AR 700.0f   // ms, decay of the asynchronous release rate (Doorn et al.)
#define U_MAX 0.5f      // per ms, saturation of the asynchronous release rate
#define X0 5.0f         // vesicles; an asynchronous event depletes 1/X0 of the pool
#define GNA_SCALE 150.0f
#define GKDR_SCALE 15.0f

__device__ __forceinline__ float exprel(float x) {
    // (exp(x) - 1) / x, stable near zero.
    return (fabsf(x) < 1e-4f) ? (1.0f + 0.5f * x) : ((__expf(x) - 1.0f) / x);
}

__device__ __forceinline__ unsigned int xorshift(unsigned int &s) {
    s ^= s << 13; s ^= s >> 17; s ^= s << 5;
    return s;
}

__device__ __forceinline__ float uniform01(unsigned int &s) {
    return (xorshift(s) >> 8) * (1.0f / 16777216.0f);
}

extern "C" __global__ void simulate(
    const float* __restrict__ theta,      // [B, 12] parameters, natural units
    const float* __restrict__ W,          // [B, NN, NN] weight from j to i at [b][j][i]
    const unsigned char* __restrict__ DB, // [NN*NN] delay slot, shared across networks
    const unsigned char* __restrict__ ISINH,  // [B, NN]
    const float* __restrict__ IBIAS,      // [B, NN]
    const int* __restrict__ ELEC,         // [NN] or [B, NN] electrode of each neuron, -1 if unseen
    const int elec_stride,                // 0: one map shared by the batch; NN: one map per network
    const int n_steps,
    const int n_transient,
    const float dt,
    int* __restrict__ out_time,           // [B, max_events] step index
    unsigned char* __restrict__ out_elec, // [B, max_events]
    int* __restrict__ out_count,          // [B]
    const int max_events,
    const unsigned int seed)
{
    extern __shared__ float sm[];
    float *V   = sm;
    float *m   = V + NN;
    float *h   = m + NN;
    float *n   = h + NN;
    float *Ca  = n + NN;
    float *ga  = Ca + NN;
    float *gn  = ga + NN;
    float *gg  = gn + NN;
    float *xd  = gg + NN;
    float *ref = xd + NN;
    float *rbE = ref + NN;             // [RB_BINS][NN]
    float *rbI = rbE + RB_BINS * NN;   // [RB_BINS][NN]
    // Who fired this step, as one bit per neuron. Reading the bits back in a
    // fixed order is what makes the run bit-reproducible: an atomic counter
    // would order the synaptic sums differently from one run to the next.
    unsigned int *wmask = (unsigned int*)(rbI + RB_BINS * NN);
    int   *ecount = (int*)(wmask + NWARP);   // [NELEC]
    int   *emask  = ecount + NELEC;          // electrodes touched this step
    float *lastEv = (float*)(emask + 1);
    unsigned char *inh = (unsigned char*)(lastEv + NELEC);

    const int b = blockIdx.x;
    const int i = threadIdx.x;

    const float noise_sd = theta[b * NPARAM + 0];
    const float g_na   = theta[b * NPARAM + 1] * GNA_SCALE;
    const float g_kdr  = theta[b * NPARAM + 2] * GKDR_SCALE;
    const float g_ahp  = theta[b * NPARAM + 3];
    const float g_ampa = theta[b * NPARAM + 4];
    const float g_nmda = theta[b * NPARAM + 5];
    const float g_gaba = theta[b * NPARAM + 6];
    const float g_tonic = theta[b * NPARAM + 7];   // bath GABA-A conductance
    const float tau_d  = theta[b * NPARAM + 10];
    const float u_rel  = theta[b * NPARAM + 11];
    const float u_asyn = theta[b * NPARAM + 15];   // asynchronous release strength

    unsigned int rs = seed ^ (b * 2654435761u) ^ (i * 40503u);
    rs |= 1u;
    for (int k = 0; k < 8; ++k) xorshift(rs);

    // Resting state, then let the transient settle the network.
    V[i] = EL + 3.0f * (uniform01(rs) - 0.5f);
    m[i] = 0.02f; h[i] = 0.9f; n[i] = 0.1f;
    Ca[i] = 0.0f; ga[i] = 0.0f; gn[i] = 0.0f; gg[i] = 0.0f;
    xd[i] = 1.0f; ref[i] = -1e9f;
    for (int k = 0; k < RB_BINS; ++k) { rbE[k * NN + i] = 0.0f; rbI[k * NN + i] = 0.0f; }
    if (i < NELEC) { lastEv[i] = -1e9f; ecount[i] = 0; }
    if (i < NWARP) wmask[i] = 0u;
    if (i == 0) emask[0] = 0;
    inh[i] = ISINH[b * NN + i];
    __syncthreads();

    const float ibias = IBIAS[b * NN + i];
    const int myelec = ELEC[(size_t)b * elec_stride + i];
    const float noise_amp = noise_sd * sqrtf(2.0f * GL / CM) * sqrtf(dt);
    const float dxd = dt / tau_d;
    const float da = __expf(-dt / TAU_AMPA);
    const float dn_ = __expf(-dt / TAU_NMDA);
    const float dg = __expf(-dt / TAU_GABA);
    const float dca = __expf(-dt / TAU_CA);
    const float dar = __expf(-dt / TAU_AR);
    // Asynchronous release, after Doorn et al.: each spike raises a release
    // rate that decays over hundreds of milliseconds. A release event
    // transmits like a spike and depletes the vesicle pool, but it is not a
    // somatic spike, so no electrode records it. This is what keeps a
    // culture firing, synaptically, between its network bursts.
    float uar = 0.0f;

    float gauss_cache = 0.0f;
    int gauss_ready = 0;
    int cur = 0;

    for (int step = 0; step < n_steps; ++step) {
        const float t = step * dt;

        // Deliver arrivals scheduled for this slot.
        if (step % RB_STEP == 0) {
            ga[i] += rbE[cur * NN + i];
            gn[i] += rbE[cur * NN + i];
            gg[i] += rbI[cur * NN + i];
            rbE[cur * NN + i] = 0.0f;
            rbI[cur * NN + i] = 0.0f;
            cur = (cur + 1) % RB_BINS;
            __syncthreads();
        }

        // Gating variables, exponential Euler.
        const float v = V[i];
        const float am = 1.28f / exprel((13.0f - v + VT) * 0.25f);
        const float bm = 1.4f / exprel((v - VT - 40.0f) * 0.2f);
        const float ah = 0.128f * __expf((17.0f - v + VT) * (1.0f / 18.0f));
        const float bh = 4.0f / (1.0f + __expf((40.0f - v + VT) * 0.2f));
        const float an = 0.16f / exprel((15.0f - v + VT) * 0.2f);
        const float bn = 0.5f * __expf((10.0f - v + VT) * 0.025f);

        float s = am + bm; float inf = am / s;
        m[i] = inf + (m[i] - inf) * __expf(-dt * s);
        s = ah + bh; inf = ah / s;
        h[i] = inf + (h[i] - inf) * __expf(-dt * s);
        s = an + bn; inf = an / s;
        n[i] = inf + (n[i] - inf) * __expf(-dt * s);

        // Conductances and the exponential-Euler voltage step.
        const float mm = m[i] * m[i] * m[i] * h[i];
        const float nn4 = n[i] * n[i] * n[i] * n[i];
        const float mg = 1.0f / (1.0f + __expf(-0.062f * v) / 3.57f);
        const float gA = g_ampa * ga[i];
        const float gN = g_nmda * gn[i] * mg;
        const float gG = g_gaba * gg[i];
        const float gH = g_ahp * Ca[i];

        const float G = GL + g_na * mm + g_kdr * nn4 + gA + gN + gG + gH + g_tonic;
        const float I0 = GL * EL + g_na * mm * ENA + g_kdr * nn4 * EK + ibias
                       + (gA + gN) * EEXC + (gG + g_tonic) * EINH + gH * EK;

        if (!gauss_ready) {
            const float u1 = fmaxf(uniform01(rs), 1e-7f);
            const float u2 = uniform01(rs);
            const float r = sqrtf(-2.0f * __logf(u1));
            gauss_cache = r * __sinf(6.2831853f * u2);
            V[i] = (I0 / G) + (v - I0 / G) * __expf(-dt * G / CM)
                 + noise_amp * r * __cosf(6.2831853f * u2);
            gauss_ready = 1;
        } else {
            V[i] = (I0 / G) + (v - I0 / G) * __expf(-dt * G / CM)
                 + noise_amp * gauss_cache;
            gauss_ready = 0;
        }

        ga[i] *= da; gn[i] *= dn_; gg[i] *= dg; Ca[i] *= dca;
        xd[i] += (1.0f - xd[i]) * dxd;
        __syncthreads();

        // Threshold crossing.
        const unsigned int fired =
            (v <= 0.0f && V[i] > 0.0f && t > ref[i]) ? 1u : 0u;
        if (fired) {
            ref[i] = t + REFRAC;
            Ca[i] += ALPHA_CA;
            uar += u_asyn * (U_MAX - uar);
        }
        uar *= dar;
        const unsigned int async =
            (!fired && uar > 1e-7f && uniform01(rs) < uar * dt) ? 1u : 0u;
        const unsigned int tx = fired | async;
        const unsigned int ballot = __ballot_sync(0xFFFFFFFFu, tx);
        if ((i & 31) == 0) wmask[i >> 5] = ballot;
        // An OR across the block is the barrier and the "did anyone fire"
        // reduction in one instruction.
        if (fired && myelec >= 0) atomicOr(emask, 1 << myelec);
        const int any = __syncthreads_or((int)tx);

        if (any) {
            // Each thread collects the arrivals addressed to its own neuron, in
            // neuron order, so no atomics are needed and the sum is fixed.
            for (int q = 0; q < NWARP; ++q) {
                unsigned int m = wmask[q];
                while (m) {
                    const int j = (q << 5) + (__ffs(m) - 1);
                    m &= (m - 1);
                    const float w = W[((size_t)b * NN + j) * NN + i];
                    if (w > 0.0f) {
                        const int slot = (cur + DB[j * NN + i]) % RB_BINS;
                        const float amt = w * xd[j];
                        if (inh[j]) rbI[slot * NN + i] += amt;
                        else                   rbE[slot * NN + i] += amt;
                    }
                }
            }
            __syncthreads();
            // Depress the sources only after every target has read xd.
            if (fired) xd[i] *= (1.0f - u_rel);
            else if (async) xd[i] *= (1.0f - 1.0f / X0);

            // One event per electrode per dead-time window, written into that
            // electrode's own slice of the buffer.
            if (i < NELEC && step >= n_transient && t - lastEv[i] >= DEADTIME) {
                if (emask[0] & (1 << i)) {
                    lastEv[i] = t;
                    const int c = ecount[i];
                    if (c < PER_ELEC) {
                        out_time[((size_t)b * NELEC + i) * PER_ELEC + c] =
                            step - n_transient;
                        out_elec[((size_t)b * NELEC + i) * PER_ELEC + c] =
                            (unsigned char)i;
                    }
                    ecount[i] = c + 1;
                }
            }
            __syncthreads();
            if (i < NWARP) wmask[i] = 0u;
            if (i == 0) emask[0] = 0;
        }
        __syncthreads();
    }

    if (i < NELEC) out_count[b * NELEC + i] = ecount[i];
}
