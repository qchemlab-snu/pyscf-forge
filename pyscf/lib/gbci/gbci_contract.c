#include <stdint.h>
#include <stddef.h>


/*
 * t1a, t1b: signed single excitations, rows [a, i, str1, str0, sign]
 * t2aa, t2bb: signed double excitations, rows [a2, i2, a1, i1, str2, str0, sign]
 * The lists are sorted by excitation first, so consecutive rows reuse the same
 * erieff slice.
 *
 * Each list is cut into nchunk equal ranges of rows. Chunk c accumulates into
 * buf[c] (nchunk x na x nb, zeroed by the caller), and the chunks are summed in
 * a fixed order at the end. There are no atomic updates, and since the chunks
 * do not depend on the number of threads, neither does the result.
 */
void gbci_contract_h_spin1(double *erieff, double *ci0, double *ci1,
int ncas, int nelecasa, int nelecasb,
int *conf_info_list, int na, uint64_t *stringsa, int nb, uint64_t *stringsb,
int num, int *t1a, int t1a_size, int *t1b, int t1b_size,
int *t2aa, int t2aa_size, int *t2bb, int t2bb_size,
double *TSc, double *energy_core, int nchunk, double *buf)
{
    (void)nelecasa;
    (void)nelecasb;
    (void)stringsa;
    (void)stringsb;

    const size_t ncas2 = (size_t)ncas * ncas;
    const size_t ncas4 = ncas2 * ncas2;
    const size_t eri_p1_stride = (size_t)num * ncas4;
    const size_t ndet = (size_t)na * nb;

#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1)
#endif
    for (int c = 0; c < nchunk; c++) {
        double *out = buf + (size_t)c * ndet;

        /* alpha-beta */
        const int e0 = (int)((int64_t)t1a_size * c / nchunk);
        const int e1 = (int)((int64_t)t1a_size * (c + 1) / nchunk);
        for (int e = e0; e < e1; e++) {
            const int aa = t1a[e*5+0];
            const int ia = t1a[e*5+1];
            const int str1a = t1a[e*5+2];
            const int str0a = t1a[e*5+3];
            const int signa = t1a[e*5+4];
            double *row = out + (size_t)str1a * nb;
            const int *conf_row = conf_info_list + (size_t)str1a * nb;
            const double *ci0_row = ci0 + (size_t)str0a * nb;
            const int *conf0_row = conf_info_list + (size_t)str0a * nb;
            const size_t eri_a = (size_t)aa * ncas * ncas2 + (size_t)ia * ncas2;

            for (int f = 0; f < t1b_size; f++) {
                const int ab = t1b[f*5+0];
                const int ib = t1b[f*5+1];
                const int str1b = t1b[f*5+2];
                const int str0b = t1b[f*5+3];
                const int signb = t1b[f*5+4];
                const int p1 = conf_row[str1b];
                const int p2 = conf0_row[str0b];
                const size_t eri_idx = (size_t)p1 * eri_p1_stride
                    + (size_t)p2 * ncas4 + eri_a + (size_t)ab * ncas + ib;
                row[str1b] += ci0_row[str0b] * erieff[eri_idx]
                    * signa * signb * TSc[p1 * num + p2] * 2.0;
            }
        }

        /* alpha-alpha */
        const int g0 = (int)((int64_t)t2aa_size * c / nchunk);
        const int g1 = (int)((int64_t)t2aa_size * (c + 1) / nchunk);
        for (int e = g0; e < g1; e++) {
            const int a1 = t2aa[e*7+0];
            const int i1 = t2aa[e*7+1];
            const int a2 = t2aa[e*7+2];
            const int i2 = t2aa[e*7+3];
            const int str1a = t2aa[e*7+4];
            const int str0a = t2aa[e*7+5];
            const int sign = t2aa[e*7+6];
            double *row = out + (size_t)str1a * nb;
            const int *conf_row = conf_info_list + (size_t)str1a * nb;
            const double *ci0_row = ci0 + (size_t)str0a * nb;
            const int *conf0_row = conf_info_list + (size_t)str0a * nb;
            const size_t eri_a = (size_t)a1 * ncas * ncas2 + (size_t)i1 * ncas2
                + (size_t)a2 * ncas + i2;

            for (int str0b = 0; str0b < nb; str0b++) {
                const int p1 = conf_row[str0b];
                const int p2 = conf0_row[str0b];
                const size_t eri_idx = (size_t)p1 * eri_p1_stride
                    + (size_t)p2 * ncas4 + eri_a;
                row[str0b] += ci0_row[str0b] * erieff[eri_idx]
                    * sign * TSc[p1 * num + p2];
            }
        }

        /* beta-beta */
        const int h0 = (int)((int64_t)t2bb_size * c / nchunk);
        const int h1 = (int)((int64_t)t2bb_size * (c + 1) / nchunk);
        for (int f = h0; f < h1; f++) {
            const int a1 = t2bb[f*7+0];
            const int i1 = t2bb[f*7+1];
            const int a2 = t2bb[f*7+2];
            const int i2 = t2bb[f*7+3];
            const int str1b = t2bb[f*7+4];
            const int str0b = t2bb[f*7+5];
            const int sign = t2bb[f*7+6];
            const size_t eri_b = (size_t)a1 * ncas * ncas2 + (size_t)i1 * ncas2
                + (size_t)a2 * ncas + i2;

            for (int str0a = 0; str0a < na; str0a++) {
                const int p1 = conf_info_list[(size_t)str0a * nb + str1b];
                const int p2 = conf_info_list[(size_t)str0a * nb + str0b];
                const size_t eri_idx = (size_t)p1 * eri_p1_stride
                    + (size_t)p2 * ncas4 + eri_b;
                out[(size_t)str0a * nb + str1b] += ci0[(size_t)str0a * nb + str0b]
                    * erieff[eri_idx] * sign * TSc[p1 * num + p2];
            }
        }
    }

    /* sum the chunks in a fixed order, then the bath energies */
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (size_t d = 0; d < ndet; d++) {
        double v = 0.0;
        for (int c = 0; c < nchunk; c++) {
            v += buf[(size_t)c * ndet + d];
        }
        ci1[d] = v + energy_core[conf_info_list[d]] * ci0[d];
    }
}
