#include <stdint.h>
#include <stddef.h>


/*
 * t1a, t1b: signed single excitations, rows [a, i, str1, str0, sign]
 * t2aa, t2bb: signed double excitations, rows [a2, i2, a1, i1, str2, str0, sign]
 */
void gbci_contract_h_spin1(double *erieff, double *ci0, double *ci1,
int ncas, int nelecasa, int nelecasb,
int *conf_info_list, int na, uint64_t *stringsa, int nb, uint64_t *stringsb,
int num, int *t1a, int t1a_size, int *t1b, int t1b_size,
int *t2aa, int t2aa_size, int *t2bb, int t2bb_size,
double *TSc, double *energy_core)
{
    (void)nelecasa;
    (void)nelecasb;
    (void)stringsa;
    (void)stringsb;

    const size_t ncas2 = (size_t)ncas * ncas;
    const size_t ncas4 = ncas2 * ncas2;
    const size_t eri_p1_stride = (size_t)num * ncas4;

#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (int i = 0; i < t1a_size; i++) {
        const int aa = t1a[i*5+0];
        const int ia = t1a[i*5+1];
        const int str1a = t1a[i*5+2];
        const int str0a = t1a[i*5+3];
        const int signa = t1a[i*5+4];

        for (int j = 0; j < t1b_size; j++) {
            const int ab = t1b[j*5+0];
            const int ib = t1b[j*5+1];
            const int str1b = t1b[j*5+2];
            const int str0b = t1b[j*5+3];
            const int signb = t1b[j*5+4];
            const int p1 = conf_info_list[str1a * nb + str1b];
            const int p2 = conf_info_list[str0a * nb + str0b];

            const size_t ci1_idx = (size_t)str1a * nb + str1b;
            const size_t ci0_idx = (size_t)str0a * nb + str0b;
            const size_t eri_idx = (size_t)p1 * eri_p1_stride
                + (size_t)p2 * ncas4
                + (size_t)aa * ncas * ncas * ncas
                + (size_t)ia * ncas2
                + (size_t)ab * ncas
                + ib;

            const double contrib = ci0[ci0_idx] * erieff[eri_idx]
                * signa * signb
                * TSc[p1 * num + p2] * 2.0;
#ifdef _OPENMP
#pragma omp atomic
#endif
            ci1[ci1_idx] += contrib;
        }
    }

#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (int i = 0; i < t2aa_size; i++) {
        const int a1 = t2aa[i*7+0];
        const int i1 = t2aa[i*7+1];
        const int a2 = t2aa[i*7+2];
        const int i2 = t2aa[i*7+3];
        const int str1a = t2aa[i*7+4];
        const int str0a = t2aa[i*7+5];
        const int sign = t2aa[i*7+6];

        for (int str0b = 0; str0b < nb; str0b++) {
            const int p1 = conf_info_list[str1a * nb + str0b];
            const int p2 = conf_info_list[str0a * nb + str0b];

            const size_t ci1_idx = (size_t)str1a * nb + str0b;
            const size_t ci0_idx = (size_t)str0a * nb + str0b;
            const size_t eri_idx = (size_t)p1 * eri_p1_stride
                + (size_t)p2 * ncas4
                + (size_t)a1 * ncas * ncas * ncas
                + (size_t)i1 * ncas2
                + (size_t)a2 * ncas
                + i2;

            const double contrib = ci0[ci0_idx] * erieff[eri_idx]
                * sign * TSc[p1 * num + p2];
#ifdef _OPENMP
#pragma omp atomic
#endif
            ci1[ci1_idx] += contrib;
        }
    }

#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (int i = 0; i < t2bb_size; i++) {
        const int a1 = t2bb[i*7+0];
        const int i1 = t2bb[i*7+1];
        const int a2 = t2bb[i*7+2];
        const int i2 = t2bb[i*7+3];
        const int str1b = t2bb[i*7+4];
        const int str0b = t2bb[i*7+5];
        const int sign = t2bb[i*7+6];

        for (int str0a = 0; str0a < na; str0a++) {
            const int p1 = conf_info_list[str0a * nb + str1b];
            const int p2 = conf_info_list[str0a * nb + str0b];

            const size_t ci1_idx = (size_t)str0a * nb + str1b;
            const size_t ci0_idx = (size_t)str0a * nb + str0b;
            const size_t eri_idx = (size_t)p1 * eri_p1_stride
                + (size_t)p2 * ncas4
                + (size_t)a1 * ncas * ncas * ncas
                + (size_t)i1 * ncas2
                + (size_t)a2 * ncas
                + i2;

            const double contrib = ci0[ci0_idx] * erieff[eri_idx]
                * sign * TSc[p1 * num + p2];
#ifdef _OPENMP
#pragma omp atomic
#endif
            ci1[ci1_idx] += contrib;
        }
    }

#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (int str0a = 0; str0a < na; str0a++) {
        for (int str0b = 0; str0b < nb; str0b++) {
            const int p = conf_info_list[str0a * nb + str0b];
            const size_t ci_idx = (size_t)str0a * nb + str0b;
            ci1[ci_idx] += energy_core[p] * ci0[ci_idx];
        }
    }
}
