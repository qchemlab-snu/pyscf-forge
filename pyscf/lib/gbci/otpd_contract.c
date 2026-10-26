#include <stdint.h>
#include <stdlib.h>

#ifdef _OPENMP
#include <omp.h>
#endif


/*
 * t1a, t1b: signed single excitations, rows [a, i, str1, str0, sign]
 */
void GBPDFTcontract_alpha_core(double *ci, int *conf_info_list, double *ov_list,
                             int *t1a, int t1a_size,
                             int ncas, int na, int nb, int ngroup,
                             double *Ka)
{
    (void)na;
    int kpair = ngroup * ngroup;
    size_t ka_size = (size_t)ncas * ncas * kpair;
    for (size_t idx = 0; idx < ka_size; idx++) {
        Ka[idx] = 0.0;
    }

    for (int inz = 0; inz < t1a_size; inz++) {
        int a = t1a[inz * 5 + 0];
        int i = t1a[inz * 5 + 1];
        int x = t1a[inz * 5 + 2];
        int y = t1a[inz * 5 + 3];
        double sign = t1a[inz * 5 + 4];

        for (int b = 0; b < nb; b++) {
            int p1 = conf_info_list[x * nb + b];
            int p2 = conf_info_list[y * nb + b];
            int pid = p1 * ngroup + p2;
            Ka[((size_t)a * ncas + i) * kpair + pid] +=
                sign * ci[x * nb + b] * ci[y * nb + b]
                * ov_list[p1 * ngroup + p2];
        }
    }
}


void GBPDFTcontract_beta_core(double *ci, int *conf_info_list, double *ov_list,
                            int *t1b, int t1b_size,
                            int ncas, int na, int nb, int ngroup,
                            double *Kb)
{
    int kpair = ngroup * ngroup;
    size_t kb_size = (size_t)ncas * ncas * kpair;
    for (size_t idx = 0; idx < kb_size; idx++) {
        Kb[idx] = 0.0;
    }

    for (int inz = 0; inz < t1b_size; inz++) {
        int a = t1b[inz * 5 + 0];
        int i = t1b[inz * 5 + 1];
        int u = t1b[inz * 5 + 2];
        int v = t1b[inz * 5 + 3];
        double sign = t1b[inz * 5 + 4];

        for (int x = 0; x < na; x++) {
            int p1 = conf_info_list[x * nb + u];
            int p2 = conf_info_list[x * nb + v];
            int pid = p1 * ngroup + p2;
            Kb[((size_t)a * ncas + i) * kpair + pid] +=
                sign * ci[x * nb + u] * ci[x * nb + v]
                * ov_list[p1 * ngroup + p2];
        }
    }
}


void GBPDFTcontract_active_pair(double *ci, int *conf_info_list, double *ov_list,
                              int *t1a, int t1a_size,
                              int *t1b, int t1b_size,
                              int ncas, int na, int nb, int ngroup,
                              double *out)
{
    (void)na;
    size_t out_size = (size_t)ncas * ncas * ncas * ncas;
    for (size_t idx = 0; idx < out_size; idx++) {
        out[idx] = 0.0;
    }

    for (int ia_nz = 0; ia_nz < t1a_size; ia_nz++) {
        int a = t1a[ia_nz * 5 + 0];
        int i = t1a[ia_nz * 5 + 1];
        int x = t1a[ia_nz * 5 + 2];
        int u = t1a[ia_nz * 5 + 3];
        double sa = t1a[ia_nz * 5 + 4];

        for (int ib_nz = 0; ib_nz < t1b_size; ib_nz++) {
            int b = t1b[ib_nz * 5 + 0];
            int j = t1b[ib_nz * 5 + 1];
            int y = t1b[ib_nz * 5 + 2];
            int v = t1b[ib_nz * 5 + 3];
            double sb = t1b[ib_nz * 5 + 4];

            int p1 = conf_info_list[x * nb + y];
            int p2 = conf_info_list[u * nb + v];
            out[(((size_t)a * ncas + i) * ncas + b) * ncas + j] +=
                ci[x * nb + y] * ci[u * nb + v]
                * ov_list[p1 * ngroup + p2] * sa * sb;
        }
    }
}
