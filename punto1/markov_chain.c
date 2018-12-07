#include <stdio.h> 
#include <stdlib.h>
#include <omp.h>
#include <math.h>

void markov(int N);

int main(int argc, char ** argv) { 
#pragma omp parallel 
{ 
   markov(1000);
 } 
return 0; 
}

void markov(int N) {
    
    double U1,U2,W,mult,X1;
    int rank=omp_get_thread_num()+1;
    char str[12];
    sprintf(str, "proceso_%d", rank); 
    int i;
    FILE * out=fopen(str,"w");
    for(i=0;i<N;i++){
        do
        {
          U1 = -1 + ((double) rand () / RAND_MAX) * 2;
          U2 = -1 + ((double) rand () / RAND_MAX) * 2;
          W = pow (U1, 2) + pow (U2, 2);
        }
        while (W >= 1 || W == 0);
        mult = sqrt ((-2 * log (W)) / W);
        X1 = U1 * mult;
        fprintf(out,"%lf\n",X1);
    }
    fclose(out);
}