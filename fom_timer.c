/*

*/

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <sys/time.h>
#include "fom_timer.h"

double get_wtime()
{
	double t;
	struct timeval tm;
	gettimeofday(&tm, NULL);
	t = tm.tv_sec + (tm.tv_usec/1.0E6);
	return(t);
}

// Convert fractional time to nanosecond counter ticks
uint64_t Get_Time_ns_ticks()
{
	return( (uint64_t)(1.0E9 * get_wtime()));
}