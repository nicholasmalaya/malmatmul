/*
	FOM timing helper routines
*/

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <sys/time.h>

#ifndef __AMD_CONTEST_TIMING__
#define __AMD_CONTEST_TIMING__

double get_wtime(); // time in seconds
uint64_t Get_Time_ns_ticks(); // timein nanosecond ticks

#endif