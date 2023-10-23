#include <stdio.h>
#include <stdlib.h>

int main()
{
	FILE* f = fopen("/proc/sys/vm/drop_caches", "w");
	if (f == NULL) {
		perror("Error opening /proc/sys/vm/drop_caches");
		return 1;
	}
	fprintf(f, "3");
	fclose(f);
	return 0;
}
