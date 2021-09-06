dlvl = ./.
include $(dlvl)/Makefile.in

all: SGD

SGD: SGD.c
	gcc -Wall -g -shared -o SGD.so -fPIC SGD.c -lm -I External_Libraries/CBLAS/include

cleanall:
	rm -f *.o SGD.so SGD SGD_lib.so
