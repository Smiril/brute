CC=clang
PREFIX=/usr/local
DOCDIR=${PREFIX}/share
LIBRARIES= -lpthread

all: bruteforce.o brute 

bruteforce.o:bruteforce.c
	$(CC) -o $@ -c $<

brute: bruteforce.o
	$(CC) -o $@ $+ $(LIBRARIES)

debug:
	${CC} -Wextra -Werror -pthread bruteforce.c -o brute

clean:
	-rm -rf *.o brute

install:
	install_name_tool -add_rpath "@executable_path/../lib/" ./brute
	install -s ./brute ${PREFIX}/bin
	-mkdir -p ${DOCDIR}
	chmod 755 ${DOCDIR}
	install -m 644 rockyou.txt ${DOCDIR}

uninstall:
	-rm -rf ${PREFIX}/bin/brute
	-rm -rf ${DOCDIR}
