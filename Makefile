CC=clang
PREFIX=/usr/local
DOCDIR=${PREFIX}/share

all:
	${CC} -pthread bruteforce.c -O2 -o brute

debug:
	${CC} -Wextra -Werror -pthread bruteforce.c -O2 -o brute

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

