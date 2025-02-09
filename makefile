DOUBLE = ON
all:
ifeq '$(DOUBLE)' 'ON'
	g++ laba1.cpp -o laba1 -DNUM
	./laba1
else
	g++ laba1.cpp -o laba1
	./laba1
endif