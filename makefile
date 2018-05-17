OBJS=simulation.o proc.o params.o gridsize.o particle.o tensor0.o tensor1.o  poisson.o communicator.o grid.o
CPP=mpiicc
CFLAGS=-O3
FFT_DIR=/home/mani/yshirian/tools/fft
FFTW_DIR=/home/mani/yshirian/tools/fftw-2.1.5
HYPRE_DIR=/home/mani/yshirian/tools/HYPRE

CINCLUDES=-I $(HYPRE_DIR)/include -I $(FFTW_DIR)/include
CDEFS = -DHAVE_CONFIG_H -DHYPRE_TIMING -DFFT_FFTW
CFLAGS2=$(CFLAGS) $(CDEFS)
LFLAGS=-L $(FFT_DIR)/Obj_certainty -L $(FFTW_DIR)/lib -L $(HYPRE_DIR)/lib
LIBS=-lfft -lfftw -lfftw_mpi -lHYPRE -lm -lstdc++


default: box

box: $(OBJS)
	$(CPP) $(CFLAGS2) $(CINCLUDES) -o box $(OBJS) $(LFLAGS) $(LIBS)

simulation.o: simulation.cpp proc.h proc.cpp params.h params.cpp gridsize.h gridsize.cpp tensor0.h tensor0.cpp tensor1.h tensor1.cpp communicator.h communicator.cpp scalar_source.h grid.h grid.cpp particle.h particle.cpp
	$(CPP) $(CFLAGS2) $(CINCLUDES) -c simulation.cpp

grid.o: proc.h proc.cpp params.h params.cpp gridsize.h gridsize.cpp tensor0.h tensor0.cpp tensor1.h tensor1.cpp poisson.h poisson.cpp communicator.h communicator.cpp scalar_source.h grid.h grid.cpp particle.h particle.cpp
	$(CPP) $(CFLAGS2) $(CINCLUDES) -c grid.cpp

communicator.o: proc.h proc.cpp params.h params.cpp gridsize.h gridsize.cpp tensor0.h tensor0.cpp tensor1.h tensor1.cpp communicator.h communicator.cpp
	$(CPP) $(CFLAGS) -c communicator.cpp

poisson.o: $(FFT_DIR)/fft_3d.h $(FFT_DIR)/fft_3d.c proc.h proc.cpp params.h params.cpp gridsize.h gridsize.cpp tensor0.h tensor0.cpp tensor1.h tensor1.cpp poisson.h poisson.cpp
	$(CPP) $(CFLAGS2) $(CINCLUDES) -c poisson.cpp

particle.o: communicator.h communicator.cpp gridsize.h gridsize.cpp proc.h proc.cpp params.h params.cpp gridsize.h gridsize.cpp particle.h particle.cpp tensor0.h tensor0.cpp tensor1.h tensor1.cpp
	$(CPP) $(CFLAGS) -c particle.cpp

tensor1.o: communicator.h communicator.cpp gridsize.h gridsize.cpp tensor0.h tensor0.cpp tensor1.h tensor1.cpp
	$(CPP) $(CFLAGS) -c tensor1.cpp

tensor0.o: communicator.h communicator.cpp gridsize.h gridsize.cpp tensor0.h tensor0.cpp tensor1.h tensor1.cpp
	$(CPP) $(CFLAGS) -c tensor0.cpp

gridsize.o: proc.h proc.cpp params.h params.cpp gridsize.h gridsize.cpp
	$(CPP) $(CFLAGS) -c gridsize.cpp

params.o: params.h params.cpp
	$(CPP) $(CFLAGS) -c params.cpp

proc.o: proc.h proc.cpp
	$(CPP) $(CFLAGS) -c proc.cpp

clean:
	rm -rf *.o box
