#ifndef communicator_h
#define communicator_h
#include <mpi.h> //MPI definitions are required
class proc;
class params;
class gridsize;
class tensor0;
class tensor1;
class communicator
{
  ////////////////////////////////required variables for data storage///////////////////////
  MPI_File Myfile;
  MPI_Status status;
  MPI_Datatype filetype;
  MPI_Datatype memtype;
  int gsizes[3];
  int psizes[3];
  int start_indices_l[3];
  int start_indices_g[3];
  int lsizes[3];
  int memsizes[3];
  int TNOD;
  ////////////////////////////////////////////////////////REQUIRED BUFERSS/////////////////////////////////////////////////////
  double **Sbuf;
  double **Rbuf;
  int N_max;
  ////////////////////////////////////////////////////////////////////////////////////////
  params *param_;
  proc *pc_;
  gridsize *size_;
  int Nx,Ny,Nz,bs;
  int proc_i,proc_j,proc_k;
  int procRIGHT,procLEFT,procTOP,procBOT,procFRONT,procREAR;
 public:
  communicator(){}
  communicator(gridsize*,params*,proc*);
  ~communicator();
  void read(tensor0&,const char*);
  void write(tensor0&,const char*);
  void read(tensor1&,const char*);
  void write(tensor1&,const char*);
  void SEND_RECV_BLOCKING(tensor0&);
  void SEND_RECV_BLOCKING(tensor1&);
  void SEND_RECV_BLOCKING_CUM(tensor0&); //(cumulative send_recv) halo cells value += to the value of neighbor
  void SEND_RECV_BLOCKING_CUM(tensor1&);
  void Sequential_Begin(int&); //This pair of functions allow you to execute a critical section of the code in a sequential manner. also you can pass an integer number amongst processes sequentially
  void Sequential_End(int&);
};
#endif
