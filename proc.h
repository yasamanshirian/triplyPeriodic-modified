#ifndef proc_h
#define proc_h
#include <mpi.h> //Need MPI definitions
#include <iostream> //Need iostream functions for MPI_SAFE definition
#define MPI_SAFE_CALL( call ) do {                               \
    int err = call;                                              \
    if (err != MPI_SUCCESS) {                                    \
      fprintf(stderr, "MPI error %d in file '%s' at line %i",  \
              err, __FILE__, __LINE__);                         \
      exit(1);                                                 \
    } } while(0)
class proc
{
  int procTOP_,procBOT_,procLEFT_,procRIGHT_,procFRONT_,procREAR_; // rnak of neighbor processes
  int procROOT_; //rank of root process
  int myRank_; // my rank
  int totalNumProcessors_; //total number of processes
  int NumProcX_,NumProcY_,NumProcZ_; //number of processes in each direction
  int proc_i_,proc_j_,proc_k_; //indecis of this process in processes grid
  void NumProcCalculator(); //how to distribute total processes in each direction
 public:
  int NX() const{return NumProcX_;}
  int NY() const{return NumProcY_;}
  int NZ() const{return NumProcZ_;}
  int I() const{return proc_i_;}
  int J() const{return proc_j_;}
  int K() const{return proc_k_;}
  int TOP() const{return procTOP_;}
  int BOT() const{return procBOT_;}
  int LEFT() const{return procLEFT_;}
  int RIGHT() const{return procRIGHT_;}
  int FRONT() const{return procFRONT_;}
  int REAR() const{return procREAR_;}
  int RANK() const{return myRank_;}
  int TOT() const{return totalNumProcessors_;}
  bool IsRoot() const{return (myRank_==procROOT_);}
  int ROOT() const{return procROOT_;}
  proc();
  ~proc(){}
};
#endif
