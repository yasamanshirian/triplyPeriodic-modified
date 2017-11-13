#ifndef particle_h
#define particle_h
#include <vector>
#include <fstream>

class params;
class proc;
class gridsize;
class tensor1;
class tensor0;

class particle
{
  params *param_;
  proc *pc_;
  gridsize *size_;
  double idx,idy,idz;

  //send and recv required buffer
  double *Sbuf0;
  int Snum0;
  double *Sbuf1;
  int Snum1;
  double *Rbuf0;
  int Rnum0;
  double *Rbuf1;
  int Rnum1;
  int Buffer_max_size; //send and recv buffers maximum size

  int OFF_SET; //number of particles owned by processes with less rank
  int *Nps; //memory to store Np of all processors
  
  std::ofstream stat_trajectory; //to store particle trajectory (only used by root process)
  std::ofstream file_Np; //to store number of particles per process (only used by root process)
  
  int x2i(double); //returns LOCAL cpu i index using global x coordinate
  int y2j(double);
  int z2k(double);
  int x2i_round(double); //returns the GLOBAL index of cell center point at the right of the particle
  int y2j_round(double); //returns the GLOBAL index of cell center point at the up of the particle
  int z2k_round(double); //returns the GLOBAL index of cell center point at the top of the particle
  void SWAP(int,int); //swap all information of two particles in the vecotr (Q,Q_int,Q_new,Q_np1  for x,y,z,u,v,w,T)
  void ToBuf(int,double*); //dump all data of a particle, IN ORDER, to the sendbuf. arguments: index of the particle, pointer to buffer first empty postion
  void FromBuf();//get particle data from buffers and put into their vecotrs
  void RESIZE(); //resize all vecotrs to Np
  void SortValidX(); //check each particle to be within this processor x-range, BASED ON X_NEW, put them in appropriate buffers, and resize the vecotr
  void SortValidY();
  void SortValidZ();
  bool find_tag(int,double&,double&,double&); //look for a flag in this process and returns true if found it, also cna obtain x,y, and z of that particle
  void MODE_Lx(int); //modify value to be in [0-Lbox_x] based on x_new  ..... Also change x[i] and x_np1[i] value corresponding to modification of x_new
  void MODE_Ly(int); //modify value to be in [0-Lbox_y]
  void MODE_Lz(int); //modify value to be in [0-Lbox_z]
  void Send_Recv_X();
  void Send_Recv_Y();
  void Send_Recv_Z();
  void Store(char*,int,double*); //store a vecotr of doubles
  void Store(char*,int,int*); //store a vecotr of ints
  void Load(char*,double*); //load a vecotr of doubles from restart files
  void Load(char*,int*); //load a vecotr of ints from restart file
 public:
  int Np; //number of valid particles in this process
  //global coordinate
  std::vector<double> x;
  std::vector<double> y;
  std::vector<double> z;

  std::vector<double> x_new;
  std::vector<double> y_new;
  std::vector<double> z_new;

  std::vector<double> x_int;
  std::vector<double> y_int;
  std::vector<double> z_int;

  std::vector<double> x_np1;
  std::vector<double> y_np1;
  std::vector<double> z_np1;

  //particle velocity
  std::vector<double> u;
  std::vector<double> v;
  std::vector<double> w;

  std::vector<double> u_new;
  std::vector<double> v_new;
  std::vector<double> w_new;

  std::vector<double> u_int;
  std::vector<double> v_int;
  std::vector<double> w_int;

  std::vector<double> u_np1;
  std::vector<double> v_np1;
  std::vector<double> w_np1;

  std::vector<double> RHS_u;
  std::vector<double> RHS_v;
  std::vector<double> RHS_w;

  //interpolated gas velocity at particle location
  std::vector<double> ug;
  std::vector<double> vg;
  std::vector<double> wg;

  //interpolated gas temperature at particle location
  std::vector<double> Tg;

  //particle temperature
  std::vector<double> T;
  std::vector<double> T_new;
  std::vector<double> T_int;
  std::vector<double> T_np1;
  std::vector<double> RHS_T;

  //particle flag: give each particle an integer number. If flag is -1 it means it is not a valid particle
  std::vector<int> flag;

  particle(){}
  particle(params*,proc*,gridsize*);
  ~particle();
  void update_position(int,double,double); //should pass the RK4 substep count, pre and post coefficients corresponding to the current sub step
  void update_velocity(int,double,double); 
  void update_Temp(int,double,double);
  void gas2part_velocity(tensor1&); //interpolate gas velocity to particles at their intermediate location
  void part2gas_velocity(tensor1&); //reverse of the above (projection)
  void gas2part_Temp_int(tensor0&); //interpolate gas temperature to particles at their intermediate location
  void part2gas_Temp_int(tensor0&); //reverse of the above (projection)

  void gas2part_Temp_new(tensor0&); //interpolate gas temperature to particles at their new location
  void part2gas_Temp_new(tensor0&); //reverse of the above (projection)
  
void part2gas_concentration(tensor0&); // compute number of particles per cell using linear projection
  void part2gas_concentration2(tensor0&); // compute number of particles per cell by counting

  void Compute_RHS_Temp_int(); //compute RHS of particle nergy equation using T_int
  void Compute_RHS_Temp_new(); //same as above but use T_new
  
  void Send_Recv(); //communicate particles with adjacent processes 
  ///////////STATISTICS///////////
  int NP_TOT(); //Use MPI reduce to compute total number of particle (should remains constant)
  double max(std::vector<double>&); //return maximum value (MPI CALCULATION) of the vecotr with presumably size of Np
  double min(std::vector<double>&); //return minimum value (MPI CALCULATION) of the vecotr with presumably size of Np
  double mean(std::vector<double>&); //return mean value (MPI CALCULATION) of the vecotr with presumably size of Np
  void trajectory(double); //store trajectory of k particles
  double Balance_Index(); //return max(Np)/min(Np) as a notion of load balance amongs processes

  ////////DATA STORAGE////////
  void Store_All(int); //Store all particle quantities at the given timestep
  void load_random(); //load particles at uniformly random locations
  void Load_All(); //Load partcle variables from Restart files , with almost equal #of particles per process, and then locate particles in appropriate processes  
  void isOverlap(); //check if two particles are overlapped.
}; 
#endif
