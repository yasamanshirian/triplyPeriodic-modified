#ifndef gridsize_h
#define gridsize_h
class params;
class proc;
class gridsize
{
  proc *pc_;
  int Nx_tot_; //total number of points in x direction
  int Ny_tot_; // ...
  int Nz_tot_; // ...
  int bs_; //BorderSize
  int Nx_; // number of nodes in x direction
  int Ny_;// ...
  int Nz_;// ...
  double Lx_;// physical length in x direction0
  double Ly_;// ...
  double Lz_;// ...
  double dx_;// dx, which is Lx/Nx
  double dy_;
  double dz_;
  int il_,ih_,jl_,jh_,kl_,kh_; //index of this chunk of data in the global chunk(e.g. 64^2 with 2x2x2 cpus) first cpu has: il=0,ih=31
  double xl_,xh_,yl_,yh_,zl_,zh_;
  int *Nxs_,*Nys_,*Nzs_; //store local grid size of each processor (without bordersize)
  int *OFFSET_x_,*OFFSET_y_,*OFFSET_z_; //number of grid points in x,y,z direction owned by processes with less I,J,K in the logical grid
 public: 
  int Nx() const{return Nx_;}
  int Ny() const{return Ny_;}
  int Nz() const{return Nz_;}
  int Nx_tot() const{return Nx_tot_;}
  int Ny_tot() const{return Ny_tot_;}
  int Nz_tot() const{return Nz_tot_;}
  int size_tot() const{return Nx_tot_*Ny_tot_*Nz_tot_;}
  int bs() const{return bs_;}
  double Lx() const{return Lx_;}
  double Ly() const{return Ly_;}
  double Lz() const{return Lz_;}
  double dx() const{return dx_;}
  double dy() const{return dy_;}
  double dz() const{return dz_;}
  double Vcell() const { return dx_*dy_*dz_; } //return volume of a cell
  int size() const{return Nx_*Ny_*Nz_;}
  int il() {return il_;}
  int ih() {return ih_;}
  int jl() {return jl_;}
  int jh() {return jh_;}
  int kl() {return kl_;}
  int kh() {return kh_;}
  double xl(){return xl_;} //return lower bound of x in the local grid (not cell center location)
  double xh(){return xh_;} //highe rlimit of x
  double yl(){return yl_;}
  double yh(){return yh_;}
  double zl(){return zl_;}
  double zh(){return zh_;}
  gridsize(){}
  gridsize(params*,proc*);
  ~gridsize();
}; 
#endif
