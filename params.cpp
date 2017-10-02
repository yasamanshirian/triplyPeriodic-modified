#include "params.h"
#include <fstream>
#include <sstream>
#include <string>
#include <iostream>
params::params(char *address)
{
  //read data from input file
  std::ifstream param(address);
  //a string buffer to save strings from files
  std::ostringstream param_trimmed_out;
  std::string input_line;
  while (!param.eof())
    {
      std::getline(param,input_line);
        //reading inputs as string to the param_trimmed_out, hashtga is 35?
      if ((int(input_line[0])!=0)&&(int(input_line[0])!=35)) param_trimmed_out<<input_line<<std::endl;
    }
  param.close();
  //copies a string we pass to it 
  std::istringstream param_trimmed_in(param_trimmed_out.str());  
  
  param_trimmed_in>>Nx_tot_>>Ny_tot_>>Nz_tot_;
  param_trimmed_in>>Lx_>>Ly_>>Lz_;
  param_trimmed_in>>bs_;
  param_trimmed_in>>Np_track_;
  param_trimmed_in>>T_final_>>dt_>>threshold_;
  param_trimmed_in>>Statistics_>>Stat_print_;
  param_trimmed_in>>Data_freq_fast_>>Data_freq_slow_;
  param_trimmed_in>>Data_dir_>>Stat_dir_;
  param_trimmed_in>>Initial_;
  param_trimmed_in>>Initial_C_;
  param_trimmed_in>>PreCond_>>Solver_;
  param_trimmed_in>>cooling_;
  param_trimmed_in>>Iteration1_>>Iteration2_>>epsilon_;
  param_trimmed_in>>TWC_>>ParticleGravity_;
  param_trimmed_in>>A1_>>A2_;
  param_trimmed_in>>Rho0_;
  param_trimmed_in>>U0_;
  param_trimmed_in>>T0_;
  param_trimmed_in>>Mu0_>>k_;
  param_trimmed_in>>gx1_>>gy1_>>gz1_>>gx2_>>gy2_>>gz2_;
  param_trimmed_in>>Cp_>>Cv_;
  param_trimmed_in>>np0_;
  param_trimmed_in>>Dp_>>Rhop_;
  param_trimmed_in>>Cvp_;
  param_trimmed_in>>epsilonp_;
  param_trimmed_in>>Nu_;
  param_trimmed_in>>I01_>>I02_;
  param_trimmed_in>>D_M_;
  param_trimmed_in>>A_g_>>K_g_;
  
  //initially update parameters
  R_=Cp_-Cv_;
  P0_=Rho0_*T0_*R_;
  Tp_=Rhop_*Dp_*Dp_/(18*Mu0_);
  mp_=Rhop_*PI*Dp_*Dp_*Dp_/6.;
  W_=Ly_;

  Iteration_=Iteration1_;
  A_=A1_;
  gx_=gx1_;
  gy_=gy1_;
  gz_=gz1_;
  I0_=I01_;
}

void params::update(double time)
{
  if (time>threshold_)
    {
      Iteration_=Iteration2_;
      A_=A2_;
      gx_=gx2_;
      gy_=gy2_;
      gz_=gz2_;
      I0_=I02_;
    }
}
