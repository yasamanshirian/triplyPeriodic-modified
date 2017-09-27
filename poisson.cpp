#include "params.h"
#include "proc.h"
#include "gridsize.h"
#include "tensor0.h"
#include "tensor1.h"
#include <mpi.h>
#include <complex>
#include "communicator.h"
#include "poisson.h"

poisson::poisson(params* par,proc* pro,gridsize* gri,communicator* com):center_coeff(gri,com),dummy(gri,com),RHS_new(gri,com),Index(gri,com)
{
  param_=par;
  pc_=pro;
  size_=gri;
  in_ilo=gri->il();
  in_ihi=gri->ih();
  in_jlo=gri->jl();
  in_jhi=gri->jh();
  in_klo=gri->kl();
  in_khi=gri->kh();
  out_ilo=in_ilo;
  out_ihi=in_ihi;
  out_jlo=in_jlo;
  out_jhi=in_jhi;
  out_klo=in_klo;
  out_khi=in_khi;
  bs_=size_->bs();
  Nx_tot_=gri->Nx_tot();
  Ny_tot_=gri->Ny_tot();
  Nz_tot_=gri->Nz_tot();
  NxNy_tot_=Nx_tot_*Ny_tot_;
  Nx_=gri->Nx(); Ny_=gri->Ny(); Nz_=gri->Nz();
  length_=(Nx_-2*bs_)*(Ny_-2*bs_)*(Nz_-2*bs_);
  epsilon_=par->epsilon();
  if ((param_->Solver()>0)&&(param_->Solver()<7))
    {
      /* HYPRE INITIALIZE */
      //determine range of rows owned by this process in the Big matrix //this part is done sequentially with no extra cost on general parallelization
      if (pc_->IsRoot()) lengthS_=new int[pc_->TOT()];
      MPI_Gather(&length_,1,MPI_INT,lengthS_,1,MPI_INT,pc_->ROOT(),MPI_COMM_WORLD);
      //compute how many particles owned by proccesses with less rank              
      if (pc_->IsRoot())
	{
	  for (int i(1);i<pc_->TOT();i++)
	    lengthS_[i]+=lengthS_[i-1];
	  for (int i(pc_->TOT()-1);i>0;i--)
	    lengthS_[i]=lengthS_[i-1];
	  lengthS_[0]=0;
	}
      MPI_Scatter(lengthS_,1,MPI_INT,&ilower,1,MPI_INT,pc_->ROOT(),MPI_COMM_WORLD);
      //load Hash table values to Index
      int gindex=ilower;
      for (int k(bs_);k<Nz_-bs_;k++)
	for (int j(bs_);j<Ny_-bs_;j++)
	  for (int i(bs_);i<Nx_-bs_;i++)
	    {
	      Index(i,j,k)=gindex;
	      gindex++;
	    }
      Index.Update_Ghosts();
      iupper=ilower+length_-1;
      /* allocate extra memory required fro Hypre */
      rhs_values = new double[length_];
      x_values = new double[length_];
      rows = new int[length_];
      nnz=7; //number of nonzeros per row in the Big matrix
      /* assuming constant mesh size */
      inv_dx2=1./(size_->dx()*size_->dx());
      inv_dy2=1./(size_->dy()*size_->dy());
      inv_dz2=1./(size_->dz()*size_->dz());
      /* Create the matrix*/
      HYPRE_IJMatrixCreate(MPI_COMM_WORLD,ilower,iupper,ilower,iupper,&A);
      /* Create the rhs*/
      HYPRE_IJVectorCreate(MPI_COMM_WORLD,ilower,iupper,&b);
      /* Create the solution*/  
      HYPRE_IJVectorCreate(MPI_COMM_WORLD,ilower,iupper,&x);
      //set matrix and vecotrs
      //Choose a parallel csr format storage
      HYPRE_IJMatrixSetObjectType(A, HYPRE_PARCSR);
      HYPRE_IJVectorSetObjectType(b, HYPRE_PARCSR);
      HYPRE_IJVectorSetObjectType(x, HYPRE_PARCSR);
      //Initialize before setting coefficients
    }
  /* FFT INITIALIZE */
  nbuff=(in_ihi-in_ilo)*(in_jhi-in_jlo)*(in_khi-in_klo);
  plan=fft_3d_create_plan(MPI_COMM_WORLD,gri->Nx_tot(),gri->Ny_tot(),gri->Nz_tot(),in_ilo,in_ihi,in_jlo,in_jhi,in_klo,in_khi,out_ilo,out_ihi,out_jlo,out_jhi,out_klo,out_khi,0,0,&nbuff);
  if (param_->Solver()==0)
    {
      fft_data=new FFT_DATA[nbuff];
    }
}

poisson::~poisson()
{
  if (param_->Solver()==0)
    {
      //  fft_3d_destroy_plan(plan);	
      delete[] fft_data;
    }
  if ((param_->Solver()>0)&&(param_->Solver()<7))
    {
      if (pc_->IsRoot()) delete [] lengthS_;
      /* release extra memory required by Hypre */
      delete[] rhs_values;
      delete[] x_values;
      delete[] rows;
      /* Hypre Clean up */
      HYPRE_IJMatrixDestroy(A);
      HYPRE_IJVectorDestroy(b);
      HYPRE_IJVectorDestroy(x);
    }
}

void poisson::Solve(tensor1 &C,tensor0 &P,tensor0 &RHS)
{
  if (param_->Solver()==0) FFT(C,P,RHS);
  if ((param_->Solver()>0)&&(param_->Solver()<7))      //Coomon part for all Hypre  solvers
    {
      HYPRE_IJMatrixInitialize(A);
      /*setting coefficients*/
      /* march through all elements of the box correponding to this process*/
      int I; //refers to the I'th row of the Big matrix
      int i_local(0); //a counter shows the position in local portion of rhs/solution
      for (int k(bs_);k<Nz_-bs_;k++)
	for (int j(bs_);j<Ny_-bs_;j++)
	  for (int i(bs_);i<Nx_-bs_;i++)
	    {
	      I=Index(i,j,k);
	      cols[0]=Index(i-1,j,k); values[0]=C.x(i,j,k)*inv_dx2;
	      cols[1]=Index(i+1,j,k); values[1]=C.x(i+1,j,k)*inv_dx2;
	      cols[2]=Index(i,j-1,k); values[2]=C.y(i,j,k)*inv_dy2;
	      cols[3]=Index(i,j+1,k); values[3]=C.y(i,j+1,k)*inv_dy2;
	      cols[4]=Index(i,j,k-1); values[4]=C.z(i,j,k)*inv_dz2;
	      cols[5]=Index(i,j,k+1); values[5]=C.z(i,j,k+1)*inv_dz2;
	      cols[6]=I; values[6]=-(inv_dx2*(C.x(i+1,j,k)+C.x(i,j,k))+inv_dy2*(C.y(i,j+1,k)+C.y(i,j,k))+inv_dz2*(C.z(i,j,k+1)+C.z(i,j,k)));
	      // Set the values for row I
	      if (I) HYPRE_IJMatrixSetValues(A,1,&nnz,&I,cols,values);
	      else   //to avoid singularity
		{
		  nnz=1;
		  cols[0]=0; values[0]=1;
		  HYPRE_IJMatrixSetValues(A,1,&nnz,&I,cols,values);
		  nnz=7;
		}
	      i_local++;
	    }
      /* Assemble after setting the coefficients */
      HYPRE_IJMatrixAssemble(A);
      /* set right_hand side, solution(guess or zero) */
      i_local=0;
      for (int k(bs_);k<Nz_-bs_;k++)
	for (int j(bs_);j<Ny_-bs_;j++)
	  for (int i(bs_);i<Nx_-bs_;i++)
	    {
	      rhs_values[i_local]=RHS(i,j,k);
	      x_values[i_local]=P(i,j,k);
	      rows[i_local]=i_local+ilower;//Index(i,j,k);
	      i_local++;
	    }
      HYPRE_IJVectorInitialize(b);
      HYPRE_IJVectorSetValues(b,iupper-ilower+1,rows,rhs_values);
      HYPRE_IJVectorAssemble(b);
      HYPRE_IJVectorInitialize(x);
      HYPRE_IJVectorSetValues(x,iupper-ilower+1,rows,x_values);
      HYPRE_IJVectorAssemble(x);
      /* Get the parcsr matrix object to use */
      HYPRE_IJMatrixGetObject(A, (void**) &parcsr_A);
      HYPRE_IJVectorGetObject(b, (void**) &par_b);
      HYPRE_IJVectorGetObject(x, (void**) &par_x);
      /* SOLVE */
      if (param_->Solver()==1) AMG(C,P,RHS);
      if (param_->Solver()==2) PCG(C,P,RHS);
      if (param_->Solver()==3) GMRES(C,P,RHS);
      if (param_->Solver()==4) FlexGMRES(C,P,RHS);
      if (param_->Solver()==5) LGMRES(C,P,RHS);
      if (param_->Solver()==6) BiCGSTAB(C,P,RHS);
      /*copy solution from X to P*/
      HYPRE_IJVectorGetValues(x,iupper-ilower+1,rows,x_values);
      i_local=0;
      for (int k(bs_);k<Nz_-bs_;k++)
	for (int j(bs_);j<Ny_-bs_;j++)
	  for (int i(bs_);i<Nx_-bs_;i++)
	    P(i,j,k)=x_values[i_local++];
      P.Update_Ghosts();//update P's ghost cells
    } //end of common part for all Hypre solvers
}


void poisson::AMG(tensor1 &C,tensor0 &P,tensor0 &RHS)
{
  HYPRE_BoomerAMGCreate(&solver); 
  /* Set some parameters*/
  HYPRE_BoomerAMGSetPrintLevel(solver, 0);  /* print solve info + parameters */
  HYPRE_BoomerAMGSetCoarsenType(solver, 6); /* Falgout coarsening */
  HYPRE_BoomerAMGSetRelaxType(solver, 3);   /* G-S/Jacobi hybrid relaxation */
  HYPRE_BoomerAMGSetNumSweeps(solver, 1);   /* Sweeeps on each level */
  HYPRE_BoomerAMGSetMaxLevels(solver, 20);  /* maximum number of levels */
  HYPRE_BoomerAMGSetTol(solver,epsilon_);   /* conv. tolerance */
  HYPRE_BoomerAMGSetStrongThreshold(solver,.5); /* strong threshold=0.5 for 3d problem*/
  HYPRE_BoomerAMGSetMaxIter(solver, param_->Iteration()); /* do only one iteration! */
  /* Now setup and solve! */
  HYPRE_BoomerAMGSetup(solver, parcsr_A, par_b, par_x);
  HYPRE_BoomerAMGSolve(solver, parcsr_A, par_b, par_x);
  /* Get solve info */
  HYPRE_BoomerAMGGetNumIterations(solver, &num_iteration_);
  /* Destroy solver */
  HYPRE_BoomerAMGDestroy(solver);
}

void poisson::PCG(tensor1 &C,tensor0 &P,tensor0 &RHS)
{
  /* Create solver */
  HYPRE_ParCSRPCGCreate(MPI_COMM_WORLD, &solver);
  /* Set some parameters */
  HYPRE_PCGSetMaxIter(solver, param_->Iteration()); /* max iterations */
  HYPRE_PCGSetTwoNorm(solver, 1); /* use the two norm as the stopping criteria */
  HYPRE_PCGSetPrintLevel(solver, 0); /* prints out the iteration info */
  HYPRE_PCGSetLogging(solver, 1); /* needed to get run info later */
  HYPRE_PCGSetTol(solver, epsilon_); /* conv. tolerance */
  /* Now set up the preconditioner and specify any parameters */
  if (param_->PreCond()==1) //AMG
    {
      HYPRE_BoomerAMGCreate(&precond);
      HYPRE_BoomerAMGSetPrintLevel(precond, 0); /* print amg solution info */
      HYPRE_BoomerAMGSetCoarsenType(precond, 6);
      HYPRE_BoomerAMGSetRelaxType(precond, 6); /* Sym G.S./Jacobi hybrid */
      HYPRE_BoomerAMGSetNumSweeps(precond, 1);
      HYPRE_BoomerAMGSetTol(precond, 0.0); /* conv. tolerance zero */
      HYPRE_BoomerAMGSetMaxIter(precond, 1); /* do only one iteration! */
      HYPRE_BoomerAMGSetStrongThreshold(precond,.5); /* strong threshold=0.5 for 3d problem*/
      /* Set the PCG preconditioner */
      HYPRE_PCGSetPrecond(solver, (HYPRE_PtrToSolverFcn) HYPRE_BoomerAMGSolve,(HYPRE_PtrToSolverFcn)HYPRE_BoomerAMGSetup,precond);
    }
  /* Now setup and solve! */
  HYPRE_ParCSRPCGSetup(solver, parcsr_A, par_b, par_x);
  HYPRE_ParCSRPCGSolve(solver, parcsr_A, par_b, par_x);
  /* Get solve info */
  HYPRE_PCGGetNumIterations(solver, &num_iteration_);
  /* Destroy solver */
  HYPRE_ParCSRPCGDestroy(solver);
  /* Destroy precon(if any) */
  if (param_->PreCond()==1) HYPRE_BoomerAMGDestroy(precond);
}

void poisson::GMRES(tensor1 &C,tensor0 &P,tensor0 &RHS)
{
  /* Create solver */
  HYPRE_ParCSRGMRESCreate(MPI_COMM_WORLD, &solver);
  /* Set some parameters */
  HYPRE_GMRESSetMaxIter(solver, param_->Iteration()); /* max iterations */
  HYPRE_GMRESSetPrintLevel(solver, 0); /* prints out the iteration info */
  HYPRE_GMRESSetLogging(solver, 1); /* needed to get run info later */
  HYPRE_GMRESSetTol(solver, epsilon_); /* conv. tolerance */
  /* Now set up the preconditioner and specify any parameters */
  if (param_->PreCond()==1) //AMG
    {
      HYPRE_BoomerAMGCreate(&precond);
      HYPRE_BoomerAMGSetPrintLevel(precond, 0); /* print amg solution info */
      HYPRE_BoomerAMGSetCoarsenType(precond, 6);
      HYPRE_BoomerAMGSetRelaxType(precond, 6); /* Sym G.S./Jacobi hybrid */
      HYPRE_BoomerAMGSetNumSweeps(precond, 1);
      HYPRE_BoomerAMGSetTol(precond, 0.0); /* conv. tolerance zero */
      HYPRE_BoomerAMGSetMaxIter(precond, 1); /* do only one iteration! */
      HYPRE_BoomerAMGSetStrongThreshold(precond,.5); /* strong threshold=0.5 for 3d problem*/
      /* Set the PCG preconditioner */
      HYPRE_GMRESSetPrecond(solver, (HYPRE_PtrToSolverFcn) HYPRE_BoomerAMGSolve,(HYPRE_PtrToSolverFcn)HYPRE_BoomerAMGSetup,precond);
    }
  /* Now setup and solve! */
  HYPRE_ParCSRGMRESSetup(solver,parcsr_A,par_b,par_x);
  HYPRE_ParCSRGMRESSolve(solver,parcsr_A,par_b,par_x);
  /*Set Tolerance*/
  HYPRE_GMRESSetTol(solver,epsilon_);
  /* Get solve info */
   HYPRE_GMRESGetNumIterations(solver, &num_iteration_);
  /* Destroy solver */
  HYPRE_ParCSRGMRESDestroy(solver);
  /* Destroy precon(if any) */
  if (param_->PreCond()==1) HYPRE_BoomerAMGDestroy(precond);
}

void poisson::FlexGMRES(tensor1 &C,tensor0 &P,tensor0 &RHS)
{
  /* Create solver */
  HYPRE_ParCSRFlexGMRESCreate(MPI_COMM_WORLD, &solver);
  /* Set some parameters */
  HYPRE_FlexGMRESSetMaxIter(solver, param_->Iteration()); /* max iterations */
  HYPRE_FlexGMRESSetPrintLevel(solver, 0); /* prints out the iteration info */
  HYPRE_FlexGMRESSetLogging(solver, 1); /* needed to get run info later */
  HYPRE_FlexGMRESSetTol(solver, epsilon_); /* conv. tolerance */
    /* Now set up the preconditioner and specify any parameters */
  if (param_->PreCond()==1) //AMG
    {
      HYPRE_BoomerAMGCreate(&precond);
      HYPRE_BoomerAMGSetPrintLevel(precond, 0); /* print amg solution info */
      HYPRE_BoomerAMGSetCoarsenType(precond, 6);
      HYPRE_BoomerAMGSetRelaxType(precond, 6); /* Sym G.S./Jacobi hybrid */
      HYPRE_BoomerAMGSetNumSweeps(precond, 1);
      HYPRE_BoomerAMGSetTol(precond, 0.0); /* conv. tolerance zero */
      HYPRE_BoomerAMGSetMaxIter(precond, 1); /* do only one iteration! */
      HYPRE_BoomerAMGSetStrongThreshold(precond,.5); /* strong threshold=0.5 for 3d problem*/
      /* Set the PCG preconditioner */
      HYPRE_FlexGMRESSetPrecond(solver, (HYPRE_PtrToSolverFcn) HYPRE_BoomerAMGSolve,(HYPRE_PtrToSolverFcn)HYPRE_BoomerAMGSetup,precond);
    }
  /* Now setup and solve! */
  HYPRE_ParCSRFlexGMRESSetup(solver,parcsr_A,par_b,par_x);
  HYPRE_ParCSRFlexGMRESSolve(solver,parcsr_A,par_b,par_x);
  /*Set Tolerance*/
  HYPRE_FlexGMRESSetTol(solver,epsilon_);
  /* Get solve info */
   HYPRE_FlexGMRESGetNumIterations(solver, &num_iteration_);
  /* Destroy solver */
  HYPRE_ParCSRFlexGMRESDestroy(solver);
  /* Destroy precon(if any) */
  if (param_->PreCond()==1) HYPRE_BoomerAMGDestroy(precond);
}

void poisson::LGMRES(tensor1 &C,tensor0 &P,tensor0 &RHS)
{
  /* Create solver */
  HYPRE_ParCSRLGMRESCreate(MPI_COMM_WORLD, &solver);
  /* Set some parameters */
  HYPRE_LGMRESSetMaxIter(solver, param_->Iteration()); /* max iterations */
  HYPRE_LGMRESSetPrintLevel(solver, 0); /* prints out the iteration info */
  HYPRE_LGMRESSetLogging(solver, 1); /* needed to get run info later */
  HYPRE_LGMRESSetTol(solver, epsilon_); /* conv. tolerance */
  /* Now set up the preconditioner and specify any parameters */
  if (param_->PreCond()==1) //AMG
    {
      HYPRE_BoomerAMGCreate(&precond);
      HYPRE_BoomerAMGSetPrintLevel(precond, 0); /* print amg solution info */
      HYPRE_BoomerAMGSetCoarsenType(precond, 6);
      HYPRE_BoomerAMGSetRelaxType(precond, 6); /* Sym G.S./Jacobi hybrid */
      HYPRE_BoomerAMGSetNumSweeps(precond, 1);
      HYPRE_BoomerAMGSetTol(precond, 0.0); /* conv. tolerance zero */
      HYPRE_BoomerAMGSetMaxIter(precond, 1); /* do only one iteration! */
      HYPRE_BoomerAMGSetStrongThreshold(precond,.5); /* strong threshold=0.5 for 3d problem*/
      /* Set the PCG preconditioner */
      HYPRE_LGMRESSetPrecond(solver, (HYPRE_PtrToSolverFcn) HYPRE_BoomerAMGSolve,(HYPRE_PtrToSolverFcn)HYPRE_BoomerAMGSetup,precond);
    }
  /* Now setup and solve! */
  HYPRE_ParCSRLGMRESSetup(solver,parcsr_A,par_b,par_x);
  HYPRE_ParCSRLGMRESSolve(solver,parcsr_A,par_b,par_x);
  /*Set Tolerance*/
  HYPRE_LGMRESSetTol(solver,epsilon_);
  /* Get solve info */
   HYPRE_LGMRESGetNumIterations(solver, &num_iteration_);
  /* Destroy solver */
  HYPRE_ParCSRLGMRESDestroy(solver);
  /* Destroy precon(if any) */
  if (param_->PreCond()==1) HYPRE_BoomerAMGDestroy(precond);
}

void poisson::BiCGSTAB(tensor1 &C,tensor0 &P,tensor0 &RHS)
{
  /* Create solver */
  HYPRE_ParCSRBiCGSTABCreate(MPI_COMM_WORLD, &solver);
  /* Set some parameters */
  HYPRE_BiCGSTABSetMaxIter(solver, param_->Iteration()); /* max iterations */
  HYPRE_BiCGSTABSetPrintLevel(solver, 0); /* prints out the iteration info */
  HYPRE_BiCGSTABSetLogging(solver, 1); /* needed to get run info later */
  HYPRE_BiCGSTABSetTol(solver, epsilon_); /* conv. tolerance */
  /* Now set up the preconditioner and specify any parameters */
  if (param_->PreCond()==1) //AMG
    {
      HYPRE_BoomerAMGCreate(&precond);
      HYPRE_BoomerAMGSetPrintLevel(precond, 0); /* print amg solution info */
      HYPRE_BoomerAMGSetCoarsenType(precond, 6);
      HYPRE_BoomerAMGSetRelaxType(precond, 6); /* Sym G.S./Jacobi hybrid */
      HYPRE_BoomerAMGSetNumSweeps(precond, 1);
      HYPRE_BoomerAMGSetTol(precond, 0.0); /* conv. tolerance zero */
      HYPRE_BoomerAMGSetMaxIter(precond, 1); /* do only one iteration! */
      HYPRE_BoomerAMGSetStrongThreshold(precond,.5); /* strong threshold=0.5 for 3d problem*/
      /* Set the PCG preconditioner */
      HYPRE_BiCGSTABSetPrecond(solver, (HYPRE_PtrToSolverFcn) HYPRE_BoomerAMGSolve,(HYPRE_PtrToSolverFcn)HYPRE_BoomerAMGSetup,precond);
    }
  /* Now setup and solve! */
  HYPRE_ParCSRBiCGSTABSetup(solver,parcsr_A,par_b,par_x);
  HYPRE_ParCSRBiCGSTABSolve(solver,parcsr_A,par_b,par_x);
  /*Set Tolerance*/
  HYPRE_BiCGSTABSetTol(solver,epsilon_);
  /* Get solve info */
   HYPRE_BiCGSTABGetNumIterations(solver, &num_iteration_);
  /* Destroy solver */
  HYPRE_ParCSRBiCGSTABDestroy(solver);
  /* Destroy precon(if any) */
  if (param_->PreCond()==1) HYPRE_BoomerAMGDestroy(precond);
}


void poisson::FFT(tensor1 &C,tensor0 &P,tensor0 &RHS)
{
  center_coeff.Equal_I_F2C(C); //compute average valuie of coefficient at cell center
  center_coeff.Equal_Divide(-1.,center_coeff); //compute minus inverse of cell center coefficients
  int i(0);
  double error(100);
  while((i<param_->Iteration())&&(error>epsilon_)) //iteration loop
    {
      dummy.Equal_Grad_C2F(P);
      dummy*=C;
      RHS_new.Equal_Div_F2C(dummy); //RHS_new=div(C grad(P))
      RHS_new-=RHS;
      RHS_new*=center_coeff; //RHS_new is the new RHS
      CCP_FFT(plan,fft_data,size_->Nx_tot(),size_->Ny_tot(),size_->Nz_tot(),size_->size_tot(),RHS_new,dummy.x,size_->bs(),size_->dx(),size_->dy(),size_->dz()); // dp is stored in dummy.x ,assumed P_final=P_guess+dp
      P+=dummy.x; //correct P_guess
      P.Update_Ghosts();
      error=dummy.x.max_abs();
      i++;
    }
  num_iteration_=i;
}

void poisson::CCP_FFT(fft_plan_3d *plan,FFT_DATA *data,int nx,int ny,int nz,int tot,tensor0& RHS,tensor0 &P,int bs,double dx,double dy,double dz)
{
  double TWO_PI(2*3.141592653589793);

  double Two_PI_Over_Lx = TWO_PI/size_->Lx();
  double Two_PI_Over_Ly = TWO_PI/size_->Ly();
  double Two_PI_Over_Lz = TWO_PI/size_->Lz();

  //load data
  int count=0;
  for (int k=in_klo;k<=in_khi;k++)
    for (int j=in_jlo;j<=in_jhi;j++)
      for (int i=in_ilo;i<=in_ihi;i++)
	{
	  data[count].im=0;
	  data[count++].re=RHS(i-in_ilo+bs,j-in_jlo+bs,k-in_klo+bs);
	}
  //Take FT of RHS
  fft_3d(data,data,1,plan);
  //divide by Modified Wave Number
  double dx2=2./(dx*dx);
  double dy2=2./(dy*dy);
  double dz2=2./(dz*dz);
  count=0;
  double ii,jj,kk;
  for (int k=in_klo;k<=in_khi;k++)
    for (int j=in_jlo;j<=in_jhi;j++)
      for (int i=in_ilo;i<=in_ihi;i++)
	{
	  if (i+j+k>0) 
	    {
	      ii=((i<nx/2)?i:i-nx)*Two_PI_Over_Lx;
	      jj=((j<ny/2)?j:j-ny)*Two_PI_Over_Ly;
	      kk=((k<nz/2)?k:k-nz)*Two_PI_Over_Lz;
	      data[count].im/=((dx2)*(1-cos(ii*dx))+(dy2)*(1-cos(jj*dy))+(dz2)*(1-cos(kk*dz)));
	      data[count].re/=((dx2)*(1-cos(ii*dx))+(dy2)*(1-cos(jj*dy))+(dz2)*(1-cos(kk*dz)));
	    }
	  count++;
	}
  //IFT
  fft_3d(data,data,-1,plan);
  //load back P!
  count=0;
  for (int k=in_klo;k<=in_khi;k++)
    for (int j=in_jlo;j<=in_jhi;j++)
      for (int i=in_ilo;i<=in_ihi;i++)
	{
	  P(i-in_ilo+bs,j-in_jlo+bs,k-in_klo+bs)=(-1)*data[count++].re/(tot);
	}
}

