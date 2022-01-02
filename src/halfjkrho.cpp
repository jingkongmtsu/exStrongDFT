/**
 * CPP files corresponding to the halfjkrho.h
 * \author  Fenglai Liu
 */
#include <iostream>
#include <cstdio>
#include "excep.h"
//#include "xcvar.h"
//#include "shellprop.h"
//#include "batchgrid.h"
//#include "shell.h"
//#include "cptrans.h"
//#include "blas.h"
#include "matrix.h"
//#include "sigatombasis.h"
//#include "xcintsinfor.h"
//#include "denmtrx.h"
//#include "dftmatrix.h"
//#include "espints.h"
//#include "integraljobs.h"
//#include "batchbasis.h"
//#include "sigshellpairinfor.h"
//#include "halfjkrho.h"
using namespace excep;
//using namespace xcvar;
//using namespace batchgrid;
//using namespace shell;
//using namespace cptrans;
//using namespace shellprop;
//using namespace blas;
using namespace matrix;
//using namespace sigatombasis;
//using namespace xcintsinfor;
//using namespace denmtrx;
//using namespace dftmatrix;
//using namespace espints;
//using namespace integraljobs;
//using namespace batchbasis;
//using namespace sigshellpairinfor;
//using namespace halfjkrho;
using namespace std;

/**
 * halfExRho: HFX energy density in a matrix format for each spin.
 *
 * 
*/

void doHalfJKRho(SpinMatrix& halfExRho, const XCIntJobInfor& infor, const XCVar& xcvar, const MolShell& ms, 
		const BatchGrid& grid, const SigAtomBasis& sigList, const BatchBasis& basis,
		const DenMtrx& den, const DoubleVec& cartDen, const SigMolShellPairInfor& spData)
{
	// do we do exchange matrix or producing the exrho?
	SpinMatrix PmunuPhinu(den.getNSpin());
	if (infor.doNumericalK() || xcvar.hasExRho()) {

		// initialize the result
		PmunuPhinu.init(ms.getNCarBas(),nGrids);

		// we need to convert the density matrix into the sig order
		// the result sigDen is in dimension of (nSigBas,nWholeBas)
		DFTMatrix sigDen(sigList,ms,den.getNSpin());
		for(UInt iSpin=0; iSpin<den.getNSpin(); iSpin++) {
			sigDen.intoSigOrder(iSpin,den.getMtrx(iSpin),sigList); 
		}

		// now let's appened the C2P transformation as well as basis set
		// scaling into the sigDen
		// after this step sigDen is in (nSigBas,nWholeCartBas)
		const GlobalInfor& ginfor = infor.getGlobalInfor();
		CPTransBasisNorm trans(ginfor,C2P_WITH_L00,DO_SCALE,WITH_MATRIX_TRANSPOSE,CP_WITH_COL);
		for(UInt iSpin=0; iSpin<sigDen.getNSpin(); iSpin++) {
			Mtrx& D = sigDen.getMtrx(iSpin);
			trans.transform(ms,ms,D);
		}

		// let's make a local copy of Phi0, and transpose it
		// so that the later matrix multiplication can be easier
		// after this step Phi0 is in dimension of (nSigBas,nG)
		Mtrx Phi0(basis.getPhi(0));
		Phi0.transpose(false);

		// now let's combine sigDen, which is Pmunu, with the Phinu value
		// which is the data in  Phi0
		// PmunuPhinu = sigDen*Phi
		// the result dimension is (nWholeCartBas,nGrids)
		for(UInt iSpin=0; iSpin<sigDen.getNSpin(); iSpin++) {
			const Mtrx& D = sigDen.getMtrx(iSpin);
			Mtrx& T = PmunuPhinu.getMtrx(iSpin);
			T.mult(D,Phi0,true,false,ONE,ZERO);
		}
	}

	// do we do Coulomb matrix?
	// if we do it we need to get the density scaled with weights
	// so that we can use it to form the halfCouRhoVecMatrix
	DoubleVec wRho;

	// extract the grid points out
	DoubleVec pts(nGrids*3);
	const Double* xyz = grid.getGridCoord(); 
	for(UInt i=0; i<nGrids; i++) {
		pts[3*i  ] = xyz[3*i  ];
		pts[3*i+1] = xyz[3*i+1];
		pts[3*i+2] = xyz[3*i+2];
	}

	// set the job
	UInt job = infor.getIntJob();
	if (isDFTJob(job) && xcvar.hasExRho()) {
		job = EX_DENSITY;
	}

	// get the gints infor
	const GIntsInfor& gintsInfor = infor.getGIntsInfor();

	// now let's form the half exrho/Coulomb rho
	ESPInts excrho(gintsInfor,job);
	excrho.doMtrx(infor,ms,spData,pts,PmunuPhinu,cartDen,wRho,
			halfExRho,halfCouRhoVec,halfCouRhoVecMatrix);
}

/**
 * halfExRho: HFX hole at a grid point for one svalue in a matrix format for each spin.
 * grid: It should contain one grid point only.
 * sValue: the interelectronic distance.
 * 
*/


void doHalfJKRhoXHole(const XCIntJobInfor& infor, const XCVar& xcvar, const MolShell& ms, 
		const BatchGrid& grid, const SigAtomBasis& sigList, const BatchBasis& basis,
			    const DenMtrx& den, const DoubleVec& cartDen, const SigMolShellPairInfor& spData, Double sValue)//yw
{
	// do we do exchange matrix or producing the exrho?
	SpinMatrix PmunuPhinu(den.getNSpin());
	if (infor.doNumericalK() || xcvar.hasExRho()) {

		// initialize the result
		PmunuPhinu.init(ms.getNCarBas(),nGrids);

		// we need to convert the density matrix into the sig order
		// the result sigDen is in dimension of (nSigBas,nWholeBas)
		DFTMatrix sigDen(sigList,ms,den.getNSpin());
		for(UInt iSpin=0; iSpin<den.getNSpin(); iSpin++) {
			sigDen.intoSigOrder(iSpin,den.getMtrx(iSpin),sigList); 
		}

		// now let's appened the C2P transformation as well as basis set
		// scaling into the sigDen
		// after this step sigDen is in (nSigBas,nWholeCartBas)
		const GlobalInfor& ginfor = infor.getGlobalInfor();
		CPTransBasisNorm trans(ginfor,C2P_WITH_L00,DO_SCALE,WITH_MATRIX_TRANSPOSE,CP_WITH_COL);
		for(UInt iSpin=0; iSpin<sigDen.getNSpin(); iSpin++) {
			Mtrx& D = sigDen.getMtrx(iSpin);
			trans.transform(ms,ms,D);
		}

		// let's make a local copy of Phi0, and transpose it
		// so that the later matrix multiplication can be easier
		// after this step Phi0 is in dimension of (nSigBas,nG)
		Mtrx Phi0(basis.getPhi(0));
		Phi0.transpose(false);

		// now let's combine sigDen, which is Pmunu, with the Phinu value
		// which is the data in  Phi0
		// PmunuPhinu = sigDen*Phi
		// the result dimension is (nWholeCartBas,nGrids)
		for(UInt iSpin=0; iSpin<sigDen.getNSpin(); iSpin++) {
			const Mtrx& D = sigDen.getMtrx(iSpin);
			Mtrx& T = PmunuPhinu.getMtrx(iSpin);
			T.mult(D,Phi0,true,false,ONE,ZERO);
		}

	}

	DoubleVec wRho;

	// extract the grid points out
	DoubleVec pts(nGrids*3);
	const Double* xyz = grid.getGridCoord(); 
	for(UInt i=0; i<nGrids; i++) {
		pts[3*i  ] = xyz[3*i  ];
		pts[3*i+1] = xyz[3*i+1];
		pts[3*i+2] = xyz[3*i+2];
	}

	// set the job
	UInt job = infor.getIntJob();
	if (isDFTJob(job) && xcvar.hasExRho()) {
		job = EX_DENSITY;
	}

	// get the gints infor
	const GIntsInfor& gintsInfor = infor.getGIntsInfor();

	// now let's form the half exrho/Coulomb rho
	ESPInts excrho(gintsInfor,job);
	excrho.doMtrxXHole(infor,ms,spData,pts,PmunuPhinu,cartDen,wRho,
		      halfExRho,halfCouRhoVec,halfCouRhoVecMatrix, sValue);//yw

}

