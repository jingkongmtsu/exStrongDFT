/**
 * CPP files corresponding to the batchxcmtrx.h
 * \author  Fenglai Liu and Jing Kong
 */
#include "shell.h"
#include "batchbasis.h"
#include "batchgrid.h"
#include "batchfunc.h"
#include "xcfunc.h"
#include "xcvar.h"
#include "blas.h"
#include "dftderivinfor.h"
#include "halfjkrho.h"
#include "xcintsinfor.h"
#include "sigatombasis.h"
#include "batchxcmtrx.h"
using namespace shell;
using namespace batchbasis;
using namespace batchgrid;
using namespace batchfunc;
using namespace halfjkrho;
using namespace xcfunc;
using namespace xcvar;
using namespace blas;
using namespace xcintsinfor;
using namespace sigatombasis;
using namespace batchxcmtrx;

void batchExrhoMtrx(Mtrx& Fxc, const MolShell& ms, const SigAtomBasis& sigList, 
		const BatchBasis& basis, const HalfJKRho& halfJKRho,
		const BatchFunc& func, const BatchGrid& grid, 
		const XCVar& xcvar, const XCIntJobInfor& infor):DFTMatrix(ms,sigList,infor.getNSpin())
{
	// if no exrho exist, just return
	if (! xcvar.hasExRho()) return;

	// we build the FXC in terms of exchange rho here
	UInt nSigBas= basis.getNSigBasis();
	UInt nGrids = grid.getNGrids();
	Mtrx halfFxc(nGrids,nSigBas);
	const SpinMatrix& halfExRho = halfJKRho.getHalfExRho(); 
	for(UInt iSpin=0; iSpin<getNSpin(); iSpin++) {

		// functional derivatives
		UInt var = EXA;
		if (iSpin == 1) var = EXB;
		UInt varPos = xcvar.getVarPos(var);
		const Double* D1F = func.getD1F(varPos);

		//
		// this part we will combine the functional derivatives
		// with the phi value. Because we do add transpose finally
		// before updating the global result (see xcints.cpp, the 
		// function of doMtrx), therefore in general we need to do
		// halfFxc = F'(r)*phi(r)*0.5
		//
		// However, the exchange energy density in our program
		// is defined as:
		// E_{x}(r) = \sum_{mu,nu,lambda,eta}P_{mu,nu}P_{lambda,eta}
		//            \phi_{mu}(r)\phi_{lambda}(r)\phi_{nu}(r')\phi_{eta}(r')
		// Here it's worthy to note that comparing with exchange energy
		// formula, it does not have 1/2. Therefore, if you sum up
		// all of exchange energy density, it's 2 times bigger than
		// the exchange energy:
		// E_{x} = 1/2*(\sum_{r}w(r)E_{x}(r))
		// w(r) is the weights per grid point.
		//
		// When you do partial derivatives for E_{x}(r) in terms of 
		// P_{mu,nu}, because P_{mu,nu} and P_{lambda,eta} are 
		// symmetrical, it will generate a factor of 2 in the result.
		// This is just like when you do partical derivative for E_{x}
		// in terms of P_{mu,nu}, the exchange Fock matrix part does 
		// not have the 1/2 anymore. 
		//
		// As a result, the 0.5 needed by addTranspose is cancelled
		// with the factor of 2 generating in doing partial derivatives.
		// This is what this long notes about.
		//
		const Mtrx& phi = basis.getPhi(0);
		const Double* wts = grid.getGridWts();
		for(UInt i=0; i<nSigBas; i++) {
			vmul(D1F,phi.getPtr(0,i),halfFxc.getPtr(0,i),nGrids);
			vmul(wts,halfFxc.getPtr(0,i),halfFxc.getPtr(0,i),nGrids);
		}

		// now we need to combine the halfFxc and 
		// the halfExRho together
		const Mtrx& hExRho = halfExRho.getMtrx(iSpin);
		//Mtrx& Fxc = getMtrx(iSpin);
		Fxc.mult(hExRho,halfFxc,true,false,ONE,ZERO);
	}
}

