/*
   Copyright (c) 2009-2014, Jack Poulson
   All rights reserved.

   This file is part of Elemental and is under the BSD 2-Clause License, 
   which can be found in the LICENSE file in the root directory, or at 
   http://opensource.org/licenses/BSD-2-Clause
*/
// NOTE: It is possible to simply include "El.hpp" instead
#include "El-lite.hpp"
#include EL_UNIFORM_INC
using namespace El;

int 
main( int argc, char* argv[] )
{
    Initialize( argc, argv );

    try
    {
        const Int m = Input("--height","height of matrix",10);
        const Int n = Input("--width","width of matrix",10);
        const bool display = Input("--display","display matrix?",true);
        const bool print = Input("--print","print matrix?",false);
        ProcessInput();
        PrintInputReport();

        DistMatrix<double> X;
        Uniform( X, m, n );
        if( display )
            Display( X, "Uniform matrix" );
        if( print )
            Print( X, "Uniform matrix" );
    }
    catch( std::exception& e ) { ReportException(e); }

    Finalize();
    return 0;
}
