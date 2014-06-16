/*
   Copyright (c) 2009-2014, Jack Poulson
   Copyright (c) 2014,      Sayan Ghosh
   Copyright (c) 2014,      Jeff Hammond
   All rights reserved.

   This file is part of Elemental and is under the BSD 2-Clause License, 
   which can be found in the LICENSE file in the root directory, or at 
   http://opensource.org/licenses/BSD-2-Clause
*/
// NOTE: It is possible to simply include "El.hpp" instead
#include "El-lite.hpp"
#include EL_ONES_INC
using namespace El;

int
main (int argc, char *argv[])
{
  Initialize(argc, argv);
  mpi::Comm comm = mpi::COMM_WORLD;
  const Int commRank = mpi::Rank(comm);
  const Int commSize = mpi::Size(comm);

  try
  {
    const Int  m       = Input("--height", "height of matrix", 10);
    const Int  n       = Input("--width", "width of matrix", 10);
    const bool display = Input("--display", "display matrix?", true);
    const bool print   = Input("--print", "print matrix?", false);
    ProcessInput();
    PrintInputReport();

    DistMatrix < double >A;
    Ones(A, m, n);

    const Grid& g     = A.Grid();
    const Int   r     = g.Height();
    const Int   c     = g.Width();
    const Int   myRow = g.Row();
    const Int   myCol = g.Col();

    if (commRank == 0) {
      std::cout << "Cells to PE mapping in terms of base_address+offset:" << std::endl;
    }

    for (int j = 0; j < n; j++) {
      for (int i = 0; i < m; i++) {
        if (commRank == A.Owner(i,j) ) {
          const Int colAlign     = (A.ColAlign() + i) % r;
          const Int rowAlign     = (A.RowAlign() + j) % c;
          const Int colShift     = Shift(myRow, colAlign, r);
          const Int rowShift     = Shift(myCol, rowAlign, c);
          const Int localHeight  = Length(n, colShift, r);
          const Int localWidth   = Length(m, rowShift, c);
          const Int iLocalOffset = Length(i, A.ColShift(), r);
          const Int jLocalOffset = Length(j, A.RowShift(), c);
          //Buffer(i,j) returns pointer to the portion of the local
          //buffer that stores entry (iLoc,jLoc)
          double *  ACol         = A.Buffer(iLocalOffset, jLocalOffset);
          std::cout << "Patch (" << i << "," << j << ") owned by rank " << commRank
                    << "; local(height,width) = " << localHeight << "," << localWidth
                    << " and patch address = " << ACol << std::endl;
        }
      }
    }
  }
  catch (std::exception & e)
  {
    ReportException(e);
  }

  Finalize();
  return 0;
}
