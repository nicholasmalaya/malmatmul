

//
// CORE COMPUTE KERNEL for Strassen
// We are hand-coding only one level of depth
//
template <const int TS> void GPU_STRASSEN(hc::array_view<const double,2> a, hc::array_view<const double,2> b, hc::array_view<double,2> c, long N)
{
  //
  // if you are here, we KNOW we are divisible by two!
  //
  long Nh = N/2;
  hc::extent<2> ex(Nh,Nh);
  hc::tiled_extent<2> t_ex = ex.tile_with_dynamic(TS,TS,TS);

  // Strassen Temporary Matrices
  hc::array_view<double,2> P1(ex);
  hc::array_view<double,2> P2(ex);
  hc::array_view<double,2> P3(ex);
  hc::array_view<double,2> P4(ex);
  hc::array_view<double,2> P5(ex);
  hc::array_view<double,2> P6(ex);
  hc::array_view<double,2> P7(ex);
  
  c.discard_data();
  hc::parallel_for_each(t_ex, [=](hc::tiled_index<2> t_idx) [[hc]]
		{
		  
		  // local 
		  int row  = t_idx.local[0];
		  int col  = t_idx.local[1];
		  
		  // global
		  int rowG = t_idx.global[0];
		  int colG = t_idx.global[1];

		  // -----------------------------
		  // Calculate P1!
		  // -----------------------------
		  // can we add tiling here?
		  // need four tiles!
		  double sum1 = 0;		  
		  for(long i = 0; i < Nh; i += TS)
		    {
		      tile_static double locA11[TS][TS]; 
		      tile_static double locA22[TS][TS]; 
		      tile_static double locB11[TS][TS];
		      tile_static double locB22[TS][TS];
		      locA11[row][col] = a(rowG, col + i);
		      locB11[row][col] = b(row + i, colG);
		      locA22[row][col] = a(rowG + Nh, col + i + Nh);
		      locB22[row][col] = b(row + i + Nh, colG + Nh);

		      // threads in tile all wait until locA,locB are filled.  
		      t_idx.barrier.wait();
		      for (long k = 0; k < TS; k++)
			{
			  sum1 += (locA11[row][k]+locA22[row][k])*(locB11[k][col]+locB22[k][col]); 
			}
		      // all threads wait until sums are calculated. 
		      t_idx.barrier.wait();
		      
		    }  
		  P1[t_idx] = sum1;

		  // -----------------------------
		  // Calculate P2!
		  // -----------------------------
		  double sum2 = 0;		  
		  for(long i = 0; i < Nh; i += TS)
		    {
		      tile_static double locA21[TS][TS]; 
		      tile_static double locA22[TS][TS]; 
		      tile_static double locB11[TS][TS];
		      locA21[row][col] = a(rowG + Nh, col + i);
		      locA22[row][col] = a(rowG + Nh, col + i + Nh);
		      locB11[row][col] = b(row + i, colG);

		      // threads in tile all wait until locA,locB are filled.  
		      t_idx.barrier.wait();
		      for (long k = 0; k < TS; k++)
			{
			  sum2 += (locA21[row][k]+locA22[row][k])*(locB11[k][col]); 
			}
		      // all threads wait until sums are calculated. 
		      t_idx.barrier.wait();
		      
		    }  
		  P2[t_idx] = sum2;

		  // -----------------------------
		  // Calculate P3!
		  // -----------------------------
		  double sum3 = 0;		  
		  for(long i = 0; i < Nh; i += TS)
		    {
		      tile_static double locA11[TS][TS]; 
		      tile_static double locB12[TS][TS]; 
		      tile_static double locB22[TS][TS];
		      locA11[row][col] = a(rowG, col + i);
		      locB12[row][col] = b(row + i, colG + Nh);
		      locB22[row][col] = b(row + i + Nh, colG + Nh);

		      // threads in tile all wait until locA,locB are filled.  
		      t_idx.barrier.wait();
		      for (long k = 0; k < TS; k++)
			{
			  sum3 += (locA11[row][k])*(locB12[k][col]-locB22[k][col]); 
			}
		      // all threads wait until sums are calculated. 
		      t_idx.barrier.wait();
		      
		    }  
		  P3[t_idx] = sum3;
		  
		  // -----------------------------
		  // Calculate P4!
		  // -----------------------------
		  double sum4 = 0;		  
		  for(long i = 0; i < Nh; i += TS)
		    {
		      tile_static double locA22[TS][TS]; 
		      tile_static double locB21[TS][TS]; 
		      tile_static double locB11[TS][TS];
		      locA22[row][col] = a(rowG + Nh, col + i + Nh);
		      locB21[row][col] = b(row + i + Nh, colG);
		      locB11[row][col] = b(row + i, colG);
		      
		      // threads in tile all wait until locA,locB are filled.  
		      t_idx.barrier.wait();
		      for (long k = 0; k < TS; k++)
			{
			  sum4 += (locA22[row][k])*(locB21[k][col]-locB11[k][col]); 
			}
		      // all threads wait until sums are calculated. 
		      t_idx.barrier.wait();
		      
		    }  
		  P4[t_idx] = sum4;

		  // -----------------------------
		  // Calculate P5!
		  // -----------------------------
		  double sum5 = 0;		  
		  for(long i = 0; i < Nh; i += TS)
		    {
		      tile_static double locA11[TS][TS]; 
		      tile_static double locA12[TS][TS]; 
		      tile_static double locB22[TS][TS];
		      locA11[row][col] = a(rowG, col + i);
		      locA12[row][col] = a(rowG, col + i + Nh);
		      locB22[row][col] = b(row + i + Nh, colG + Nh);
		      
		      // threads in tile all wait until locA,locB are filled.  
		      t_idx.barrier.wait();
		      for (long k = 0; k < TS; k++)
			{
			  sum5 += (locA11[row][k]+locA12[row][k])*(locB22[k][col]); 
			}
		      // all threads wait until sums are calculated. 
		      t_idx.barrier.wait();
		      
		    }  
		  P5[t_idx] = sum5;

		  // -----------------------------
		  // Calculate P6!
		  // -----------------------------
		  double sum6 = 0;		  
		  for(long i = 0; i < Nh; i += TS)
		    {
		      tile_static double locA21[TS][TS]; 
		      tile_static double locA11[TS][TS]; 
		      tile_static double locB11[TS][TS];
		      tile_static double locB12[TS][TS];
		      locA21[row][col] = a(rowG + Nh, col + i);
		      locA11[row][col] = a(rowG, col + i);
		      locB11[row][col] = b(row + i, colG);
		      locB12[row][col] = b(row + i, colG + Nh);
		      
		      // threads in tile all wait until locA,locB are filled.  
		      t_idx.barrier.wait();
		      for (long k = 0; k < TS; k++)
			{
			  sum6 += (locA21[row][k]-locA11[row][k])*(locB11[k][col]+locB12[k][col]); 
			}
		      // all threads wait until sums are calculated. 
		      t_idx.barrier.wait();
		      
		    }  
		  P6[t_idx] = sum6;
		  
		  // -----------------------------
		  // Calculate P7!
		  // -----------------------------
		  double sum7 = 0;		  
		  for(long i = 0; i < Nh; i += TS)
		    {
		      tile_static double locA12[TS][TS]; 
		      tile_static double locA22[TS][TS]; 
		      tile_static double locB21[TS][TS];
		      tile_static double locB22[TS][TS];
		      locA12[row][col] = a(rowG, col + i + Nh);
		      locA22[row][col] = a(rowG + Nh, col + i + Nh);
		      locB21[row][col] = b(row + i + Nh, colG);
		      locB22[row][col] = b(row + i + Nh, colG + Nh);
		      
		      // threads in tile all wait until locA,locB are filled.  
		      t_idx.barrier.wait();
		      for (long k = 0; k < TS; k++)
			{
			  sum7 += (locA12[row][k]-locA22[row][k])*(locB21[k][col]+locB22[k][col]); 
			}
		      // all threads wait until sums are calculated. 
		      t_idx.barrier.wait();
		      
		    }  
		  P7[t_idx] = sum7;
		  
		  //
		  // I think I need a synchronization here
		  //
		  P1.synchronize();
		  P2.synchronize();
		  P3.synchronize();
		  P4.synchronize();
		  P5.synchronize();
		  P6.synchronize();
		  P7.synchronize();
		  
		  // -----------------------------
		  // Final Matrix Assembly
		  // -----------------------------
		  // 
		  // try tiling?
		  //
		  // double sum7 = 0;		  
		  // for(long i = 0; i < Nh; i += TS)
		  //   {
		  //     tile_static double locP1[TS][TS]; 
		  //     tile_static double locP2[TS][TS]; 
		  //     tile_static double locP3[TS][TS];
		  //     tile_static double locP4[TS][TS];
		  //     tile_static double locP5[TS][TS];
		  //     tile_static double locP6[TS][TS];
		  //     tile_static double locP7[TS][TS];
		  //     locA12[row][col] = a(rowG, col + i + Nh);
		  //     locA22[row][col] = a(rowG + Nh, col + i + Nh);
		  //     locB21[row][col] = b(row + i + Nh, colG);
		  //     locB22[row][col] = b(row + i + Nh, colG + Nh);


		  // stupid loop
		  // C_11
		  c[t_idx] = P1[t_idx]+P4[t_idx]-P5[t_idx]+P7[t_idx];

		  // C_12
		  c[t_idx.global[0]][t_idx.global[1]+Nh] = P3[t_idx]+P5[t_idx];

		  // C_21
		  c[t_idx.global[0]+Nh][t_idx.global[1]] = P2[t_idx]+P4[t_idx];
		  
		  // C_22
		  c[t_idx.global[0]+Nh][t_idx.global[1]+Nh] = P1[t_idx]+P3[t_idx]-P2[t_idx]+P6[t_idx];
		  

		  // if( (rowG < N/2 ) and (colG < N/2 ) )              // C_11
		  //   {
		  //     // C11 = P1 + P4 - P5 + P7
		  //     c[t_idx] = a(rowG,colG) + b(rowG,colG);
		  //   }  
		  // else if( (rowG < N/2 ) and (colG > N/2 - 1 ) )     // C_12
		  //   {
		  //     // C12 = P3 + P5
		  //     c[t_idx] = a(rowG,colG) + b(rowG,colG);
		  //   }
		  // else if( (rowG > N/2 - 1 ) and (colG < N/2 ) )     // C_21
		  //   {
		  //     // C21 = P2 + P4
		  //     c[t_idx] = a(rowG,colG) + b(rowG,colG);
		  //   }
		  // else if( (rowG > N/2 - 1 ) and (colG > N/2 - 1 ) ) // C_22
		  //   {
		  //     // C22 = P1 + P3 - P2 + P6
		  //     c[t_idx] = a(rowG,colG) + b(rowG,colG);
		  //   }
		  
		});
  c.synchronize();

}
//
// https://github.com/arbenson/fast-matmul/blob/master/codegen/algorithms/strassen
//

//
// CORE COMPUTE KERNEL for Strassen (OPTIMIZED)
// We are hand-coding only one level of depth
//
template <const int TS> void GPU_STRASSEN_OPT(hc::array_view<const double,2> a, hc::array_view<const double,2> b, hc::array_view<double,2> c, long N)
{
  //
  // if you are here, we KNOW we are divisible by two!
  //
  long Nh = N/2;
  hc::extent<2> ex(Nh,Nh);
  hc::tiled_extent<2> t_ex = ex.tile_with_dynamic(TS,TS,TS);

  // Strassen Temporary Matrices
  hc::array_view<double,2> P1(ex);
  hc::array_view<double,2> P2(ex);
  hc::array_view<double,2> P3(ex);
  hc::array_view<double,2> P4(ex);
  hc::array_view<double,2> P5(ex);
  hc::array_view<double,2> P6(ex);
  hc::array_view<double,2> P7(ex);
  
  //c.discard_data();
  hc::parallel_for_each(t_ex, [=](hc::tiled_index<2> t_idx) [[hc]]
		{
		  
		  // local 
		  int row  = t_idx.local[0];
		  int col  = t_idx.local[1];
		  
		  // global
		  int rowG = t_idx.global[0];
		  int colG = t_idx.global[1];

		  // -----------------------------
		  // Calculate P1!
		  // -----------------------------
		  // can we add tiling here?
		  // need four tiles!
		  double sum1 = 0;		  
		  for(long i = 0; i < Nh; i += TS)
		    {
		      tile_static double locA11[TS][TS]; 
		      tile_static double locA22[TS][TS]; 
		      tile_static double locB11[TS][TS];
		      tile_static double locB22[TS][TS];
		      locA11[row][col] = a(rowG, col + i);
		      locB11[row][col] = b(row + i, colG);
		      locA22[row][col] = a(rowG + Nh, col + i + Nh);
		      locB22[row][col] = b(row + i + Nh, colG + Nh);

		      // threads in tile all wait until locA,locB are filled.  
		      t_idx.barrier.wait();
		      for (long k = 0; k < TS; k++)
			{
			  sum1 += (locA11[row][k]+locA22[row][k])*(locB11[k][col]+locB22[k][col]); 
			}
		      // all threads wait until sums are calculated. 
		      t_idx.barrier.wait();
		      
		    }  
		  P1[t_idx] = sum1;

		  // -----------------------------
		  // Calculate P2!
		  // -----------------------------
		  double sum2 = 0;		  
		  for(long i = 0; i < Nh; i += TS)
		    {
		      tile_static double locA21[TS][TS]; 
		      tile_static double locA22[TS][TS]; 
		      tile_static double locB11[TS][TS];
		      locA21[row][col] = a(rowG + Nh, col + i);
		      locA22[row][col] = a(rowG + Nh, col + i + Nh);
		      locB11[row][col] = b(row + i, colG);

		      // threads in tile all wait until locA,locB are filled.  
		      t_idx.barrier.wait();
		      for (long k = 0; k < TS; k++)
			{
			  sum2 += (locA21[row][k]+locA22[row][k])*(locB11[k][col]); 
			}
		      // all threads wait until sums are calculated. 
		      t_idx.barrier.wait();
		      
		    }  
		  P2[t_idx] = sum2;

		  // -----------------------------
		  // Calculate P3!
		  // -----------------------------
		  double sum3 = 0;		  
		  for(long i = 0; i < Nh; i += TS)
		    {
		      tile_static double locA11[TS][TS]; 
		      tile_static double locB12[TS][TS]; 
		      tile_static double locB22[TS][TS];
		      locA11[row][col] = a(rowG, col + i);
		      locB12[row][col] = b(row + i, colG + Nh);
		      locB22[row][col] = b(row + i + Nh, colG + Nh);

		      // threads in tile all wait until locA,locB are filled.  
		      t_idx.barrier.wait();
		      for (long k = 0; k < TS; k++)
			{
			  sum3 += (locA11[row][k])*(locB12[k][col]-locB22[k][col]); 
			}
		      // all threads wait until sums are calculated. 
		      t_idx.barrier.wait();
		      
		    }  
		  P3[t_idx] = sum3;
		  
		  // -----------------------------
		  // Calculate P4!
		  // -----------------------------
		  double sum4 = 0;		  
		  for(long i = 0; i < Nh; i += TS)
		    {
		      tile_static double locA22[TS][TS]; 
		      tile_static double locB21[TS][TS]; 
		      tile_static double locB11[TS][TS];
		      locA22[row][col] = a(rowG + Nh, col + i + Nh);
		      locB21[row][col] = b(row + i + Nh, colG);
		      locB11[row][col] = b(row + i, colG);
		      
		      // threads in tile all wait until locA,locB are filled.  
		      t_idx.barrier.wait();
		      for (long k = 0; k < TS; k++)
			{
			  sum4 += (locA22[row][k])*(locB21[k][col]-locB11[k][col]); 
			}
		      // all threads wait until sums are calculated. 
		      t_idx.barrier.wait();
		      
		    }  
		  P4[t_idx] = sum4;

		  // -----------------------------
		  // Calculate P5!
		  // -----------------------------
		  double sum5 = 0;		  
		  for(long i = 0; i < Nh; i += TS)
		    {
		      tile_static double locA11[TS][TS]; 
		      tile_static double locA12[TS][TS]; 
		      tile_static double locB22[TS][TS];
		      locA11[row][col] = a(rowG, col + i);
		      locA12[row][col] = a(rowG, col + i + Nh);
		      locB22[row][col] = b(row + i + Nh, colG + Nh);
		      
		      // threads in tile all wait until locA,locB are filled.  
		      t_idx.barrier.wait();
		      for (long k = 0; k < TS; k++)
			{
			  sum5 += (locA11[row][k]+locA12[row][k])*(locB22[k][col]); 
			}
		      // all threads wait until sums are calculated. 
		      t_idx.barrier.wait();
		      
		    }  
		  P5[t_idx] = sum5;

		  // -----------------------------
		  // Calculate P6!
		  // -----------------------------
		  double sum6 = 0;		  
		  for(long i = 0; i < Nh; i += TS)
		    {
		      tile_static double locA21[TS][TS]; 
		      tile_static double locA11[TS][TS]; 
		      tile_static double locB11[TS][TS];
		      tile_static double locB12[TS][TS];
		      locA21[row][col] = a(rowG + Nh, col + i);
		      locA11[row][col] = a(rowG, col + i);
		      locB11[row][col] = b(row + i, colG);
		      locB12[row][col] = b(row + i, colG + Nh);
		      
		      // threads in tile all wait until locA,locB are filled.  
		      t_idx.barrier.wait();
		      for (long k = 0; k < TS; k++)
			{
			  sum6 += (locA21[row][k]-locA11[row][k])*(locB11[k][col]+locB12[k][col]); 
			}
		      // all threads wait until sums are calculated. 
		      t_idx.barrier.wait();
		      
		    }  
		  P6[t_idx] = sum6;
		  
		  // -----------------------------
		  // Calculate P7!
		  // -----------------------------
		  double sum7 = 0;		  
		  for(long i = 0; i < Nh; i += TS)
		    {
		      tile_static double locA12[TS][TS]; 
		      tile_static double locA22[TS][TS]; 
		      tile_static double locB21[TS][TS];
		      tile_static double locB22[TS][TS];
		      locA12[row][col] = a(rowG, col + i + Nh);
		      locA22[row][col] = a(rowG + Nh, col + i + Nh);
		      locB21[row][col] = b(row + i + Nh, colG);
		      locB22[row][col] = b(row + i + Nh, colG + Nh);
		      
		      // threads in tile all wait until locA,locB are filled.  
		      t_idx.barrier.wait();
		      for (long k = 0; k < TS; k++)
			{
			  sum7 += (locA12[row][k]-locA22[row][k])*(locB21[k][col]+locB22[k][col]); 
			}
		      // all threads wait until sums are calculated. 
		      t_idx.barrier.wait();
		      
		    }  
		  P7[t_idx] = sum7;
		  
		  //
		  // I think I need a synchronization here
		  //
		  // P1.synchronize();
		  // P2.synchronize();
		  // P3.synchronize();
		  // P4.synchronize();
		  // P5.synchronize();
		  // P6.synchronize();
		  // P7.synchronize();
		  
		  // -----------------------------
		  // Final Matrix Assembly
		  // -----------------------------
		  // 
		  // probably faster to tile in rows... 
		  //	  
		  for(long i = 0; i < Nh; i += TS)
		    {
		      tile_static double locP1[TS][TS]; 
		      tile_static double locP2[TS][TS]; 
		      tile_static double locP3[TS][TS];
		      tile_static double locP4[TS][TS];
		      tile_static double locP5[TS][TS];
		      tile_static double locP6[TS][TS];
		      tile_static double locP7[TS][TS];
		      locP1[row][col] = P1[t_idx];
		      locP2[row][col] = P2[t_idx];
		      locP3[row][col] = P3[t_idx];
		      locP4[row][col] = P4[t_idx];
		      locP5[row][col] = P5[t_idx];
		      locP6[row][col] = P6[t_idx];
		      locP7[row][col] = P7[t_idx];
		      
		      // C_11
		      c[t_idx] = locP1[row][col]+locP4[row][col]-locP5[row][col]+locP7[row][col];
		      
		      // C_12
		      c[t_idx.global[0]][t_idx.global[1]+Nh] = locP3[row][col]+locP5[row][col];
		      
		      // C_21
		      c[t_idx.global[0]+Nh][t_idx.global[1]] = locP2[row][col]+locP4[row][col];
		      
		      // C_22
		      c[t_idx.global[0]+Nh][t_idx.global[1]+Nh] = locP1[row][col]+locP3[row][col]-locP2[row][col]+locP6[row][col];
		    }

		  
		});
  c.synchronize();

}
//
// END KERNEL
//


//
// GPU ADD KERNEL
//
void GPU_ADD(hc::array_view<const double,2> a, hc::array_view<const double,2> b, hc::array_view<double,2> c)
{  

  c.discard_data();
  hc::parallel_for_each(c.get_extent(), [=](hc::index<2> idx) [[hc]]
		{
		  c[idx] = a[idx] + b[idx];
   		});
  c.synchronize();

}

//
// GPU SUBTRACT KERNEL
//
void GPU_SUB(hc::array_view<const double,2> a, hc::array_view<const double,2> b, hc::array_view<double,2> c)
{  

  c.discard_data();
  hc::parallel_for_each(c.get_extent(), [=](hc::index<2> idx) [[hc]]
		{
		  c[idx] = a[idx] - b[idx];
   		});
  c.synchronize();

}
 
