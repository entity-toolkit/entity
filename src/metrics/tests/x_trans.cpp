M.x_Code2Phys(x_Code, x_Phys);
wrongs += not equal<Dim::_3D>(x_Cart, x_Phys);
M.x_Cart2Code(x_Cart, x_Code2);
wrongs += not equal<Dim::_3D>(x_Code, x_Code2);
M.x_Phys2Code(x_Phys, x_Code2);
wrongs += not equal<Dim::_3D>(x_Code, x_Code2);

// code <-> sph
Kokkos::parallel_reduce(
  "sph",
  CreateRangePolicy<Dim::_3D>({ N_GHOSTS, N_GHOSTS, N_GHOSTS },
                              { 128 + N_GHOSTS, 64 + N_GHOSTS, 32 + N_GHOSTS }),
  Lambda(index_t i1, index_t i2, index_t i3, unsigned long& wrongs) {
    const coord_t<Dim::_3D> x_Code { COORD(i1) + HALF,
                                     COORD(i2) + HALF,
                                     COORD(i3) + HALF };
    coord_t<Dim::_3D>       x_Cart { ZERO, ZERO, ZERO };
    coord_t<Dim::_3D>       x_Sph { ZERO, ZERO, ZERO };
    coord_t<Dim::_3D>       x_Code2 { ZERO, ZERO, ZERO };
    M.x_Code2Cart(x_Code, x_Cart);
    M.x_Code2Sph(x_Code, x_Sph);
    M.x_Sph2Code(x_Sph, x_Code2);
    
      // !TODO: why is backward conversion so innacurate?
      // const auto a = not equal<Dim::_3D>(x_Code, x_Code2, 1000.0);
      // if (a) {
      //   printf("%.12f %.12f %.12f\n", x_Sph[0], x_Sph[1], x_Sph[2]);
      //   printf("%.12f %.12f %.12f\n", x_Cart[0], x_Cart[1], x_Cart[2]);
      //   printf("%.12f %.12f %.12f\n", x_Code[0], x_Code[1], x_Code[2]);
      // }
      wrongs += not equal<Dim::_3D>(x_Code, x_Code2, 1000.0);
    }
  },
  all_wrongs);
