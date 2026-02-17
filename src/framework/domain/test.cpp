//auto x1
      auto x1_extent = mesh().extent(in::x1); // min and max
      // number of cells in global mesh().metric.convert<1, Crd::Ph, Crd::Cd>(x1_extent.first+(real_t)(i)*dx1)from what basis towhat basis>
      // global coordinates go from x_1 and n_!
      // physical coordinates are same whether we are talking local vs global
      // corners
      
      auto x1_corners = array_t<real_t*>{"x1BinCorners", nx1};
      for (auto i=0; i<nx1; ++1)
      
      const auto &tag = specied.tag:
      const auto &local_metric = domain.mesh().metric
      const auto [x1_min, x1_max] = domain.mesh().metric
      auto advancedSectra_scatter = Kokkos::Experimental::create_scatter_view(advancedSpectra)
      Kokkos::parallel_for(
        "BinParticles",
        species.RangeActiveParticles(),
        Lambda(npart_t p){
          coord_t<D> x_Cd;
          x_Cd = static_cast<real_t>(i1(p))+static_cast<real_t>(dx1(p));

          coord_t<D> x_Ph {ZERO};
          metric.convert<Crd::Cd, Crd::Ph>(X_Cd, x_Ph);
        }
        bin_x1= static_cast<unsigned int>((x_Ph[0]-x1_min)/(x1_max-x1_min)*(real_t)(nx1_1))

        const auto gamma = U2GAMMA(ux1(p), ux2(p), ux3(p));
        const auto bin_energy = static_cast<unsigned int>()

        advancedSpectra(bin_x1, bin_x2, bin_x3)

        );
      )

      // saving the spectra, pass in 4D vector
      unknown = std::vector<adios2::Dim>{adios2::UnknownDim, ...}
      io.DefineVariable<real_t>("dn_%s", {adios2::})
auto var=io.InquireVariable<T>("dn_%s");
// local selectrion
var.SetShape({local_nx1, local_nx2, local_nx3, nenergy});
// offset is stride
var.SetSelection(adios2::Box<adios2::Dims>({offset_nx1, offset_nx2, offset_nx2, 0}, {1,1,1,1}));
writer.Put(var, &advancedSpectra_h.data());