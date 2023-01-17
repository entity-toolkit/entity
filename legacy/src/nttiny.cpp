
#elif defined(GRPIC_ENGINE)
          // interpolate and transform to spherical
          // !TODO: mirrors for em0, aux etc
          ntt::vec_t<ntt::Dim3> Dsph {ZERO}, Bsph {ZERO}, D0sph {ZERO}, B0sph {ZERO};
          if ((i >= 0) && (i < sx1) && (j >= 0) && (j < sx2)) {
            if (m_fields_to_plot[f].at(0) == 'D') {
              if (m_fields_to_plot[f].at(1) == '0') {
                real_t Dx1, Dx2, Dx3;
                // interpolate to cell center
                Dx1 = 0.5
                      * (m_sim.meshblock.em0(I, J, ntt::em::ex1)
                         + m_sim.meshblock.em0(I, J + 1, ntt::em::ex1));
                Dx2 = 0.5
                      * (m_sim.meshblock.em0(I, J, ntt::em::ex2)
                         + m_sim.meshblock.em0(I + 1, J, ntt::em::ex2));
                Dx3 = 0.25
                      * (m_sim.meshblock.em0(I, J, ntt::em::ex3)
                         + m_sim.meshblock.em0(I + 1, J, ntt::em::ex3)
                         + m_sim.meshblock.em0(I, J + 1, ntt::em::ex3)
                         + m_sim.meshblock.em0(I + 1, J + 1, ntt::em::ex3));
                m_sim.meshblock.metric.v_Cntr2SphCntrv(
                  {i_ + HALF, j_ + HALF}, {Dx1, Dx2, Dx3}, D0sph);
              } else {
                real_t Dx1, Dx2, Dx3;
                // interpolate to cell center
                Dx1 = 0.5
                      * (m_sim.meshblock.em(I, J, ntt::em::ex1)
                         + m_sim.meshblock.em(I, J + 1, ntt::em::ex1));
                Dx2 = 0.5
                      * (m_sim.meshblock.em(I, J, ntt::em::ex2)
                         + m_sim.meshblock.em(I + 1, J, ntt::em::ex2));
                Dx3 = 0.25
                      * (m_sim.meshblock.em(I, J, ntt::em::ex3)
                         + m_sim.meshblock.em(I + 1, J, ntt::em::ex3)
                         + m_sim.meshblock.em(I, J + 1, ntt::em::ex3)
                         + m_sim.meshblock.em(I + 1, J + 1, ntt::em::ex3));
                m_sim.meshblock.metric.v_Cntr2SphCntrv(
                  {i_ + HALF, j_ + HALF}, {Dx1, Dx2, Dx3}, Dsph);
              }
            } else if (m_fields_to_plot[f].at(0) == 'B') {
              if (m_fields_to_plot[f].at(1) == '0') {
                real_t Bx1, Bx2, Bx3;
                // interpolate to cell center
                Bx1 = 0.5
                      * (m_sim.meshblock.em0(I + 1, J, ntt::em::bx1)
                         + m_sim.meshblock.em0(I, J, ntt::em::bx1));
                Bx2 = 0.5
                      * (m_sim.meshblock.em0(I, J + 1, ntt::em::bx2)
                         + m_sim.meshblock.em0(I, J, ntt::em::bx2));
                Bx3 = m_sim.meshblock.em0(I, J, ntt::em::bx3);
                m_sim.meshblock.metric.v_Cntr2SphCntrv(
                  {i_ + HALF, j_ + HALF}, {Bx1, Bx2, Bx3}, B0sph);
              } else {
                real_t Bx1, Bx2, Bx3;
                // interpolate to cell center
                Bx1 = 0.5
                      * (m_sim.meshblock.em(I + 1, J, ntt::em::bx1)
                         + m_sim.meshblock.em(I, J, ntt::em::bx1));
                Bx2 = 0.5
                      * (m_sim.meshblock.em(I, J + 1, ntt::em::bx2)
                         + m_sim.meshblock.em(I, J, ntt::em::bx2));
                Bx3 = m_sim.meshblock.em(I, J, ntt::em::bx3);
                m_sim.meshblock.metric.v_Cntr2SphCntrv(
                  {i_ + HALF, j_ + HALF}, {Bx1, Bx2, Bx3}, Bsph);
              }
            }
          } else {
            Dsph[0]  = m_sim.meshblock.em(I, J, ntt::em::ex1);
            Dsph[1]  = m_sim.meshblock.em(I, J, ntt::em::ex2);
            Dsph[2]  = m_sim.meshblock.em(I, J, ntt::em::ex3);
            Bsph[0]  = m_sim.meshblock.em(I, J, ntt::em::bx1);
            Bsph[1]  = m_sim.meshblock.em(I, J, ntt::em::bx2);
            Bsph[2]  = m_sim.meshblock.em(I, J, ntt::em::bx3);
            D0sph[0] = m_sim.meshblock.em0(I, J, ntt::em::ex1);
            D0sph[1] = m_sim.meshblock.em0(I, J, ntt::em::ex2);
            D0sph[2] = m_sim.meshblock.em0(I, J, ntt::em::ex3);
            B0sph[0] = m_sim.meshblock.em0(I, J, ntt::em::bx1);
            B0sph[1] = m_sim.meshblock.em0(I, J, ntt::em::bx2);
            B0sph[2] = m_sim.meshblock.em0(I, J, ntt::em::bx3);
          }
          real_t val {ZERO};
          if (m_fields_to_plot[f] == "Dr") {
            val = Dsph[0];
          } else if (m_fields_to_plot[f] == "Dtheta") {
            val = Dsph[1];
          } else if (m_fields_to_plot[f] == "Dphi") {
            val = Dsph[2];
          } else if (m_fields_to_plot[f] == "Br") {
            val = Bsph[0];
          } else if (m_fields_to_plot[f] == "Btheta") {
            val = Bsph[1];
          } else if (m_fields_to_plot[f] == "Bphi") {
            val = Bsph[2];
          } else if (m_fields_to_plot[f] == "Er") {
            val = m_sim.meshblock.aux(I, J, ntt::em::ex1);
          } else if (m_fields_to_plot[f] == "Etheta") {
            val = m_sim.meshblock.aux(I, J, ntt::em::ex2);
          } else if (m_fields_to_plot[f] == "Ephi") {
            val = m_sim.meshblock.aux(I, J, ntt::em::ex3);
          } else if (m_fields_to_plot[f] == "Hr") {
            val = m_sim.meshblock.aux(I, J, ntt::em::bx1);
          } else if (m_fields_to_plot[f] == "Htheta") {
            val = m_sim.meshblock.aux(I, J, ntt::em::bx2);
          } else if (m_fields_to_plot[f] == "Hphi") {
            val = m_sim.meshblock.aux(I, J, ntt::em::bx3);
          } else if (m_fields_to_plot[f] == "D0r") {
            val = D0sph[0];
          } else if (m_fields_to_plot[f] == "D0theta") {
            val = D0sph[1];
          } else if (m_fields_to_plot[f] == "D0phi") {
            val = D0sph[2];
          } else if (m_fields_to_plot[f] == "B0r") {
            val = B0sph[0];
          } else if (m_fields_to_plot[f] == "B0theta") {
            val = B0sph[1];
          } else if (m_fields_to_plot[f] == "B0phi") {
            val = B0sph[2];
          } else if (m_fields_to_plot[f] == "Aphi") {
            val = m_sim.meshblock.aphi(I, J, 0);
          }
          auto idx                                 = Index(i, j);
          (this->fields)[m_fields_to_plot[f]][idx] = val;
#endif
