---
hide:
  - footer
---

## Development status

__PIC__

- [x] spherical/qspherical metrics (2D)
- [x] minkowski field solver (1D/2D/3D)
- [x] curvilinear field solver (2D)
- [x] minkowski particle pusher (Boris; 1D/2D/3D)
- [x] curvilinear particle pusher (Boris; 2D)
- [x] minkowski current deposition (1D/2D/3D)
- [x] curvilinear current deposition (2D)
- [ ] cubed sphere metric (3D)

__GRPIC__

- [x] spherical/qspherical Kerr-Schild metrics (2D)
- [x] field solver (2D)
- [ ] particle pusher (1D/2D/3D)
- [ ] current deposition (2D)
- [ ] cartesian Kerr-Schild metrics (1D/2D/3D)

__Known bugs / minor issues to fix__

- [ ] `$(CURDIR)` seems to fail in some instances (need a more robust apprch)
- [x] check python `subprocess.run` command during the configure stage
- [ ] check if compilation of `glfw` is possible (or if `glfw` is available)
- [ ] same for `freetype`
- [ ] clarify `nttiny_path` w.r.t. what (maybe add an error messages in configure script)