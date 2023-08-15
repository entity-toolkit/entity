if (document.getElementById("plot_cubed_sphere")) {
  window.addEventListener("load", (event) => {
    let color_bg;
    const getBGColor = () => {
      color_bg = getComputedStyle(document.body).getPropertyValue('--md-default-bg-color');
    }
    const w = document.getElementsByTagName('article')[0].offsetWidth;
    const width = w, height = parseInt(w * 0.75);

    class Rect {
      constructor(p1, p2, p3, p4) {
        this.points = [p1, p2, p3, p4];
      }
      bake(array, col) {
        let rect_arr = [];
        this.points.forEach((p) => {
          rect_arr.push([p.x, p.y, p.z]);
        });
        array.push({ 'type': 'rect', 'mesh': rect_arr, 'color': col });
      }
    }

    class Edge {
      constructor(pts = undefined) {
        if (pts === undefined) {
          this.points = [];
        } else {
          this.points = [...pts];
        }
      }
      bake(array, col = 'black') {
        let edge_arr = [];
        this.points.forEach((p) => {
          edge_arr.push([p.x, p.y, p.z]);
        });
        array.push({ 'type': 'edge', 'mesh': edge_arr, 'color': col });
      }
    }

    class Face {
      constructor(p1, p2, p3, res) {
        this.points = [];
        const a21 = p5.Vector.normalize(p5.Vector.sub(p1, p2));
        const a23 = p5.Vector.normalize(p5.Vector.sub(p3, p2));
        const side = p5.Vector.mag(p5.Vector.sub(p3, p2));
        for (let i = 0; i <= res; i++) {
          let points_i = [];
          for (let j = 0; j <= res; j++) {
            const p_i = p5.Vector.mult(a21, i * (side / res));
            const p_j = p5.Vector.mult(a23, j * (side / res));
            points_i.push(p5.Vector.add(p2, p5.Vector.add(p_i, p_j)));
          }
          this.points.push(points_i);
        }
        this.rects = [];
        this.edges = [new Edge(), new Edge(), new Edge(), new Edge()];
        for (let i = 0; i <= res; i++) {
          for (let j = 0; j <= res; j++) {
            if (i === 0) {
              this.edges[0].points.push(this.points[i][j]);
            }
            if (i === res) {
              this.edges[1].points.push(this.points[i][j]);
            }
            if (j === 0) {
              this.edges[2].points.push(this.points[i][j]);
            }
            if (j === res) {
              this.edges[3].points.push(this.points[i][j]);
            }
            if (i < res && j < res) {
              this.rects.push(
                new Rect(
                  this.points[i][j],
                  this.points[i + 1][j],
                  this.points[i + 1][j + 1],
                  this.points[i][j + 1]
                )
              );
            }
          }
        }
      }
      draw(ctx) {
        this.rects.forEach((r) => r.draw(ctx, ctx));
        this.edges.forEach((edge) => {
          ctx.push();
          ctx.beginShape(ctx.LINES);
          edge.forEach((p) => {
            ctx.vertex(p.x, p.y, p.z);
          });
          ctx.endShape(ctx.CLOSE);
          ctx.pop();
        });
      }
    }

    class Cube {
      constructor(center, side, resolution, ctx) {
        this.faces = [];
        this.center = center;
        const normal_arr = [
          p5.Vector.add(center, ctx.createVector(0, 0, side / 2)), // Front
          p5.Vector.add(center, ctx.createVector(0, 0, -side / 2)), // Back
          p5.Vector.add(center, ctx.createVector(0, side / 2, 0)), // Top
          p5.Vector.add(center, ctx.createVector(0, -side / 2, 0)), // Bottom
          p5.Vector.add(center, ctx.createVector(side / 2, 0, 0)), // Right
          p5.Vector.add(center, ctx.createVector(-side / 2, 0, 0)), // Left
        ];
        const side1_arr = [
          ctx.createVector(0, side / 2, 0),
          ctx.createVector(0, side / 2, 0),
          ctx.createVector(side / 2, 0, 0),
          ctx.createVector(side / 2, 0, 0),
          ctx.createVector(0, side / 2, 0),
          ctx.createVector(0, side / 2, 0),
        ];
        const side2_arr = [
          ctx.createVector(side / 2, 0, 0),
          ctx.createVector(side / 2, 0, 0),
          ctx.createVector(0, 0, side / 2),
          ctx.createVector(0, 0, side / 2),
          ctx.createVector(0, 0, side / 2),
          ctx.createVector(0, 0, side / 2),
        ];

        for (let i = 0; i < normal_arr.length; i++) {
          const normal = normal_arr[i];
          const side1 = side1_arr[i];
          const side2 = side2_arr[i];
          this.faces.push(
            new Face(
              p5.Vector.add(normal, p5.Vector.sub(side1, side2)),
              p5.Vector.add(normal, p5.Vector.add(side1, side2)),
              p5.Vector.add(normal, p5.Vector.sub(side2, side1)),
              resolution
            )
          );
        }
      }
      draw(ctx) {
        this.faces.forEach((f) => {
          f.draw(ctx);
        });
      }
    }

    const isInQuadrant = (p, strong = false) => {
      if (p instanceof p5.Vector) {
        return strong ? (p.x > 0 && p.y < 0 && p.z > 0) : (p.x >= 0 && p.y <= 0 && p.z >= 0);
      } else if (p instanceof Rect) {
        return p.points.every((p) => isInQuadrant(p, strong));
      }
    }

    const getComp = (p, i) => (i % 3 == 0 ? p.x : i % 3 == 1 ? p.y : p.z);

    class CubedSphere {
      constructor(cube, radius) {
        this.center = cube.center;
        this.radius = radius;
        this.grid = this.project_grid(cube);
        this.patches = this.project_patches(cube);
        this.patches = this.patches.map((f) => f.map((e) => new Edge(e)));
        this.slice_points = [];
      }
      project(p) {
        if (p instanceof p5.Vector) {
          return p5.Vector.mult(
            p5.Vector.normalize(p5.Vector.sub(p, this.center)),
            this.radius
          );
        } else {
          throw "Type error";
        }
      }
      project_grid(p) {
        if (p instanceof Cube) {
          return p.faces.map((f) => this.project_grid(f));
        } else if (p instanceof Face) {
          return p.rects.map((r) => this.project_grid(r));
        } else if (p instanceof Rect) {
          const pts = p.points.map((pt) => this.project(pt));
          return new Rect(pts[0], pts[1], pts[2], pts[3]);
        } else {
          throw "Type error";
        }
      }
      project_patches(p) {
        if (p instanceof Cube) {
          return p.faces.map((f) => this.project_patches(f));
        } else if (p instanceof Face) {
          return p.edges.map((e) => this.project_patches(e));
        } else if (p instanceof Edge) {
          return p.points.map((pt) => this.project(pt));
        } else {
          throw "Type error";
        }
      }
      findSlicePoints(cutCondition) {
        this.grid.forEach((face, i) => {
          face.filter((f) => cutCondition(f))
            .forEach((rect) => {
              this.slice_points.push(
                ...rect.points.filter((pt) => pt.x == 0 || pt.y == 0 || pt.z == 0)
              );
            });
        });
        this.slice_points = uniqBy(this.slice_points, (p) =>
          JSON.stringify([p.x, p.y, p.z])
        );
      }
      bake(cutCondition, drawIndent, array) {
        this.findSlicePoints(cutCondition);
        this.grid.forEach((face, i) => {
          face.filter((f) => !cutCondition(f))
            .forEach((rect) => {
              rect.bake(array, sphere_params.palette[i]);
            });
        });
        this.patches.forEach((f) => {
          f.forEach((e) => {
            let e_ = (new Edge(e.points.filter((pt) => !cutCondition(pt, true))))
            e_.bake(array);
            if ((e_.points.length < e.points.length) && (e.points.length > 0) && (drawIndent > 0)) {
              if (cutCondition(e.points[0], true)) {
                (new Edge([e_.points[0], [0, 0, 0]])).bake(array);
              } else {
                (new Edge([e_.points[e_.points.length - 1], [0, 0, 0]])).bake(array);
              }
            }
          });
        });
        if (drawIndent > 0) {
          [0, 1, 2].forEach((C) => {
            let edge = this.slice_points
              .filter((pt) => getComp(pt, C) == 0)
              .sort((p1, p2) => getComp(p1, C + 1) - getComp(p2, C + 1));
            for (let nr = 1; nr <= drawIndent; nr++) {
              let newedge = [];
              for (let i = 0; i < edge.length - 1; i++) {
                let p1 = p5.Vector.mult(
                  p5.Vector.normalize(p5.Vector.sub(edge[i], this.center)),
                  r_grid(nr)
                );
                let p2 = p5.Vector.mult(
                  p5.Vector.normalize(p5.Vector.sub(edge[i + 1], this.center)),
                  r_grid(nr)
                );
                newedge.push(p1);
                if (i == edge.length - 2) {
                  newedge.push(p2);
                }
                let ang1 = p5.Vector.add(edge[i + 1], edge[i])
                  .mult(0.5)
                  .angleBetween(edge[i].copy().set(0, 0, 1));
                let ang2 = p5.Vector.add(edge[i + 1], edge[i])
                  .mult(0.5)
                  .angleBetween(edge[i].copy().set(1, 0, 0));
                let col;
                if (ang1 < Math.PI / 4) {
                  col = newShade(sphere_params.palette[0], -60);
                } else if (ang2 < Math.PI / 4) {
                  col = newShade(sphere_params.palette[4], -60);
                } else {
                  col = newShade(sphere_params.palette[3], -60);
                }
                new Rect(p1, p2, edge[i + 1], edge[i]).bake(array, col);
              }
              edge = newedge;
            }
          });
        }
      }
    }

    const shuffle = (values) => {
      let index = values.length, randomIndex;
      while (index != 0) {
        randomIndex = Math.floor(Math.random() * index);
        index--;
        [values[index], values[randomIndex]] = [
          values[randomIndex], values[index]];
      }
      return values;
    }

    const newShade = (hexColor, magnitude) => {
      hexColor = hexColor.replace(`#`, ``);
      if (hexColor.length === 6) {
        const decimalColor = parseInt(hexColor, 16);
        let r = (decimalColor >> 16) + magnitude;
        r > 255 && (r = 255);
        r < 0 && (r = 0);
        let g = (decimalColor & 0x0000ff) + magnitude;
        g > 255 && (g = 255);
        g < 0 && (g = 0);
        let b = ((decimalColor >> 8) & 0x00ff) + magnitude;
        b > 255 && (b = 255);
        b < 0 && (b = 0);
        return `#${(g | (b << 8) | (r << 16)).toString(16)}`;
      } else {
        return hexColor;
      }
    };

    const uniqBy = (a, key) => {
      var seen = {};
      return a.filter(function (item) {
        var k = key(item);
        return seen.hasOwnProperty(k) ? false : (seen[k] = true);
      });
    }

    let cube_params = {
      side: 100,
      res: 16,
    };

    let sphere_params = {
      radius: 350,
      rmin: 100,
      nindent: 12,
      palette: ["#d37b94", "#e89f94", "#edd29e", "#c0ccb9", "#80b7c0", "#a097a1"]
    };

    const r_grid = (n) => {
      return (
        sphere_params.radius *
        (sphere_params.rmin / sphere_params.radius) ** (n / sphere_params.nindent)
      );
    };

    // let cube, cubed_sphere, cubed_sphere_in, cam;

    const draw_baked = (ctx, data) => {
      data.forEach((obj) => {
        if (obj.type === 'rect') {
          ctx.fill(obj.color);
          ctx.beginShape(ctx.QUADS);
          obj.mesh.forEach((p) => {
            ctx.vertex(p[0], p[1], p[2]);
          });
          ctx.endShape(ctx.CLOSE);
        } else if (obj.type === 'edge') {
          ctx.push();
          ctx.noFill();
          ctx.strokeWeight(4);
          ctx.stroke(23);
          ctx.beginShape();
          obj.mesh.forEach((p) => {
            ctx.vertex(p[0], p[1], p[2]);
          });
          ctx.endShape();
          ctx.pop();
        }
      });
    }

    const initialize = (ctx, baked_mesh) => {
      const cube = new Cube(
        ctx.createVector(0, 0, 0),
        cube_params.side,
        cube_params.res,
        ctx
      );
      const cubed_sphere = new CubedSphere(cube, sphere_params.radius);
      const cubed_sphere_in = new CubedSphere(cube, r_grid(sphere_params.nindent));

      baked_mesh.length = 0;
      cubed_sphere.bake(isInQuadrant, sphere_params.nindent, baked_mesh);
      cubed_sphere_in.bake((f) => !isInQuadrant(f), 0, baked_mesh);
    };

    const frame = (ctx, baked_mesh) => {
      getBGColor();
      ctx.background(color_bg);
      ctx.orbitControl(1, 1, 0.1);
      ctx.push();
      ctx.lights();
      ctx.ambientMaterial(240);
      ctx.shadedModelColors();
      draw_baked(ctx, baked_mesh);
      ctx.pop();
    };

    const createSliderInGroup = (ctx, panel, label, value, min, max, step, callback_slider, callback_input) => {
      const slider_tr = ctx.createElement("tr")
        .parent(panel);
      ctx.createElement("th", label).parent(slider_tr);
      let slider, input;
      slider = ctx.createSlider(min, max, value, step)
        .parent(ctx.createElement("td").parent(slider_tr))
        .input(() => callback_slider(slider, input));
      input = ctx.createInput(String(value), 'number')
        .parent(ctx.createElement("td").parent(slider_tr))
        .input(() => callback_input(input, slider));
    };

    const sketch = (ctx) => {
      let baked_mesh = [];
      let cam;

      ctx.setup = () => {
        /* ---------------------------- parameter sliders --------------------------- */
        const panel1 = ctx.createElement("tbody")
          .parent(ctx.createElement("table")
            .parent('plot_cubed_sphere')
            .style('display', 'inline-block')
            .style('margin-right', '25px'));
        const panel2 = ctx.createElement("tbody")
          .parent(ctx.createElement("table")
            .parent('plot_cubed_sphere')
            .style('display', 'inline-block'));
        createSliderInGroup(ctx, panel1, "θϕ resolution (even):", cube_params.res, 2, 64, 2,
          (slider, input) => {
            if (slider.value() > 64) slider.value(64);
            cube_params.res = slider.value();
            input.value(slider.value());
            initialize(ctx, baked_mesh);
            frame(ctx, baked_mesh);
          },
          (input, slider) => {
            if (input.value() > 64) input.value(64);
            let val = 2 * Math.floor(input.value() / 2);
            cube_params.res = val;
            slider.value(val);
            initialize(ctx, baked_mesh);
            frame(ctx, baked_mesh);
          });
        createSliderInGroup(ctx, panel1, "r resolution:", sphere_params.nindent, 2, 64, 1,
          (slider, input) => {
            if (slider.value() > 64) slider.value(64);
            sphere_params.nindent = slider.value();
            input.value(slider.value());
            initialize(ctx, baked_mesh);
            frame(ctx, baked_mesh);
          },
          (input, slider) => {
            if (input.value() > 64) input.value(64);
            sphere_params.nindent = input.value();
            slider.value(input.value());
            initialize(ctx, baked_mesh);
            frame(ctx, baked_mesh);
          });
        createSliderInGroup(ctx, panel2, "rmin:", sphere_params.rmin / sphere_params.radius, 0.025, 0.95, 0.001,
          (slider, input) => {
            if (slider.value() > 1) slider.value(0.95);
            sphere_params.rmin = slider.value() * sphere_params.radius;
            input.value(slider.value());
            initialize(ctx, baked_mesh);
            frame(ctx, baked_mesh);
          },
          (input, slider) => {
            if (input.value() > 1) input.value(0.95);
            sphere_params.rmin = input.value() * sphere_params.radius;
            slider.value(input.value());
            initialize(ctx, baked_mesh);
            frame(ctx, baked_mesh);
          });

        /* ------------------------------- scene setup ------------------------------ */
        sphere_params.palette = shuffle(sphere_params.palette);
        const cnv = ctx.createCanvas(width, height, ctx.WEBGL);
        cnv.parent('plot_cubed_sphere');
        initialize(ctx, baked_mesh);

        cam = ctx.createCamera();
        cam.perspective(ctx.radians(35));
        cam.setPosition(117.1283668639852, -502.43968432573826, 1033.8049238911865);
        cam.lookAt(0, 0, 0);
        ctx.smooth();

        /* --------------------------- save as png button --------------------------- */
        const savePNGbutton = ctx.createButton('Save as png');
        savePNGbutton.parent('plot_cubed_sphere');
        savePNGbutton.style('width', '120px');
        savePNGbutton.mousePressed(() => {
          ctx.pixelDensity(3.0);
          frame(ctx, baked_mesh);
          ctx.save("cubed-sphere-3d.png");
          ctx.pixelDensity();
        });
      };
      ctx.draw = () => {
        ctx.frameRate(10);
        frame(ctx, baked_mesh);
      };
    };

    let p5_sketch = new p5(sketch, "sketch");
  });
}