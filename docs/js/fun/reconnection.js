const ifDocumentContains = (id, callback) => {
  const element = document.getElementById(id)
  if (element) {
    callback(element)
  }
}

ifDocumentContains("plot_reconnection", e => {

  window.addEventListener("load", (event) => {
    const w = document.getElementsByTagName('article')[0].offsetWidth;
    let color_fg, color_bg;

    const getColors = () => {
      color_bg = getComputedStyle(document.body).getPropertyValue('--md-default-bg-color');
      color_fg = getComputedStyle(document.body).getPropertyValue('--md-default-fg-color');
      color_fg = (color_fg.split(',').length == 4) ? color_fg.split(',').slice(0, 3).join() + ',1)' : color_fg;
    }

    const width = w, height = 600;
    const configs = {
      y0: height / 2,
      dr: 0.025,
      dt: 2,
      c: 1,
    };

    class Plasmoid {
      constructor(x0, r0) {
        this.r = r0;
        this.x = x0;
        this.u = 10.0 * 2.0 * (Math.random() - 0.5);
      }

      move() {
        let v = this.u / Math.sqrt(1.0 + this.u ** 2)
        this.x += v * configs.dt;
      }
      grow() {
        let momentum = this.r * this.u;
        this.r += configs.dr * configs.dt;
        this.u = momentum / (this.r + 0.1);
      }

      draw(p5inst) {
        p5inst.noStroke();
        p5inst.fill(color_fg);
        p5inst.ellipse(this.x, configs.y0, 2 * this.r, 1.5 * this.r);
      }
    }
    class CurrentSheet {
      constructor(N0) {
        this.plasmoids = Array.from({ length: N0 }, () => new Plasmoid(Math.random() * width, 2));
      }

      grow() {
        this.plasmoids.forEach((p) => {
          p.grow()
        });
      }
      move() {
        this.plasmoids.forEach((p) => {
          p.move()
        });
      }
      draw(p5inst) {
        p5inst.strokeWeight(2);
        p5inst.stroke(color_fg);
        p5inst.line(0, configs.y0, width, configs.y0);
        this.plasmoids.forEach((p) => {
          p.draw(p5inst)
        });
      }
      merge() {
        for (let i = 0; i < this.plasmoids.length; i++) {
          for (let j = i + 1; j < this.plasmoids.length; j++) {
            let p1 = this.plasmoids[i];
            let p2 = this.plasmoids[j];
            if (Math.abs(p1.x - p2.x) < 0.5 * Math.min(p1.r, p2.r) + Math.max(p1.r, p2.r)) {
              if (p1.r > p2.r) {
                [p1, p2] = [p2, p1];
              }
              let momentum = p1.r * p1.u + p2.r * p2.u;
              p2.r = Math.sqrt(p1.r ** 2 + p2.r ** 2);
              p2.u = momentum / (p2.r + 1);
              p1.x = -10000;
            }
          }
        }
      }
      outflow() {
        this.plasmoids = this.plasmoids.filter(p => ((p.x > -p.r) && (p.x < width + p.r)));
      }
      reconnect() {
        let pl = 0.0;
        this.plasmoids.forEach((p) => {
          if (p.x - p.r < 0) {
            pl += p.x + p.r
          } else if (p.x + p.r > width) {
            pl += width - (p.x - p.r);
          } else {
            pl += 2 * p.r;
          }
        });
        let nnew = parseInt(Math.abs(width - pl) / 100);
        for (let i = 0; i < nnew; i++) {
          let occupied = true;
          let xnew, cntr = 0;
          while (occupied && cntr < 100) {
            xnew = Math.random() * width;
            let canExit = true;
            this.plasmoids.forEach((p) => {
              if (Math.abs(xnew - p.x) < p.r) {
                // occupied = true
                canExit = false;
                return null;
              }
            });
            cntr++;
            occupied = !canExit;
          }
          this.plasmoids.push(new Plasmoid(xnew, 5));
        }
      }
    }
    let cs;
    const sketch = (p) => {
      p.setup = () => {
        cnv = p.createCanvas(width, height);
        cnv.parent('plot_reconnection');
        cs = new CurrentSheet(10);
      };

      p.draw = () => {
        getColors();
        p.background(color_bg);
        cs.reconnect();
        cs.grow();
        cs.merge();
        cs.outflow();
        cs.move();
        cs.draw(p);
      }

    };

    const myp5 = new p5(sketch);

  }, false);

})